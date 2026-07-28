// Measures the memory ceiling for the PoM walk's access pattern, with none of the walk's math.
//
// WHY: the miner sustains ~315 GB/s of chunk reads on a 5070 Ti (38.4 Mnonce/s x 256 steps x 32 B)
// against an 896 GB/s spec. Every software lever we control measures flat -- ILP x2, block size
// 64..1024, removing the 64-bit modulo from the dependency chain -- and the kernel already runs at
// maximum occupancy (26 registers, zero spill). That is consistent with being at a hardware
// ceiling, but "consistent with" is not a measurement. This measures it.
//
// Two kernels, same 32-byte random reads over the same ~8 GB footprint:
//
//   chase  DEPENDENT. The next index is READ from the chunk, so each thread has exactly one
//          outstanding miss at a time -- the same shape as the walk, whose next offset cannot be
//          known until the current chunk comes back. No modulo, no binary search, no mix64 in the
//          loop: pure memory.
//
//   indep  INDEPENDENT. The index is COMPUTED, so a thread can have many misses in flight at once.
//          This is what the memory system can do when the dependency chain is removed.
//
// Reading the result against the miner's ~315 GB/s:
//
//   chase ~= 315          the walk is at the dependent-access ceiling; the kernel is done.
//   chase >> 315          the walk's own math is costing throughput after all -- go look again.
//   indep >> chase        the limit is the serial chain (outstanding misses per thread), not
//                         granularity or DRAM efficiency. More memory-level parallelism would pay,
//                         which is worth knowing since ILP x2 keeps total MLP flat by halving the
//                         thread count -- it never actually tested this.
//   indep ~= chase        the limit is the memory system itself (access granularity, row
//                         cycling, TLB), and nothing in software reaches it.
//
// Build (nvcc that knows your arch -- 12.8+ for sm_120):
//   nvcc -O3 -arch=sm_120 tools/bench_walk_mem.cu -o bench_walk_mem
// Run (needs ~8.3 GB free, so stop the miner or point it at an idle GPU):
//   ./bench_walk_mem            # defaults to the tier-2 chunk count
//   ./bench_walk_mem <chunks> <threads> <steps> <device>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error at line %d: %s\n", __LINE__, cudaGetErrorString(e_)); return 1; } } while (0)

__device__ __forceinline__ unsigned long long mix64(unsigned long long x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

// chunk i is two ulonglong2 at buf[2i], buf[2i+1]; word0 holds the next index for the chase.
__global__ void fill(ulonglong2* buf, unsigned long long n) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long stride = gridDim.x * (unsigned long long)blockDim.x;
    for (; i < n; i += stride) {
        unsigned long long nxt = mix64(i) % n;
        buf[2 * i]     = make_ulonglong2(nxt, i * 3 + 1);
        buf[2 * i + 1] = make_ulonglong2(i * 5 + 2, i * 7 + 3);
    }
}

__global__ void chase(const ulonglong2* buf, unsigned int K,
                      unsigned long long* out, unsigned long long n) {
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long idx = mix64(tid) % n;
    unsigned long long acc = 0;
    for (unsigned int k = 0; k < K; k++) {
        ulonglong2 a = buf[2 * idx];
        ulonglong2 c = buf[2 * idx + 1];
        acc ^= a.y ^ c.x ^ c.y;   // consume the other 24 bytes so the read cannot be elided
        idx = a.x;                // dependent: next address comes from memory
    }
    if (acc == 0xDEADBEEFULL) out[0] = idx;   // never true; keeps the loop live
}

__global__ void indep(const ulonglong2* buf, unsigned int K,
                      unsigned long long* out, unsigned long long mask) {
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long s = mix64(tid);
    unsigned long long acc = 0;
    for (unsigned int k = 0; k < K; k++) {
        // Index is computed, and masked rather than reduced, so the address does not depend on
        // any load and the ALU cost stays negligible.
        unsigned long long idx = mix64(s + k) & mask;
        ulonglong2 a = buf[2 * idx];
        ulonglong2 c = buf[2 * idx + 1];
        acc ^= a.x ^ a.y ^ c.x ^ c.y;
    }
    if (acc == 0xDEADBEEFULL) out[0] = acc;
}

static double gbps(double bytes, float ms) { return bytes / (ms * 1e-3) / 1e9; }

int main(int argc, char** argv) {
    unsigned long long n       = (argc > 1) ? strtoull(argv[1], nullptr, 10) : 258040832ULL;
    unsigned long long threads = (argc > 2) ? strtoull(argv[2], nullptr, 10) : (1ULL << 20);
    unsigned int       K       = (argc > 3) ? (unsigned)atoi(argv[3]) : 256u;
    int                dev     = (argc > 4) ? atoi(argv[4]) : 0;

    CK(cudaSetDevice(dev));
    cudaDeviceProp p{};
    CK(cudaGetDeviceProperties(&p, dev));

    // Peak = bus width x memory clock x 2 (DDR). Reported for context only.
    double peak = (double)p.memoryBusWidth / 8.0 * (double)p.memoryClockRate * 1e3 * 2.0 / 1e9;
    size_t bytes = (size_t)n * 32;
    printf("device      : %s (sm_%d%d, %d SMs)\n", p.name, p.major, p.minor, p.multiProcessorCount);
    printf("spec peak   : %.0f GB/s\n", peak);
    printf("footprint   : %.2f GB (%llu chunks x 32 B)\n", bytes / 1e9, n);
    printf("launch      : %llu threads x %u steps\n\n", threads, K);

    ulonglong2* buf = nullptr;
    if (cudaMalloc(&buf, bytes) != cudaSuccess) {
        printf("cudaMalloc of %.2f GB failed - free some VRAM (stop the miner) or lower <chunks>\n",
               bytes / 1e9);
        return 1;
    }
    unsigned long long* out = nullptr;
    CK(cudaMalloc(&out, sizeof(unsigned long long)));

    fill<<<4096, 256>>>(buf, n);
    CK(cudaDeviceSynchronize());

    unsigned long long mask = 1;
    while (mask * 2 <= n) mask *= 2;
    mask -= 1;   // largest power-of-two footprint that fits, so `indep` can mask instead of divide

    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));
    const unsigned int block = 256;
    unsigned int grid = (unsigned int)((threads + block - 1) / block);
    double moved = (double)threads * K * 32.0;

    for (int which = 0; which < 2; which++) {
        const char* name = which == 0 ? "chase (dependent)  " : "indep (independent)";
        float best = 1e30f;
        for (int rep = 0; rep < 4; rep++) {   // first rep warms caches/TLB; keep the best of the rest
            CK(cudaEventRecord(t0));
            if (which == 0) chase<<<grid, block>>>(buf, K, out, n);
            else            indep<<<grid, block>>>(buf, K, out, mask);
            CK(cudaEventRecord(t1));
            CK(cudaEventSynchronize(t1));
            CK(cudaGetLastError());
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, t0, t1));
            if (rep > 0 && ms < best) best = ms;
        }
        printf("%s : %7.1f GB/s  (%.1f%% of spec, %6.2f ms)\n",
               name, gbps(moved, best), 100.0 * gbps(moved, best) / peak, best);
    }

    printf("\nminer for comparison: ~315 GB/s at 38.4 Mnonce/s x 256 steps x 32 B\n");
    cudaFree(buf);
    cudaFree(out);
    return 0;
}
