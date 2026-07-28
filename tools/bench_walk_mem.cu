// Measures the memory ceiling for the PoM walk's access pattern, with none of the walk's math.
//
// WHY: the miner sustains ~315 GB/s of chunk reads on a 5070 Ti (38.4 Mnonce/s x 256 steps x 32 B)
// against an 896 GB/s spec. Every software lever we control measures flat -- ILP x2, block size
// 64..1024, removing the 64-bit modulo from the dependency chain -- and the kernel already runs at
// maximum occupancy (26 registers, zero spill). That is consistent with being at a hardware
// ceiling, but "consistent with" is not a measurement. This measures it.
//
// ---------------------------------------------------------------------------------------------
// MEASURED 2026-07-28 -- RTX 5070 Ti (sm_120, 70 SMs), CUDA 13.x toolkit, driver API 13.3
//   ./bench_walk_mem      # 258040832 chunks (8.26 GB), 1<<20 threads, 256 steps
//
//     chase (dependent)   : 343.6 GB/s  (25.00 ms)
//     indep (independent) : 317.6 GB/s  (27.05 ms)
//     chase + tensor srch : 367.5 GB/s  (23.38 ms)
//     miner, same card    : ~315   GB/s
//
//   * ~343-367 GB/s is the CEILING for this access pattern -- roughly 38-41% of the 896 GB/s
//     datasheet figure. Random 32-byte reads over 8 GB do not reach sequential peak: that is DRAM
//     row cycling and access granularity, not an inefficiency anywhere in the kernel.
//   * indep <= chase, so memory-level parallelism is NOT the limit. Independent access with many
//     misses in flight per thread buys nothing over a strict pointer chase, and anything that adds
//     MLP (ILP variants, higher occupancy, more threads) therefore cannot help.
//   * The TENSOR LOOKUP IS FREE. chase+search touches identical addresses in identical order --
//     bases[lo] + 2*local resolves to exactly buf + 2*idx -- yet doing strictly MORE work came out
//     7% ahead of the plain chase. No confident explanation; the likeliest is that the extra ~9
//     cached loads space out the DRAM requests and a less bursty stream hits fewer bank/row
//     conflicts, which would also explain indep (the burstiest) coming in lowest. Either way the
//     conclusion does not rest on the explanation: the ~9 dependent prefix[] loads are not what
//     separates the miner from the ceiling.
//
//   That closes the last open idea. A bucket index (host-built chunk-range -> tensor table,
//   collapsing the binary search to one load plus a short scan) was the remaining candidate and
//   would have been the most invasive edit considered, in consensus-critical code. TESTED AND
//   REJECTED here -- do not rebuild the case for it without a measurement that contradicts this.
//   What is left between the miner and the ceiling is the four chained mix64, which consensus
//   fixes and no implementation may change.
//
//   Consistent with every other measurement on this card: ILP x2 flat (38.4 vs 38.6), block size
//   64/128/256 flat (three different winners, same throughput), replacing the 64-bit modulo with a
//   verified reciprocal flat (reverted in effd557), and the kernel already at maximum occupancy
//   (26 registers, zero spill -- see -Xptxas -v in the build log).
//
//   PoM kernel optimisation is CLOSED except for the open question below. The levers that move
//   this workload are memory clock and model tier, neither of which lives in the kernel.
// ---------------------------------------------------------------------------------------------
//
// Three kernels, same 32-byte random reads over the same ~8 GB footprint:
//
//   chase        DEPENDENT. The next index is READ from the chunk, so each thread has exactly one
//                outstanding miss at a time -- the walk's shape, since its next offset cannot be
//                known until the current chunk arrives. No modulo, no search, no mix64: pure
//                memory. This is what sets the ceiling above.
//
//   indep        INDEPENDENT. The index is COMPUTED, so many misses can be in flight per thread.
//
//   chase+search chase plus the miner's tensor lookup: binary search prefix[], index off bases[].
//                Still no mix64, no modulo, and the SAME addresses in the same order as chase --
//                so the delta against chase is purely the cost of the lookup. It came out
//                negative, which is how the bucket-index idea was ruled out.
//
// Build (nvcc that knows your arch -- 12.8+ for sm_120):
//   nvcc -O3 -arch=sm_120 tools/bench_walk_mem.cu -o bench_walk_mem
// Run (needs ~8.3 GB free, so stop the miner or point it at an idle GPU):
//   ./bench_walk_mem            # defaults to the tier-2 chunk count
//   ./bench_walk_mem <chunks> <threads> <steps> <device> <tensors>

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

// chase + the miner's tensor lookup, and nothing else. Same pointer chase, but the chunk address
// is resolved the way pom_mine resolves it: binary search prefix[] for the owning tensor, then
// index off bases[]. No mix64, no modulo -- so the delta against `chase` is the cost of the
// ~9 dependent prefix[] loads plus the bases[] load, isolated from every other difference.
//
// Result: 367.5 vs 343.6 GB/s, i.e. the lookup is free (see MEASURED above). Kept in the file
// because that is the evidence retiring the bucket-index idea, and a negative result is only
// worth anything if the experiment that produced it can be re-run.
__global__ void chase_search(const ulonglong2* buf, unsigned int K, unsigned long long* out,
                             unsigned long long n, const unsigned long long* prefix,
                             const unsigned long long* bases, unsigned int T) {
    unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    unsigned long long idx = mix64(tid) % n;
    unsigned long long acc = 0;
    for (unsigned int k = 0; k < K; k++) {
        unsigned int lo = 0, hi = T;
        while (lo + 1 < hi) {
            unsigned int mid = (lo + hi) >> 1;
            if (prefix[mid] <= idx) lo = mid; else hi = mid;
        }
        unsigned long long local = idx - prefix[lo];
        const ulonglong2* p = (const ulonglong2*)bases[lo];
        ulonglong2 a = p[2 * local];
        ulonglong2 c = p[2 * local + 1];
        acc ^= a.y ^ c.x ^ c.y;
        idx = a.x;
    }
    if (acc == 0xDEADBEEFULL) out[0] = idx;
}

static double gbps(double bytes, float ms) { return bytes / (ms * 1e-3) / 1e9; }

int main(int argc, char** argv) {
    unsigned long long n       = (argc > 1) ? strtoull(argv[1], nullptr, 10) : 258040832ULL;
    unsigned long long threads = (argc > 2) ? strtoull(argv[2], nullptr, 10) : (1ULL << 20);
    unsigned int       K       = (argc > 3) ? (unsigned)atoi(argv[3]) : 256u;
    int                dev     = (argc > 4) ? atoi(argv[4]) : 0;
    unsigned int       T       = (argc > 5) ? (unsigned)atoi(argv[5]) : 400u;   // tensor count

    CK(cudaSetDevice(dev));
    cudaDeviceProp p{};
    CK(cudaGetDeviceProperties(&p, dev));

    // No spec-peak lookup: cudaDeviceProp::memoryClockRate was deprecated in CUDA 12 and REMOVED
    // in 13, so reading it will not compile on a current toolkit. Only these four fields are
    // relied on. Compare the absolute GB/s below against the card's datasheet figure.
    size_t bytes = (size_t)n * 32;
    printf("device      : %s (sm_%d%d, %d SMs)\n", p.name, p.major, p.minor, p.multiProcessorCount);
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

    // Tensor table for chase_search: T contiguous spans over the same buffer. Real tensors vary in
    // size, but the search cost is log2(T) either way, and T is what sets that.
    unsigned long long* prefix_d = nullptr;
    unsigned long long* bases_d = nullptr;
    {
        unsigned long long* hp = (unsigned long long*)malloc((T + 1) * sizeof(unsigned long long));
        unsigned long long* hb = (unsigned long long*)malloc(T * sizeof(unsigned long long));
        for (unsigned int j = 0; j <= T; j++) hp[j] = (unsigned long long)((__int128)n * j / T);
        for (unsigned int j = 0; j < T; j++)
            hb[j] = (unsigned long long)(uintptr_t)(buf + 2 * hp[j]);
        CK(cudaMalloc(&prefix_d, (T + 1) * sizeof(unsigned long long)));
        CK(cudaMalloc(&bases_d, T * sizeof(unsigned long long)));
        CK(cudaMemcpy(prefix_d, hp, (T + 1) * sizeof(unsigned long long), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(bases_d, hb, T * sizeof(unsigned long long), cudaMemcpyHostToDevice));
        free(hp);
        free(hb);
    }

    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));
    const unsigned int block = 256;
    unsigned int grid = (unsigned int)((threads + block - 1) / block);
    double moved = (double)threads * K * 32.0;

    for (int which = 0; which < 3; which++) {
        const char* name = which == 0 ? "chase (dependent)  "
                         : which == 1 ? "indep (independent)"
                                      : "chase + tensor srch";
        float best = 1e30f;
        for (int rep = 0; rep < 4; rep++) {   // first rep warms caches/TLB; keep the best of the rest
            CK(cudaEventRecord(t0));
            if (which == 0)      chase<<<grid, block>>>(buf, K, out, n);
            else if (which == 1) indep<<<grid, block>>>(buf, K, out, mask);
            else                 chase_search<<<grid, block>>>(buf, K, out, n, prefix_d, bases_d, T);
            CK(cudaEventRecord(t1));
            CK(cudaEventSynchronize(t1));
            CK(cudaGetLastError());
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, t0, t1));
            if (rep > 0 && ms < best) best = ms;
        }
        printf("%s : %7.1f GB/s  (%6.2f ms)\n", name, gbps(moved, best), best);
    }

    printf("\nminer for comparison: ~315 GB/s at 38.4 Mnonce/s x 256 steps x 32 B\n");
    cudaFree(prefix_d);
    cudaFree(bases_d);
    cudaFree(buf);
    cudaFree(out);
    return 0;
}
