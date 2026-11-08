#!/usr/bin/env bash
# Builds cuda/pom_mine.cu into pom_mine_nextgen.fatbin (repo root), in Docker.
#
# Every image but one comes from the 12.8 toolkit: sm_89/90/100/120 SASS and compute_120 PTX
# (ISA 8.7), so a driver as old as r570 loads them. A 12.9 build would raise that PTX to ISA 8.8
# (r575+), and r570 Blackwell rigs then lost the nextgen image. sm_103 (B300 class) alone needs
# the 12.9 toolkit; those cards need r575+ anyway. The 12.9 fatbinary packs all of them.
#
# The gencode set must cover every cc that pom_gpu.rs::is_nextgen_device routes here (major > 8,
# or 8.9), including sm_90: compute_120 PTX cannot JIT *down* to Hopper.
#
# Usage: pom-nextgen-fatbin.sh <cuda-12.8 devel image> <cuda-12.9 devel image>
set -euo pipefail
OLD_IMAGE=${1:?cuda 12.8 devel image}
NEW_IMAGE=${2:?cuda 12.9 devel image}
WORK=fatbin-work
rm -rf "$WORK" && mkdir -p "$WORK"

docker run --rm -v "$PWD":/src -w /src "$OLD_IMAGE" bash -euo pipefail -c "
  for sm in 89 90 100 120; do
    nvcc -cubin -O3 -Xptxas -v -arch=sm_\$sm cuda/pom_mine.cu -o $WORK/sm_\$sm.cubin
  done
  nvcc -ptx -O3 -arch=compute_120 cuda/pom_mine.cu -o $WORK/compute_120.ptx
  grep -m1 '^\.version' $WORK/compute_120.ptx
"

docker run --rm -v "$PWD":/src -w /src "$NEW_IMAGE" bash -euo pipefail -c "
  nvcc -cubin -O3 -Xptxas -v -arch=sm_103 cuda/pom_mine.cu -o $WORK/sm_103.cubin
  fatbinary -64 --create=pom_mine_nextgen.fatbin \
    --image3=kind=elf,sm=89,file=$WORK/sm_89.cubin \
    --image3=kind=elf,sm=90,file=$WORK/sm_90.cubin \
    --image3=kind=elf,sm=100,file=$WORK/sm_100.cubin \
    --image3=kind=elf,sm=103,file=$WORK/sm_103.cubin \
    --image3=kind=elf,sm=120,file=$WORK/sm_120.cubin \
    --image3=kind=ptx,sm=120,file=$WORK/compute_120.ptx
  cuobjdump --list-elf --list-ptx pom_mine_nextgen.fatbin
"

sudo chown "$(id -u):$(id -g)" pom_mine_nextgen.fatbin
rm -rf "$WORK"
ls -la pom_mine_nextgen.fatbin
