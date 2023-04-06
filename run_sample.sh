#!/bin/sh
module load CUDA/10.1.243-GCC-8.3.0
nvcc sample.cu -o sample
srun --reservation=fri -G1 -n1 sample
