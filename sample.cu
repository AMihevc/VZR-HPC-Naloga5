#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// zaglavja za cuda funkcije in strukture
#include <cuda.h>
#include <cuda_runtime.h>

// v pomoč pri debugganju kode
#include "helper_cuda.h"

#define BLOCK_SIZE 16
//#define UNIFIED_MEMORY

// CUDA kernel 
// predpona __global__ pomeni da gre na graficno drugace je obicajna funkcija
__global__ void printGPU(unsigned char *text)
{  
    // text je pointer na spomin na graficni kartici

    //thread id
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // glej predavanja kako to določiš ampak je itak vedno isto
    if (tid==0)
    {
        printf("(tid: %d) %s",tid , text);
    }
}

#ifndef UNIFIED_MEMORY
char h_text[] = "Hello from GPU!\n";
// Allocate memory on the GPU
#else
//Unified memory address
__managed__ unsigned char h_text[] = "Hello from GPU!\n";
//For dynamic allocation: cudaMallocManaged()
#endif

int main(void)
{
   
    // Set the thread execution grid (1 block)
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize(1);

    // Create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start
    cudaEventRecord(start);

    #ifndef UNIFIED_MEMORY
    // Copy memory to GPU
    unsigned char *d_text;
    checkCudaErrors(cudaMalloc(&d_text, sizeof(h_text)));
    checkCudaErrors(cudaMemcpy(d_text, h_text, sizeof(h_text), cudaMemcpyHostToDevice));
    #endif

    // Run function on the GPU
    #ifndef UNIFIED_MEMORY
    printGPU<<<gridSize, blockSize>>>(d_text);
    #else
    printGPU<<<gridSize, blockSize>>>(h_text);
    #endif
    getLastCudaError("printGPU() execution failed\n");

    // Record stop
    cudaEventRecord(stop);

    // Wait for all events to finish
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel & Memcpy Execution time is: %0.3f milliseconds \n", milliseconds);

    // Free memory
    #ifndef UNIFIED_MEMORY
    checkCudaErrors(cudaFree(d_text));
    #endif
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    return 0;
}
