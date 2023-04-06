#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

// v pomoč pri debugganju kode
#include "helper_cuda.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define BINS 256
#define BLOCK_SIZE 16 //TODO this probably needs to be changed to 256 or 512

unsigned int *h_hist;
unsigned int *d_histGPU;

void histogramCPU(unsigned char *imageIn,
                  unsigned int *hist,
                  int width, int height, int cpp)
{
    // Each color channel is 1 byte long, there are 4 channels RED, BLUE, GREEN,  and ALPHA
    // The order is RED|GREEN|BLUE|ALPHA for each pixel, we ignore the ALPHA channel when computing the histograms
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
        {
            hist[imageIn[(i * width + j) * cpp]]++;                // RED
            hist[imageIn[(i * width + j) * cpp + 1] + BINS]++;     // GREEN
            hist[imageIn[(i * width + j) * cpp + 2] + 2 * BINS]++; // BLUE
        }
} // end of histogramCPU


void printHistogram(unsigned int *hist)
{
    printf("Colour\tNo. Pixels\n");
    for (int i = 0; i < BINS; i++)
    {
        if (hist[i] > 0)
            printf("%dR\t%d\n", i, hist[i]);
        if (hist[i + BINS] > 0)
            printf("%dG\t%d\n", i, hist[i + BINS]);
        if (hist[i + 2 * BINS] > 0)
            printf("%dB\t%d\n", i, hist[i + 2 * BINS]);
    }
}


//TODO write a GPU kernel to compute the histogram on the GPU
__global__ void histogramGPU(unsigned char *imageIn,
                             unsigned int *hist,
                             int width, int height, int cpp)
{
    // Each color channel is 1 byte long, there are 4 channels RED, BLUE, GREEN,  and ALPHA
    // The order is RED|GREEN|BLUE|ALPHA for each pixel, we ignore the ALPHA channel when computing the histograms

    //TODO calculate global and local id ant use them to calculate the pixel index
    int tid_global_i = blockDim.x * blockIdx.x + threadIdx.x;
    int tid_global_j = blockDim.y * blockIdx.y + threadIdx.y;

    int tid_local_i = threadIdx.x;
    int tid_local_j = threadIdx.y; 

    //initialize shared memory of each chanel R G B for each block of threads 
    __shared__ unsigned int local_histR[BINS];
    __shared__ unsigned int local_histG[BINS];
    __shared__ unsigned int local_histB[BINS];

    //initialize the tables to zero 
    local_histR[tid_local_i * blockDim.x + tid_local_j] = 0;
    local_histG[tid_local_i * blockDim.x + tid_local_j] = 0;
    local_histB[tid_local_i * blockDim.x + tid_local_j] = 0;

    // wait for all threads in block to finish
    __syncthreads();


    //

    //TODO check that threads dont check pixels "outside" the image
    if ( tid_global_i < height && tid_global_j < width) {

        atomicInc(&local_histR[imageIn[(tid_local_i * width + tid_local_j) * cpp]],1);                  // RED
        atomicInc(&local_histG[imageIn[(tid_local_i * width + tid_local_j) * cpp + 1] + BINS],1);       // GREEN
        atomicInc(&local_histB[imageIn[(tid_local_i * width + tid_local_j) * cpp + 2] + 2 * BINS],1);   // BLUE

    }


    // wait for all threads in block to finish
    __syncthreads();

    //TODO use atomic operations to update the global histogram
    atomicAdd(&hist[tid_local_i * blockDim.x + tid_local_j], local_histR[tid_local_i * blockDim.x + tid_local_j]);
    atomicAdd(&hist[tid_local_i * blockDim.x + tid_local_j + BINS], local_histG[tid_local_i * blockDim.x + tid_local_j]);
    atomicAdd(&hist[tid_local_i * blockDim.x + tid_local_j + 2 * BINS], local_histB[tid_local_i * blockDim.x + tid_local_j]);


}

int main(int argc, char **argv)
{
    char *image_file = argv[1];

    if (argc > 1)
    {
        image_file = argv[1];
    }
    else
    {
        fprintf(stderr, "Not enough arguments\n");
        fprintf(stderr, "Usage: %s <IMAGE_PATH>\n", argv[0]);
        exit(1);
    }

    // Initalize the histogram
    h_hist = (unsigned int *)calloc(3 * BINS, sizeof(unsigned int));

    int width, height, cpp;
    unsigned char *image_in = stbi_load(image_file, &width, &height, &cpp, 0); // Load the image
    
    //########## CPU ##########
    /*
    if (image_in)
    {
        // Compute and print the histogram

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        histogramCPU(image_in, h_hist, width, height, cpp);
        cudaEventRecord(stop);

        // Wait for the event to finish
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("Time: %0.3f milliseconds \n", milliseconds);
        printHistogram(h_hist);
    }
    else
    {
        fprintf(stderr, "Error loading image %s!\n", image_file);
    }
    */
    
    
    //########## GPU ##########

    if (image_in)
    {
        // Compute and print the histogram

        // Set the thread execution grid (1 block)
        dim3 blockSize(BLOCK_SIZE);

        //TODO calculate the grid size so that there is enough blocks to cover the whole image (1 pixel per thread)
        dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);


        // allocate memory for the histogram on the GPU
        d_histGPU = (unsigned int *)calloc(3 * BINS, sizeof(unsigned int));
        // allocate memory for the image on the GPU
        unsigned char *d_imageGPU;
        d_imageGPU = (unsigned char *)calloc(width * height * cpp, sizeof(unsigned char));

        // copy the histogram to the GPU
        checkCudaErrors(cudaMalloc(&d_histGPU, sizeof(h_hist)));
        checkCudaErrors(cudaMemcpy(d_histGPU, h_hist, sizeof(h_hist), cudaMemcpyHostToDevice));

        // copy the image to the GPU
        checkCudaErrors(cudaMalloc(&d_imageGPU, sizeof(image_in)));
        checkCudaErrors(cudaMemcpy(d_imageGPU, image_in, sizeof(image_in), cudaMemcpyHostToDevice));


        //TODO call the kernel
        histogramGPU<<<gridSize, blockSize>>>(image_in, d_histGPU, width, height, cpp);
        


        //TODO copy the histogram back to the CPU

        //TODO print the histogram

        //TODO free the memory on the GPU

        //time mesurement
        cudaEventRecord(stop);

        // Wait for the event to finish
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("Time: %0.3f milliseconds \n", milliseconds);
        printHistogram(h_hist); //TODO is this the correct histogram? to display
    }
    else
    {
        fprintf(stderr, "Error loading image %s!\n", image_file);
    }



    //########## CLEAN UP ##########

    // Free the image
    stbi_image_free(image_in);

    return 0;
}