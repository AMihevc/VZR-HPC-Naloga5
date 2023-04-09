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
#define BLOCK_SIZE 32 //TODO this probably needs to be changed to 256 or 512



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


// a GPU kernel to compute the histogram on the GPU
__global__ void histogramGPU(unsigned char *imageIn,
                             unsigned int *hist,
                             int width, int height, int cpp)
{
    // Each color channel is 1 byte long, there are 4 channels RED, BLUE, GREEN,  and ALPHA
    // The order is RED|GREEN|BLUE|ALPHA for each pixel, we ignore the ALPHA channel when computing the histograms

    // calculate global and local id and use them to calculate the pixel index
    int tid_global_j = blockDim.x * blockIdx.x + threadIdx.x;
    int tid_global_i = blockDim.y * blockIdx.y + threadIdx.y;

    // check that threads dont check pixels "outside" the image
    if ( tid_global_i < height && tid_global_j < width) {

        atomicInc(&hist[imageIn[(tid_global_i * width + tid_global_j) * cpp]], 1);  // RED
        atomicInc(&hist[imageIn[(tid_global_i * width + tid_global_j) * cpp +1]+ BINS], 1);  // GREEN
        atomicInc(&hist[imageIn[(tid_global_i * width + tid_global_j) * cpp+ 2]+ 2*BINS], 1);  // BLUE
    }

}

int main(int argc, char **argv)
{
    char *image_file = argv[1];

    // if image not provided exit
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

    // Initalize variables
    unsigned int *h_hist;
    unsigned int *h_hist_seq;
    unsigned int *d_histGPU;
    unsigned char *d_imageGPU;
    int width, height, cpp;

    // load the image
    unsigned char *image_in = stbi_load(image_file, &width, &height, &cpp, 0); 
    
    // allocate memory for the histogram on the CPU
    h_hist_seq = (unsigned int *)calloc(3 * BINS, sizeof(unsigned int));
    
    //########## CPU ##########
    
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

        printf("CPU time: %0.3f milliseconds \n", milliseconds);
        printHistogram(h_hist);
    }
    else
    {
        fprintf(stderr, "Error loading image %s!\n", image_file);
    }
    
    
    
    //########## GPU ##########

    // Compute and print the histogram
    if (image_in) // if image loaded
    {
        // initialize the timig variables
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // allocate memory for the histogram, calcualted on the GPU, on the host 
        h_hist = (unsigned int *)calloc(3 * BINS, sizeof(unsigned int));

        // allocate and copy the histogram to the GPU
        checkCudaErrors(cudaMalloc(&d_histGPU, sizeof(h_hist)));
        checkCudaErrors(cudaMemcpy(d_histGPU, h_hist, sizeof(h_hist), cudaMemcpyHostToDevice));

        // allocate and copy the image to the GPU
        checkCudaErrors(cudaMalloc(&d_imageGPU, sizeof(image_in)));
        checkCudaErrors(cudaMemcpy(d_imageGPU, image_in, sizeof(image_in), cudaMemcpyHostToDevice));

        // Set the thread execution grid (1 block)
        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

        // calculate the grid size so that there is enough blocks to cover the whole image (1 pixel per thread)
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

        //time mesurement start
        cudaEventRecord(start);

        // call the kernel
        histogramGPU<<<gridSize, blockSize>>>(image_in, d_histGPU, width, height, cpp);
        
        //time mesurement stop
        cudaEventRecord(stop);

        // Wait for the event to finish
        cudaEventSynchronize(stop);

        //Error checking
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
        {
            printf("cudaGetDeviceCount error %d\n-> %s\n", err, cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // copy the histogram back to the CPU
        checkCudaErrors(cudaMemcpy(h_hist, d_histGPU, sizeof(d_histGPU), cudaMemcpyDeviceToHost));

        //free the GPU memory
        cudaFree(d_histGPU);
        cudaFree(d_imageGPU);

        // Display time mesurments
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("GPU time: %0.3f milliseconds \n", milliseconds);

        // Display the histogram 
        printHistogram(h_hist); 

        // Check if the histograms are the same   
        if (h_hist_seq == h_hist)
        {
            printf("The histograms are the same\n");
        }
        else
        {
            printf("The histograms are different\n");
        }
    }
    else
    {
        fprintf(stderr, "Error loading image %s!\n", image_file);
    }

    //########## CLEAN UP ##########

    // Free the image
    stbi_image_free(image_in);
    
    // Free the histograms
    free(h_hist);
    free(h_hist_seq);

    return 0;
}