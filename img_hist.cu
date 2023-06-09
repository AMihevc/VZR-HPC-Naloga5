#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define BINS 256

unsigned int *hist;

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
}
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
    hist = (unsigned int *)calloc(3 * BINS, sizeof(unsigned int));

    int width, height, cpp;
    unsigned char *image_in = stbi_load(image_file, &width, &height, &cpp, 0);
    if (image_in)
    {
        // Compute and print the histogram

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        histogramCPU(image_in, hist, width, height, cpp);
        cudaEventRecord(stop);

        // Wait for the event to finish
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("Time: %0.3f milliseconds \n", milliseconds);
        printHistogram(hist);
    }
    else
    {
        fprintf(stderr, "Error loading image %s!\n", image_file);
    }

    return 0;
}