#include <stdio.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <time.h>

const int numSamples = 1 << 24; // Total number of random points
const int numThreads = 256; // Number of threads per block
const int numBlocks = (numSamples + numThreads - 1) / numThreads; // Calculate number of blocks
__constant__ float d_radius = 1.0f; // Radius of the circle

__global__ void monteCarloPiKernel(int *globalCount, unsigned long long seed) {
    __shared__ int localCount; // Shared memory for counting points inside the circle

    if (threadIdx.x == 0) localCount = 0; // Initialize shared memory for local count

    curandState state;
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index
    curand_init(seed, idx, 0, &state); // Initialize the random state

    // Each thread generates a number of points
    int samplesPerThread = numSamples / numBlocks / numThreads; // Samples per thread
    for (int i = 0; i < samplesPerThread; i++) {
        float x = curand_uniform(&state) * d_radius; // Scale by radius
        float y = curand_uniform(&state) * d_radius; // Scale by radius
        if (x * x + y * y <= d_radius * d_radius) {
            atomicAdd(&localCount, 1); // Increment local count in shared memory
        }
    }

    // Synchronize threads before writing to global count
    __syncthreads();

    // Only the first thread in the block writes the local count to global count
    if (threadIdx.x == 0) {
        atomicAdd(globalCount, localCount); // Add local count to global count
    }
}

int main() {
    int *d_globalCount;
    int h_globalCount = 0;

    // Allocate device memory for the global counter
    cudaMalloc((void**)&d_globalCount, sizeof(int));

    // Initialize the global count to zero on the device
    cudaMemcpy(d_globalCount, &h_globalCount, sizeof(int), cudaMemcpyHostToDevice);

    // Start timing
    clock_t start = clock();

    // Launch the kernel
    monteCarloPiKernel<<<numBlocks, numThreads>>>(d_globalCount, time(0));

    // Check for errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for the kernel to finish
    cudaDeviceSynchronize();

    // Copy the result back to host
    cudaMemcpy(&h_globalCount, d_globalCount, sizeof(int), cudaMemcpyDeviceToHost);

    // Check for errors in memory copy
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error copying memory: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Stop timing
    clock_t end = clock();
    double elapsedTime = double(end - start) / CLOCKS_PER_SEC; // Calculate elapsed time

    // Clean up
    cudaFree(d_globalCount);

    // Estimate π
    double piEstimate = 4.0 * h_globalCount / (double)numSamples;

    // Print results
    printf("Total points inside circle: %d\n", h_globalCount);
    printf("Estimated value of π: %f\n", piEstimate);
    printf("Number of points generated: %d\n", numSamples);
    
    // Print radius value correctly
    float radius;
    cudaMemcpyFromSymbol(&radius, d_radius, sizeof(float)); // Copy from constant memory
    printf("Radius of circle: %f\n", radius); // Print radius

    printf("Execution time: %f seconds\n", elapsedTime);

    return 0;
}
