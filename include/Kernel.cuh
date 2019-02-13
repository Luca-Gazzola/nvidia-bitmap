#ifndef KERNEL_CUH
#define KERNEL_CUH
// CUDA includes
#include <curand.h>
#include <curand_kernel.h>
#include "../resource/jetbrains_cuda_c.h"

/**
 * Host kernels
 */
 // Private host functions
__host__ inline static int DetermineThreads(int width, int height);
__host__ inline static void SetupKernelThreadParams(dim3* threadsPerBlock, dim3* blocks, int threadCount, int width, int height);
__host__ static cudaTextureObject_t SetupTextureObjectArray(uchar4* rawTexture, int width, int height);

// Public callable host functions
// Static Generation
__host__ unsigned char* LaunchStaticGeneration(int width,
                                               int height);
__host__ unsigned char* LaunchStaticGeneration(int width,
                                               int height,
                                               double performance[]);
__host__ unsigned char* LaunchStaticGenerationTexture(int width,
                                                      int height,
                                                      unsigned char* texture);
__host__ unsigned char* LaunchStaticGenerationTexture(int width,
                                                      int height,
                                                      unsigned char* texture,
                                                      double performance[]);

// Random Generation
__host__ uchar4* LaunchRandomGeneration(int width, int height);
__host__ uchar4* LaunchRandomGeneration(int width, int height,
                                        double performance[]);
__host__ uchar4* LaunchRandomGenerationTexture(int width, int height,
                                               uchar4* texture);
__host__ uchar4* LaunchRandomGenerationTexture(int width, int height,
                                               uchar4* texture,
                                               double performance[]);

// Addition
__host__ uchar4* LaunchAddition(uchar4* colorMap, int width, int height,
                                uchar4* otherMap, int otherWidth, int otherHeight);
__host__ uchar4* LaunchAddition(uchar4* colorMap, int width, int height,
                                uchar4* otherMap, int otherWidth, int otherHeight,
                                double performance[]);

// Subtract
__host__ uchar4* LaunchSubtract(uchar4* colorMap, int width, int height,
                                uchar4* otherMap, int otherWidth, int otherHeight);
__host__ uchar4* LaunchSubtract(uchar4* colorMap, int width, int height,
                                uchar4* otherMap, int otherWidth, int otherHeight,
                                double performance[]);

// Matrix Multiply
__host__ uchar4* LaunchMatrixMult(uchar4* colorMap, int width, int height,
                                  uchar4* otherMap, int otherWidth, int otherHeight);
__host__ uchar4* LaunchMatrixMult(uchar4* colorMap, int width, int height,
                                  uchar4* otherMap, int otherWidth, int otherHeight,
                                  double performance[]);


/**
 * Global private kernels, launched by host kernels
 */
 // Static Generation Kernels
__global__ static void __StaticGeneration(int width, int height,
                                          unsigned char* map, int pitch,
                                          curandState* state);
__global__ static void __StaticGeneration(int width, int height,
                                          unsigned char* map, int pitch,
                                          cudaTextureObject_t text);

// Random Generation Kernels
__global__ static void __RandomGeneration(int width, int height,
                                          uchar4* map, int pitch,
                                          curandState* state);
__global__ static void __RandomGeneration(int width, int height,
                                          uchar4* map, int pitch,
                                          cudaTextureObject_t tex);

// Addition Kernel
__global__ static void __Addition(uchar4* map, int pitch,
                                  uchar4* colorMap, int colorPitch, int width, int height,
                                  uchar4* otherMap, int otherPitch, int otherWidth, int otherHeight);

// Subtract Kernel
__global__ static void __Subtract(uchar4* map, int pitch,
                                  uchar4* colorMap, int colorPitch, int width, int height,
                                  uchar4* otherMap, int otherPitch, int otherWidth, int otherHeight);

// Matrix Multiplication Kernel
__global__ static void __MatrixMult(int width, int height, uchar4* map, int pitch,
                                    int colRowLength, cudaTextureObject_t texLeft,
                                    cudaTextureObject_t texRight);

/**
 * Kernel initialization
 */
__global__ static void __KernelInit(curandState* state, unsigned long long seed,
                                    int width);


/**
 * Device kernels, called by private global kernels
 */
__device__ static unsigned char device_GenerateChar(curandState* state, int idx,
                                                    int idy, int width);

#endif //KERNEL_CUH