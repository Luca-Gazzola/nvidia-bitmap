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

// Public callable host functions
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
__host__ uchar4* LaunchRandomGeneration(int width, int height);
__host__ uchar4* LaunchRandomGeneration(int width, int height,
                                        double performance[]);
__host__ uchar4* LaunchRandomGenerationTexture(int width, int height,
                                               uchar4* texture);
__host__ uchar4* LaunchRandomGenerationTexture(int width, int height,
                                               uchar4* texture,
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