// System includes
#include <stdio.h>
#include <math.h>
// CUDA includes
#include <cuda.h>
#include <cuda_runtime.h>
// File includes
#include "../include/Kernel.cuh"
#include "../resource/cuda_error_check.cuh"

/**
 * This kernel file contains kernel functions to run
 * on the device as well as the wrappers necessary
 * to call them in the Bitmap class without having
 * to worry about needing to convert the class into
 * a CUDA C compatible class. All that is done within
 * this file.
 */

// Preprocessor definitions
#define THREAD_COUNT 256



/**
 * Host-only kernels
 */

/**
 * Thread/Block count functions for host
 */
__host__ inline static int DetermineThreads(int width, int height)
{
    if (height < THREAD_COUNT && width < THREAD_COUNT)
        return (int)(sqrt(sqrt(height * width)) + 1);
    else
        return (int)(sqrt(THREAD_COUNT));
}

__host__ inline static void SetupKernelThreadParams(dim3* threadsPerBlock, dim3* blocks, int threadCount, int width, int height)
{
    threadsPerBlock->x = threadCount * .25;
    threadsPerBlock->y = threadCount; // xy = (threadCount * .25) * threadCount
    blocks = blocks; // stop warnings in compiler
    int blockWidth = (width + threadsPerBlock->x - 1) / threadsPerBlock->x * .25 +
                     1; // *.25 due to 4:1 thread unroll and +1 for ceiling
    int blockHeight = (height + threadsPerBlock->y - 1) / threadsPerBlock->y;
    blocks->x = blockWidth != 0 ? blockWidth : 1;
    blocks->y = blockHeight != 0 ? blockHeight : 1;
}

__host__ static cudaTextureObject_t SetupTextureObjectArray(uchar4* rawTexture, int width, int height)
{
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaArray* texArray = NULL;
    CUDA_CALL( cudaMallocArray(&texArray, &channelDesc, width, height) );

    // CUDA Allocation of CPU texture import
    CUDA_CALL( cudaMemcpy2DToArray(texArray, 0, 0, rawTexture, width*sizeof(uchar4), width*sizeof(uchar4), height, cudaMemcpyHostToDevice) );

    // CUDA Kepler texture setup
    cudaResourceDesc texResource;
    cudaTextureDesc texDescription;
    memset(&texResource, 0, sizeof(texResource));
    memset(&texDescription, 0, sizeof(texDescription));
    texResource.resType = cudaResourceTypeArray;
    texResource.res.array.array = texArray;
    texDescription.addressMode[0] = cudaAddressModeWrap;
    texDescription.addressMode[1] = cudaAddressModeWrap;
    texDescription.addressMode[2] = cudaAddressModeWrap;
    texDescription.filterMode = cudaFilterModePoint;
    texDescription.readMode = cudaReadModeElementType;

    // CUDA Kepler texture
    cudaTextureObject_t texObject;
    CUDA_CALL( cudaCreateTextureObject(&texObject, &texResource, &texDescription, NULL) );

    return texObject;
}



/**
 * Host calls to global kernels
 */

// Allocates memory for device, then launches kernel to fill
// with random values of either 0 or 255 (black or white) to
// produce static. Whatever is created is copied back to the
// host into a locally created variable map that is returned.
// KEEP IN MIND that the return value is a pointer to a
// flattened array rather than a 2D array (CUDA's fault).
__host__ unsigned char* LaunchStaticGeneration(int width,
                                               int height)
{
    // Host Allocation (Flattened 2D array)
    unsigned char* map = (unsigned char*)malloc(height*width*sizeof(unsigned char));

    // CUDA Allocation
    unsigned char* d_map;
    size_t d_pitch;
    CUDA_CALL( cudaMallocPitch( (void **)&d_map, &d_pitch, width*sizeof(unsigned char), height ) );

    // Thread setup
    dim3 threadsPerBlock(1,1);
    dim3 blocks(1,1);
    int threadCount = DetermineThreads(width, height);
    SetupKernelThreadParams(&threadsPerBlock, &blocks, threadCount, width, height);

    // cuRand Generator and Generator Allocation
    curandState* d_state;
    CUDA_CALL( cudaMalloc(&d_state, pow(threadCount, 2)*blocks.x*blocks.y*sizeof(curandState)) );
    unsigned long long seed = (unsigned long long)time(0);
    __KernelInit<<<blocks, threadsPerBlock>>>(d_state, seed, width);
    CUDA_KERNEL_CALL();
    CUDA_CALL( cudaDeviceSynchronize() );

    // Kernel call
    __StaticGeneration<<<blocks, threadsPerBlock>>>(width, height, d_map, d_pitch, d_state);
    CUDA_KERNEL_CALL();
    CUDA_CALL( cudaDeviceSynchronize() );

    // Copy results back to host
    CUDA_CALL( cudaMemcpy2D(map, width*sizeof(unsigned char), d_map, d_pitch, width*sizeof(unsigned char), height, cudaMemcpyDeviceToHost) );

    // CUDA Deallocation
    CUDA_CALL( cudaFree(d_map) );
    CUDA_CALL( cudaFree(d_state) );

    // CUDA Exit
    return map;
}

// Texture variant of above
__host__ unsigned char* LaunchStaticGenerationTexture(int width,
                                                      int height,
                                                      unsigned char* texture)
{
    // Host Allocation (Flattened 2D array)
    unsigned char* map = (unsigned char*)malloc(height*width*sizeof(unsigned char));

    // CUDA Allocation
    unsigned char* d_map;
    size_t d_pitch;
    CUDA_CALL( cudaMallocPitch( (void **)&d_map, &d_pitch, width*sizeof(unsigned char), height ) );

    // CUDA Allocation of CPU texture import
    unsigned char* d_texture;
    size_t d_tex_pitch;
    CUDA_CALL( cudaMallocPitch( (void **)&d_texture, &d_tex_pitch, width*sizeof(unsigned char), height) );
    CUDA_CALL( cudaMemcpy2D(d_texture, d_tex_pitch, texture, width*sizeof(unsigned char), width*sizeof(unsigned char), height, cudaMemcpyHostToDevice) );

    // CUDA Kepler texture setup
    cudaResourceDesc texResource;
    cudaTextureDesc texDescription;
    memset(&texResource, 0, sizeof(texResource));
    memset(&texDescription, 0, sizeof(texDescription));
    texResource.resType = cudaResourceTypeLinear;
    texResource.res.linear.devPtr = d_texture;
    texResource.res.linear.desc.f = cudaChannelFormatKindUnsigned;
    texResource.res.linear.desc.x = 8;
    texResource.res.linear.sizeInBytes = d_tex_pitch*height*sizeof(unsigned char);
    texDescription.addressMode[0] = cudaAddressModeClamp;
    texDescription.readMode = cudaReadModeElementType;

    // CUDA Kepler texture
    cudaTextureObject_t tex;
    CUDA_CALL( cudaCreateTextureObject(&tex, &texResource, &texDescription, NULL) );

    // Thread setup
    dim3 threadsPerBlock(1,1);
    dim3 blocks(1,1);
    int threadCount = DetermineThreads(width, height);
    SetupKernelThreadParams(&threadsPerBlock, &blocks, threadCount, width, height);

    // Kernel call
    __StaticGeneration<<<blocks, threadsPerBlock>>>(width, height, d_map, d_pitch, tex);
    CUDA_KERNEL_CALL();
    CUDA_CALL( cudaDeviceSynchronize() );

    // Copy results back to host
    CUDA_CALL( cudaMemcpy2D(map, width*sizeof(unsigned char), d_map, d_pitch, width*sizeof(unsigned char), height, cudaMemcpyDeviceToHost) );

    // CUDA Deallocation
    CUDA_CALL( cudaFree(d_map) );
    CUDA_CALL( cudaFree(d_texture) );
    CUDA_CALL( cudaDestroyTextureObject(tex) );

    // CUDA Exit
    return map;
}

// Performance-tracking variants of LaunchStaticGeneration
__host__ unsigned char* LaunchStaticGeneration(int width,
                                               int height,
                                               double performance[])
{
    // Host Allocation (Flattened 2D array)
    unsigned char* map = (unsigned char*)malloc(height*width*sizeof(unsigned char));

    // CUDA Allocation
    unsigned char* d_map;
    size_t d_pitch;
    CUDA_CALL( cudaMallocPitch( (void **)&d_map, &d_pitch, width*sizeof(unsigned char), height ) );

    // Thread setup
    dim3 threadsPerBlock(1,1);
    dim3 blocks(1,1);
    int threadCount = DetermineThreads(width, height);
    SetupKernelThreadParams(&threadsPerBlock, &blocks, threadCount, width, height);

    // Kernel init timer start
    cudaEvent_t init_start, init_stop;
    float init_time;
    CUDA_CALL( cudaEventCreate(&init_start) );
    CUDA_CALL( cudaEventCreate(&init_stop) );
    CUDA_CALL( cudaEventRecord(init_start, 0) );
    // cuRand Generator and Generator Allocation
    curandState* d_state;
    CUDA_CALL( cudaMalloc(&d_state, pow(threadCount, 2)*blocks.x*blocks.y*sizeof(curandState)) );
    unsigned long long seed = (unsigned long long)time(0);
    __KernelInit<<<blocks, threadsPerBlock>>>(d_state, seed, width);
    CUDA_KERNEL_CALL();
    CUDA_CALL( cudaDeviceSynchronize() );
    // Timer end
    CUDA_CALL( cudaEventRecord(init_stop, 0) );
    CUDA_CALL( cudaEventSynchronize(init_stop) );
    CUDA_CALL( cudaEventElapsedTime(&init_time, init_start, init_stop) );
    performance[0] = init_time;

    // Timer start
    cudaEvent_t kernel_start, kernel_stop;
    float kernel_time;
    CUDA_CALL( cudaEventCreate(&kernel_start) );
    CUDA_CALL( cudaEventCreate(&kernel_stop) );
    CUDA_CALL( cudaEventRecord(kernel_start, 0) );
    // Kernel call
    __StaticGeneration<<<blocks, threadsPerBlock>>>(width, height, d_map, d_pitch, d_state);
    CUDA_KERNEL_CALL();
    CUDA_CALL( cudaDeviceSynchronize() );
    // Timer end
    CUDA_CALL( cudaEventRecord(kernel_stop, 0) );
    CUDA_CALL( cudaEventSynchronize(kernel_stop) );
    CUDA_CALL( cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop) );
    performance[1] = kernel_time;

    // Timer start
    cudaEvent_t copy_start, copy_stop;
    float copy_time;
    CUDA_CALL( cudaEventCreate(&copy_start) );
    CUDA_CALL( cudaEventCreate(&copy_stop) );
    CUDA_CALL( cudaEventRecord(copy_start, 0) );
    // Copy results back to host
    CUDA_CALL( cudaMemcpy2D(map, width*sizeof(unsigned char), d_map, d_pitch, width*sizeof(unsigned char), height, cudaMemcpyDeviceToHost) );
    // Timer end
    CUDA_CALL( cudaEventRecord(copy_stop, 0) );
    CUDA_CALL( cudaEventSynchronize(copy_stop) );
    CUDA_CALL( cudaEventElapsedTime(&copy_time, copy_start, copy_stop) );
    performance[2] = copy_time;

    // CUDA Deallocation
    CUDA_CALL( cudaFree(d_map) );
    CUDA_CALL( cudaFree(d_state) );

    // CUDA Exit
    return map;
}

__host__ unsigned char* LaunchStaticGenerationTexture(int width,
                                                      int height,
                                                      unsigned char* texture,
                                                      double performance[])
{
    // Host Allocation (Flattened 2D array)
    unsigned char* map = (unsigned char*)malloc(height*width*sizeof(unsigned char));

    // CUDA Allocation
    unsigned char* d_map;
    size_t d_pitch;
    CUDA_CALL( cudaMallocPitch( (void **)&d_map, &d_pitch, width*sizeof(unsigned char), height ) );

    // CUDA Allocation of CPU texture import
    unsigned char* d_texture;
    size_t d_tex_pitch;
    CUDA_CALL( cudaMallocPitch( (void **)&d_texture, &d_tex_pitch, width*sizeof(unsigned char), height) );
    CUDA_CALL( cudaMemcpy2D(d_texture, d_tex_pitch, texture, width*sizeof(unsigned char), width*sizeof(unsigned char), height, cudaMemcpyHostToDevice) );

    // CUDA Kepler texture setup
    cudaResourceDesc texResource;
    cudaTextureDesc texDescription;
    memset(&texResource, 0, sizeof(texResource));
    memset(&texDescription, 0, sizeof(texDescription));
    texResource.resType = cudaResourceTypeLinear;
    texResource.res.linear.devPtr = d_texture;
    texResource.res.linear.desc.f = cudaChannelFormatKindUnsigned;
    texResource.res.linear.desc.x = 8;
    texResource.res.linear.sizeInBytes = d_tex_pitch*height*sizeof(unsigned char);
    texDescription.addressMode[0] = cudaAddressModeClamp;
    texDescription.readMode = cudaReadModeElementType;

    // CUDA Kepler texture
    cudaTextureObject_t tex;
    CUDA_CALL( cudaCreateTextureObject(&tex, &texResource, &texDescription, NULL) );

    // Thread setup
    dim3 threadsPerBlock(1,1);
    dim3 blocks(1,1);
    int threadCount = DetermineThreads(width, height);
    SetupKernelThreadParams(&threadsPerBlock, &blocks, threadCount, width, height);

    // Timer start
    cudaEvent_t kernel_start, kernel_stop;
    float kernel_time;
    CUDA_CALL( cudaEventCreate(&kernel_start) );
    CUDA_CALL( cudaEventCreate(&kernel_stop) );
    CUDA_CALL( cudaEventRecord(kernel_start, 0) );
    // Kernel call
    __StaticGeneration<<<blocks, threadsPerBlock>>>(width, height, d_map, d_pitch, tex);
    CUDA_KERNEL_CALL();
    CUDA_CALL( cudaDeviceSynchronize() );
    // Timer end
    CUDA_CALL( cudaEventRecord(kernel_stop, 0) );
    CUDA_CALL( cudaEventSynchronize(kernel_stop) );
    CUDA_CALL( cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop) );
    performance[1] = kernel_time;

    // Timer start
    cudaEvent_t copy_start, copy_stop;
    float copy_time;
    CUDA_CALL( cudaEventCreate(&copy_start) );
    CUDA_CALL( cudaEventCreate(&copy_stop) );
    CUDA_CALL( cudaEventRecord(copy_start, 0) );
    // Copy results back to host
    CUDA_CALL( cudaMemcpy2D(map, width*sizeof(unsigned char), d_map, d_pitch, width*sizeof(unsigned char), height, cudaMemcpyDeviceToHost) );
    // Timer end
    CUDA_CALL( cudaEventRecord(copy_stop, 0) );
    CUDA_CALL( cudaEventSynchronize(copy_stop) );
    CUDA_CALL( cudaEventElapsedTime(&copy_time, copy_start, copy_stop) );
    performance[2] = copy_time;

    // CUDA Deallocation
    CUDA_CALL( cudaFree(d_map) );
    CUDA_CALL( cudaFree(d_texture) );
    CUDA_CALL( cudaDestroyTextureObject(tex) );

    // CUDA Exit
    return map;
}

// Allocates memory consisting of uchar4 (struct with x,y,z,w
// component), and generates unsigned chars to fill that
// allocated memory (flattened 2D array). Each x,y,z,w component
// is filled with a number between 0 and 255, excluding w
// which is always 0 (used to pad). Returns a flattened uchar4
// array.
__host__ uchar4* LaunchRandomGeneration(int width, int height)
{
    // Host Allocation for the container holding the 3 color flattened arrays
    uchar4* map = (uchar4*)malloc(width*height*sizeof(uchar4));

    // CUDA Allocation per color channel
    uchar4* d_map;
    size_t d_pitch;
    CUDA_CALL( cudaMallocPitch( (void **)&d_map, &d_pitch, width*sizeof(uchar4), height ) );

    // Thread setup
    dim3 threadsPerBlock(1,1);
    dim3 blocks(1,1);
    int threadCount = DetermineThreads(width, height);
    SetupKernelThreadParams(&threadsPerBlock, &blocks, threadCount, width, height);

    // cuRand Generator and Generator Allocation
    curandState* d_state;
    CUDA_CALL( cudaMalloc(&d_state, pow(threadCount, 2)*blocks.x*blocks.y*sizeof(curandState)) );
    unsigned long long seed = (unsigned long long)time(0);
    __KernelInit<<<blocks, threadsPerBlock>>>(d_state, seed, width);
    CUDA_KERNEL_CALL();
    CUDA_CALL( cudaDeviceSynchronize() );

    // Kernel call
    __RandomGeneration<<<blocks, threadsPerBlock>>>(width, height, d_map, d_pitch, d_state);
    CUDA_KERNEL_CALL();
    CUDA_CALL( cudaDeviceSynchronize() );

    // Copy results back to host
    CUDA_CALL( cudaMemcpy2D(map, width*sizeof(uchar4), d_map, d_pitch,
                            width*sizeof(uchar4), height, cudaMemcpyDeviceToHost) );

    // CUDA Deallocation
    CUDA_CALL( cudaFree(d_map) );
    CUDA_CALL( cudaFree(d_state) );

    // CUDA Exit
    return map;
}

// Texture variant of above
__host__ uchar4* LaunchRandomGenerationTexture(int width, int height,
                                               uchar4* texture)
{
    // Host Allocation for the container holding the 3 color flattened arrays
    uchar4* map = (uchar4*)malloc(width*height*sizeof(uchar4));

    // CUDA Allocation per color channel
    uchar4* d_map;
    size_t d_pitch;
    CUDA_CALL( cudaMallocPitch( (void **)&d_map, &d_pitch, width*sizeof(uchar4), height ) );
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaArray* texArray = NULL;
    CUDA_CALL( cudaMallocArray(&texArray, &channelDesc, width, height) );

    // CUDA Allocation of CPU texture import
    CUDA_CALL( cudaMemcpy2DToArray(texArray, 0, 0, texture, width*sizeof(uchar4), width*sizeof(uchar4), height, cudaMemcpyHostToDevice) );

    // CUDA Kepler texture setup
    cudaResourceDesc texResource;
    cudaTextureDesc texDescription;
    memset(&texResource, 0, sizeof(texResource));
    memset(&texDescription, 0, sizeof(texDescription));
    texResource.resType = cudaResourceTypeArray;
    texResource.res.array.array = texArray;
    texDescription.addressMode[0] = cudaAddressModeWrap;
    texDescription.addressMode[1] = cudaAddressModeWrap;
    texDescription.addressMode[2] = cudaAddressModeWrap;
    texDescription.filterMode = cudaFilterModePoint;
    texDescription.readMode = cudaReadModeElementType;

    // CUDA Kepler texture
    cudaTextureObject_t tex;
    CUDA_CALL( cudaCreateTextureObject(&tex, &texResource, &texDescription, NULL) );

    // Thread setup
    dim3 threadsPerBlock(1,1);
    dim3 blocks(1,1);
    int threadCount = DetermineThreads(width, height);
    SetupKernelThreadParams(&threadsPerBlock, &blocks, threadCount, width, height);

    // Kernel call
    __RandomGeneration<<<blocks, threadsPerBlock>>>(width, height, d_map, d_pitch, tex);
    CUDA_KERNEL_CALL();
    CUDA_CALL( cudaDeviceSynchronize() );

    // Copy results back to host
    CUDA_CALL( cudaMemcpy2D(map, width*sizeof(uchar4), d_map, d_pitch,
                            width*sizeof(uchar4), height, cudaMemcpyDeviceToHost) );

    // CUDA Deallocation
    CUDA_CALL( cudaFree(d_map) );
    CUDA_CALL( cudaDestroyTextureObject(tex) );

    // CUDA Exit
    return map;
}

// Performance-tracking variant of LaunchRandomGeneration
__host__ uchar4* LaunchRandomGeneration(int width, int height,
                                        double performance[])
{
    // Host Allocation for the container holding the 3 color flattened arrays
    uchar4* map = (uchar4*)malloc(width*height*sizeof(uchar4));

    // CUDA Allocation per color channel
    uchar4* d_map;
    size_t d_pitch;
    CUDA_CALL( cudaMallocPitch( (void **)&d_map, &d_pitch, width*sizeof(uchar4), height ) );

    // Thread setup
    dim3 threadsPerBlock(1,1);
    dim3 blocks(1,1);
    int threadCount = DetermineThreads(width, height);
    SetupKernelThreadParams(&threadsPerBlock, &blocks, threadCount, width, height);

    // Kernel init timer start
    cudaEvent_t init_start, init_stop;
    float init_time;
    CUDA_CALL( cudaEventCreate(&init_start) );
    CUDA_CALL( cudaEventCreate(&init_stop) );
    CUDA_CALL( cudaEventRecord(init_start, 0) );
    // cuRand Generator and Generator Allocation
    curandState* d_state;
    CUDA_CALL( cudaMalloc(&d_state, pow(threadCount, 2)*blocks.x*blocks.y*sizeof(curandState)) );
    unsigned long long seed = (unsigned long long)time(0);
    __KernelInit<<<blocks, threadsPerBlock>>>(d_state, seed, width);
    CUDA_KERNEL_CALL();
    CUDA_CALL( cudaDeviceSynchronize() );
    // Timer end
    CUDA_CALL( cudaEventRecord(init_stop, 0) );
    CUDA_CALL( cudaEventSynchronize(init_stop) );
    CUDA_CALL( cudaEventElapsedTime(&init_time, init_start, init_stop) );
    performance[0] = init_time;

    // Timer start
    cudaEvent_t kernel_start, kernel_stop;
    float kernel_time;
    CUDA_CALL( cudaEventCreate(&kernel_start) );
    CUDA_CALL( cudaEventCreate(&kernel_stop) );
    CUDA_CALL( cudaEventRecord(kernel_start, 0) );
    // Kernel call
    __RandomGeneration<<<blocks, threadsPerBlock>>>(width, height, d_map, d_pitch, d_state);
    CUDA_KERNEL_CALL();
    CUDA_CALL( cudaDeviceSynchronize() );
    // Timer end
    CUDA_CALL( cudaEventRecord(kernel_stop, 0) );
    CUDA_CALL( cudaEventSynchronize(kernel_stop) );
    CUDA_CALL( cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop) );
    performance[1] = kernel_time;

    // Timer start
    cudaEvent_t copy_start, copy_stop;
    float copy_time;
    CUDA_CALL( cudaEventCreate(&copy_start) );
    CUDA_CALL( cudaEventCreate(&copy_stop) );
    CUDA_CALL( cudaEventRecord(copy_start, 0) );
    // Copy results back to host
    CUDA_CALL( cudaMemcpy2D(map, width*sizeof(uchar4), d_map, d_pitch,
                            width*sizeof(uchar4), height, cudaMemcpyDeviceToHost) );
    // Timer end
    CUDA_CALL( cudaEventRecord(copy_stop, 0) );
    CUDA_CALL( cudaEventSynchronize(copy_stop) );
    CUDA_CALL( cudaEventElapsedTime(&copy_time, copy_start, copy_stop) );
    performance[2] = copy_time;

    // CUDA Deallocation
    CUDA_CALL( cudaFree(d_map) );
    CUDA_CALL( cudaFree(d_state) );

    // CUDA Exit
    return map;
}

__host__ uchar4* LaunchRandomGenerationTexture(int width, int height,
                                               uchar4* texture,
                                               double performance[])
{
    // Host Allocation for the container holding the 3 color flattened arrays
    uchar4* map = (uchar4*)malloc(width*height*sizeof(uchar4));

    // CUDA Allocation per color channel
    uchar4* d_map;
    size_t d_pitch;
    CUDA_CALL( cudaMallocPitch( (void **)&d_map, &d_pitch, width*sizeof(uchar4), height ) );
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaArray* texArray = NULL;
    CUDA_CALL( cudaMallocArray(&texArray, &channelDesc, width, height) );

    // CUDA Allocation of CPU texture import
    CUDA_CALL( cudaMemcpy2DToArray(texArray, 0, 0, texture, width*sizeof(uchar4), width*sizeof(uchar4), height, cudaMemcpyHostToDevice) );

    // CUDA Kepler texture setup
    cudaResourceDesc texResource;
    cudaTextureDesc texDescription;
    memset(&texResource, 0, sizeof(texResource));
    memset(&texDescription, 0, sizeof(texDescription));
    texResource.resType = cudaResourceTypeArray;
    texResource.res.array.array = texArray;
    texDescription.addressMode[0] = cudaAddressModeWrap;
    texDescription.addressMode[1] = cudaAddressModeWrap;
    texDescription.addressMode[2] = cudaAddressModeWrap;
    texDescription.filterMode = cudaFilterModePoint;
    texDescription.readMode = cudaReadModeElementType;

    // CUDA Kepler texture
    cudaTextureObject_t tex;
    CUDA_CALL( cudaCreateTextureObject(&tex, &texResource, &texDescription, NULL) );

    // Thread setup
    dim3 threadsPerBlock(1,1);
    dim3 blocks(1,1);
    int threadCount = DetermineThreads(width, height);
    SetupKernelThreadParams(&threadsPerBlock, &blocks, threadCount, width, height);

    // Timer start
    cudaEvent_t kernel_start, kernel_stop;
    float kernel_time;
    CUDA_CALL( cudaEventCreate(&kernel_start) );
    CUDA_CALL( cudaEventCreate(&kernel_stop) );
    CUDA_CALL( cudaEventRecord(kernel_start, 0) );
    // Kernel call
    __RandomGeneration<<<blocks, threadsPerBlock>>>(width, height, d_map, d_pitch, tex);
    CUDA_KERNEL_CALL();
    CUDA_CALL( cudaDeviceSynchronize() );
    // Timer end
    CUDA_CALL( cudaEventRecord(kernel_stop, 0) );
    CUDA_CALL( cudaEventSynchronize(kernel_stop) );
    CUDA_CALL( cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop) );
    performance[1] = kernel_time;

    // Timer start
    cudaEvent_t copy_start, copy_stop;
    float copy_time;
    CUDA_CALL( cudaEventCreate(&copy_start) );
    CUDA_CALL( cudaEventCreate(&copy_stop) );
    CUDA_CALL( cudaEventRecord(copy_start, 0) );
    // Copy results back to host
    CUDA_CALL( cudaMemcpy2D(map, width*sizeof(uchar4), d_map, d_pitch,
                            width*sizeof(uchar4), height, cudaMemcpyDeviceToHost) );

    // Timer end
    CUDA_CALL( cudaEventRecord(copy_stop, 0) );
    CUDA_CALL( cudaEventSynchronize(copy_stop) );
    CUDA_CALL( cudaEventElapsedTime(&copy_time, copy_start, copy_stop) );
    performance[2] = copy_time;

    // CUDA Deallocation
    CUDA_CALL( cudaFree(d_map) );
    CUDA_CALL( cudaDestroyTextureObject(tex) );

    // CUDA Exit
    return map;
}

// Performance-tracking variant of LaunchMatrixMult
__host__ uchar4* LaunchMatrixMult(uchar4* colorMap, int width, int height,
                                  uchar4* other, int otherWidth, int otherHeight,
                                  double performance[])
{
    // Host Allocation for the container holding the 3 color flattened arrays
    // The resultant matrix using the height and otherWidth because
    // height = x of resultant && otherWidth = y of resultant
    uchar4* map = (uchar4*)malloc(height*otherWidth*sizeof(uchar4));

    // CUDA Allocation per color channel
    uchar4* d_map;
    size_t d_pitch;
    CUDA_CALL( cudaMallocPitch( (void **)&d_map, &d_pitch, height*sizeof(uchar4), otherWidth ) );

    // CUDA Kepler texture
    cudaTextureObject_t texMatrixLeft = SetupTextureObjectArray(colorMap, width, height);
    cudaTextureObject_t texMatrixRight = SetupTextureObjectArray(other, otherWidth, otherHeight);

    // Thread setup
    dim3 threadsPerBlock(1,1);
    dim3 blocks(1,1);
    int threadCount = DetermineThreads(height, otherWidth);
    SetupKernelThreadParams(&threadsPerBlock, &blocks, threadCount, height, otherWidth);

    // Timer start
    cudaEvent_t kernel_start, kernel_stop;
    float kernel_time;
    CUDA_CALL( cudaEventCreate(&kernel_start) );
    CUDA_CALL( cudaEventCreate(&kernel_stop) );
    CUDA_CALL( cudaEventRecord(kernel_start, 0) );
    // Kernel call
    __MatrixMult<<<blocks, threadsPerBlock>>>(height, otherWidth, d_map, d_pitch,
                                              width, texMatrixLeft, texMatrixRight);
    CUDA_KERNEL_CALL();
    CUDA_CALL( cudaDeviceSynchronize() );
    // Timer end
    CUDA_CALL( cudaEventRecord(kernel_stop, 0) );
    CUDA_CALL( cudaEventSynchronize(kernel_stop) );
    CUDA_CALL( cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop) );
    performance[1] = kernel_time;

    // Timer start
    cudaEvent_t copy_start, copy_stop;
    float copy_time;
    CUDA_CALL( cudaEventCreate(&copy_start) );
    CUDA_CALL( cudaEventCreate(&copy_stop) );
    CUDA_CALL( cudaEventRecord(copy_start, 0) );
    // Copy results back to host
    CUDA_CALL( cudaMemcpy2D(map, height*sizeof(uchar4), d_map, d_pitch,
                            height*sizeof(uchar4), otherWidth, cudaMemcpyDeviceToHost) );

    // Timer end
    CUDA_CALL( cudaEventRecord(copy_stop, 0) );
    CUDA_CALL( cudaEventSynchronize(copy_stop) );
    CUDA_CALL( cudaEventElapsedTime(&copy_time, copy_start, copy_stop) );
    performance[2] = copy_time;

    // CUDA Deallocation
    CUDA_CALL( cudaFree(d_map) );
    CUDA_CALL( cudaDestroyTextureObject(texMatrixLeft) );
    CUDA_CALL( cudaDestroyTextureObject(texMatrixRight) );

    // CUDA Exit
    return map;
}



/**
 * Global kernels
 */

/**
 * Bitmap generation
 */
// Black and white only (off or on basically)
__global__ static void __StaticGeneration(int width, int height,
                                          unsigned char* map, int pitch,
                                          curandState* state)
{
    // Find x and y coordinates
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = 4 * idx;

    // Get x and y index, then set that element to either 0 or 255 depending
    // on the random number (0 --> <127 // 255 --> >127).
    // Utilizes unroll 4:1 per thread for more efficient access
    if (row < height && col < width)
    {
        // Generate and scale the random float produced (0-255)
        unsigned char random;
        // Retrieve matrix position
        int max = (width - col < 4) ? width - col : 4;
        unsigned char* current = (unsigned char*)((char*)map + row * pitch);
        #pragma unroll
        for (int i = 0; i < max; ++i)
        {
            random = device_GenerateChar(state, idx, row, width);
            current[col + i] = (random < 127) ? (unsigned char)0 : (unsigned char)255;
        }
    }

    return;
}

__global__ static void __StaticGeneration(int width, int height,
                                          unsigned char* map, int pitch,
                                          cudaTextureObject_t tex)
{
    // Find x and y coordinates
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = 4 * idx;

    // Get x and y index, then set that element to either 0 or 255 depending
    // on the random number (0 --> <127 // 255 --> >127).
    // Utilizes unroll 4:1 per thread for more efficient access
    if (row < height && col < width)
    {
        // Retrieve matrix position
        unsigned char* current = (unsigned char*)((char*)map + row * pitch);
        // Temporary value holder
        unsigned char texValue;
        int max = (width - col < 4) ? width - col : 4;
        #pragma unroll
        for (int i = 0; i < max; ++i)
        {
            texValue = tex1Dfetch<unsigned char>(tex, col + row * pitch + i);
            current[col + i] = (texValue < 127) ? (unsigned char)0 : (unsigned char)255;
        }
    }

    return;
}

// Random Generation Kernels
// Takes the dimensions (WxH), each of the color channels, the
// the allocated pitch for offset, and the curand State
__global__ static void __RandomGeneration(int width, int height,
                                          uchar4* map, int pitch,
                                          curandState* state)
{
    // Find x and y coordinates
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = 4 * idx;

    // Retrieve matrix position
    uchar4* current = (uchar4*)((char*)map + idy * pitch);

    // Set elements [current, current + 4] to a random char
    if (idy < height && col < width)
    {
        int max = (width - col < 4) ? width - col : 4;
        #pragma unroll
        for (int i = 0; i < max; ++i)
        {
            current[col + i].x = device_GenerateChar(state, idx, idy, width);
            current[col + i].y = device_GenerateChar(state, idx, idy, width);
            current[col + i].z = device_GenerateChar(state, idx, idy, width);
            current[col + i].w = (unsigned char)0;
        }
    }

    return;
}

__global__ static void __RandomGeneration(int width, int height,
                                          uchar4* map, int pitch,
                                          cudaTextureObject_t tex)
{
    // Find x and y coordinates
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = 4 * idx;

    // Retrieve matrix position
    uchar4* current = (uchar4*)((char*)map + idy * pitch);

    // Set elements [current, current + 4] to a random char
    if (idy < height && col < width)
    {
        int max = (width - col < 4) ? width - col : 4;
        #pragma unroll
        for (int i = 0; i < max; ++i)
        {
            uchar4 texel = tex2D<uchar4>(tex, col + i, idy);
            uchar4* curElement = current + col + i;
            curElement->x = texel.x;
            curElement->y = texel.y;
            curElement->z = texel.z;
            curElement->w = (unsigned char)0;
        }
    }

    return;
}

__global__ static void __MatrixMult(int width, int height, uchar4* map, int pitch,
                                    int colRowLength, cudaTextureObject_t texLeft,
                                    cudaTextureObject_t texRight)
{
    // Find x and y coordinates
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = 4 * idx;

    // Retrieve matrix position
    uchar4* current = (uchar4*)((char*)map + idy * pitch);

    // col + i and idy are the resultant matrix access
    // Still using i, j, k for consistency of the CPU variant
    double heightDownscale = 1.0 / (double)height;
    double charDownscale = 1.0 / 255.0;

    if (idy < height && col < width)
    {
        int max = (width - col < 4) ? width - col : 4;

        #pragma unroll
        for (int i = 0; i < max; ++i)
        {
            double sumR = 0.0;
            double sumG = 0.0;
            double sumB = 0.0;

            // Iterate through row of left and col of right
            for (int k = 0; k < colRowLength; ++k)
            {
                uchar4 texelLeft  = tex2D<uchar4>(texLeft, k, idy);
                uchar4 texelRight = tex2D<uchar4>(texRight, col + i, k);

                sumR += (double) texelLeft.x * (double) texelRight.x * heightDownscale * charDownscale;
                sumG += (double) texelLeft.y * (double) texelRight.y * heightDownscale * charDownscale;
                sumB += (double) texelLeft.z * (double) texelRight.z * heightDownscale * charDownscale;
            }

            uchar4* curElement = current + col + i;
            curElement->x = (unsigned char) sumR;
            curElement->y = (unsigned char) sumG;
            curElement->z = (unsigned char) sumB;
            curElement->w = (unsigned char) 0;
        }

        return;
    }
}

/**
 * Kernel initialization
 */
 // Setup generator
 __global__ static void __KernelInit(curandState* state, unsigned long long seed, int width)
{
     const int idx = (blockIdx.x * blockDim.x + threadIdx.x);
     const int idy = (blockIdx.y * blockDim.y + threadIdx.y);

     curand_init(seed*clock64() + idx + idy * width, 0, 0, &state[idx + idy * width]);
}



/**
 * Device kernels, called by private global kernels
 */
__device__ static unsigned char device_GenerateChar(curandState* state, int idx,
                                                    int idy, int width)
{
    // Generate and scale the random float produced (0-255)
    curandState localState = state[idx + idy * width];
    float random = curand_uniform(&localState);
    state[idx + idy * width] = localState;
    random *= (255 + 0.999999);

    return (unsigned char)random;
}