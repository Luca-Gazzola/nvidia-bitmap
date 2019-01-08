#ifndef CUDA_ERROR_CHECK_CUH
#define CUDA_ERROR_CHECK_CUH
#ifdef __cplusplus
extern "C" {
#endif //__cplusplus
// System includes
#include <stdio.h>
// CUDA includes
#include <cuda.h>



/**
 * Preprocessor declarations
 */
#define CUDA_CALL(call) { CudaAssert((call), __FILE__, __LINE__); }
#define CUDA_DRIVER_CALL(call) { CudaCUResultCall((call), __FILE__, __LINE__); }
#define CURAND_CALL(call) { CurandAssert((call), __FILE__, __LINE__); }
#define CUDA_KERNEL_CALL() { CudaKernelCheck(__FILE__, __LINE__); }


/**
 * Preprocessor definitions
 */
static inline void CudaAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
#ifdef CUDA_ERROR_CHECK_CUH
    if (code != cudaSuccess)
    {
        fprintf(stderr, "[CUDA ERROR] %s %s <Line %d>\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
#endif // CUDA_ERROR_CHECK_CUH
}

static inline void CudaCUResultCall(CUresult code, const char *file, int line, bool abort = true)
{
#ifdef CUDA_ERROR_CHECK_CUH
    if (code != CUDA_SUCCESS)
    {
        const char** errorString = NULL;
        cuGetErrorString(code, errorString);
        fprintf(stderr, "[CURESULT ERROR] %s %s <Line %d>\n", errorString == NULL ? "UNKNOWN" : *errorString, file, line);
        if (abort) exit(code);
    }
#endif // CUDA_ERROR_CHECK_CUH
}

static inline void CurandAssert(curandStatus_t code, const char *file, int line, bool abort = true)
{
#ifdef CUDA_ERROR_CHECK_CUH
    if (code != CURAND_STATUS_SUCCESS)
    {
        fprintf(stderr, "[CURAND ERROR] %s <Line %d>\n", file, line);
        if (abort) exit(code);
    }
#endif // CUDA_ERROR_CHECK_CUH
}

static inline void CudaKernelCheck(const char *file, int line, bool abort = true)
{
#ifdef CUDA_ERROR_CHECK_CUH
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "[CUDA KERNEL ERROR] %s %s <Line %d>\n", cudaGetErrorString(err), file, line);
        if (abort) exit(err);
    }
#endif // CUDA_ERROR_CHECK_CUH
}

#ifdef __cplusplus
}
#endif //__cplusplus
#endif //CUDA_ERROR_CHECK_CUH