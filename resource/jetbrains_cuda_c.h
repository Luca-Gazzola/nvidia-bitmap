/**
 * Since CLion is picky with syntax with CUDA C, this
 * header file predefines the CUDA C Extensions that
 * the lanuguage uses for its kernel instantiation.
 * This does not have to be included in any makefiles
 * or compilers as this file is only meant to be
 * for CLion's preprocessor.
 */

#ifndef TEST_PRINTF_JETBRAINS_CUDA_C_H
#define TEST_PRINTF_JETBRAINS_CUDA_C_H

#ifdef __JETBRAINS_IDE__
    #define __host__
    #define __device__
    #define __shared__
    #define __constant__
    #define __global__
#endif

#endif //TEST_PRINTF_JETBRAINS_CUDA_C_H
