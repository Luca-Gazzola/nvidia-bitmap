#ifndef WRAPPER_H
#define WRAPPER_H
// File includes
#include "../include/Kernel.cuh"
#include "../resource/bitmap_resources.h"

namespace Kernel
{
    void CudaSetup();

    // Generation
    RGBQUAD** StaticGeneration(Processor_Type hardware, int width, int height);
    RGBQUAD** StaticGeneration(Processor_Type hardware, int width, int height, double performance[]);
    RGBQUAD** RandomGeneration(Processor_Type hardware, int width, int height);
    RGBQUAD** RandomGeneration(Processor_Type hardware, int width, int height, double performance[]);
    // Modification
    // Addition Variants
    RGBQUAD** Addition(Processor_Type hardware, const RGBQUAD* const* colorMap, int width, int height,
                       const RGBQUAD* const* other, int otherWidth, int otherHeight);
    RGBQUAD** Addition(Processor_Type hardware, const RGBQUAD* const* colorMap, int width, int height,
                       const RGBQUAD* const* other, int otherWidth, int otherHeight, double performance[]);
    RGBQUAD** Addition(Processor_Type hardware, const RGBQUAD* const* colorMap, int width, int height,
                       int constant);
    RGBQUAD** Addition(Processor_Type hardware, const RGBQUAD* const* colorMap, int width, int height,
                       int constant, double performance[]);
    // Subtraction Variants
    RGBQUAD** Subtract(Processor_Type hardware, const RGBQUAD* const* colorMap, int width, int height,
                       const RGBQUAD* const* other, int otherWidth, int otherHeight);
    RGBQUAD** Subtract(Processor_Type hardware, const RGBQUAD* const* colorMap, int width, int height,
                       const RGBQUAD* const* other, int otherWidth, int otherHeight, double performance[]);
    RGBQUAD** Subtract(Processor_Type hardware, const RGBQUAD* const* colorMap, int width, int height,
                       int constant);
    RGBQUAD** Subtract(Processor_Type hardware, const RGBQUAD* const* colorMap, int width, int height,
                       int constant, double performance[]);
    // Point-wise Multiplication Variants
    RGBQUAD** Multiply(Processor_Type hardware, const RGBQUAD* const* colorMap, int width, int height,
                       const RGBQUAD* const* other, int otherWidth, int otherHeight);
    RGBQUAD** Multiply(Processor_Type hardware, const RGBQUAD* const* colorMap, int width, int height,
                       const RGBQUAD* const* other, int otherWidth, int otherHeight, double performance[]);
    RGBQUAD** Multiply(Processor_Type hardware, const RGBQUAD* const* colorMap, int width, int height,
                       int constant);
    RGBQUAD** Multiply(Processor_Type hardware, const RGBQUAD* const* colorMap, int width, int height,
                       int constant, double performance[]);
    // Point-wise Division Variants
    RGBQUAD** Division(Processor_Type hardware, const RGBQUAD* const* colorMap, int width, int height,
                       const RGBQUAD* const* other, int otherWidth, int otherHeight);
    RGBQUAD** Division(Processor_Type hardware, const RGBQUAD* const* colorMap, int width, int height,
                       const RGBQUAD* const* other, int otherWidth, int otherHeight, double performance[]);
    RGBQUAD** Division(Processor_Type hardware, const RGBQUAD* const* colorMap, int width, int height,
                       int constant);
    RGBQUAD** Division(Processor_Type hardware, const RGBQUAD* const* colorMap, int width, int height,
                       int constant, double performance[]);
    // Matrix Multiplication Variants
    RGBQUAD** MatrixMult(Processor_Type hardware, const RGBQUAD* const* colorMap, int width, int height,
                         const RGBQUAD* const* other, int otherWidth, int otherHeight);
    RGBQUAD** MatrixMult(Processor_Type hardware, const RGBQUAD* const* colorMap, int width, int height,
                         const RGBQUAD* const* other, int otherWidth, int otherHeight, double performance[]);
}

/**
* Private member functions
*/
namespace
{
    template<typename Type>
    Type** AllocateDynamic2D(int width, int height);
    template<typename Type>
    void DeallocateDynamic2D(Type** map, int width);
    uchar4* FlattenCopyDynamic2D(const RGBQUAD* const* map, int width, int height);
    RGBQUAD** UnflattenCopyToDynamic2D(unsigned char* map, int width, int height);
    RGBQUAD** UnflattenCopyToDynamic2D(uchar4* map, int width, int height);

    // CPU Variants of GPU functionality
    // CPU Generation of Maps
    unsigned char* CPUGenerateMap(int width, int height, const unsigned char& type);
    uchar4* CPUGenerateMap(int width, int height, const uchar4& type);
    unsigned char* CPUGenerateStaticMap(int width, int height);
    uchar4* CPUGenerateRandomMap(int width, int height);
    // CPU Modifications
    // Addition
    RGBQUAD** CPUAddition(const RGBQUAD* const* colorMap, int width, int height,
                          const RGBQUAD* const* other);
    RGBQUAD** CPUAddition(const RGBQUAD* const* colorMap, int width, int height,
                          int constant);
    // Subtraction
    RGBQUAD** CPUSubtract(const RGBQUAD* const* colorMap, int width, int height,
                          const RGBQUAD* const* other);
    RGBQUAD** CPUSubtract(const RGBQUAD* const* colorMap, int width, int height,
                          int constant);
    // Point-wise Multiplication
    RGBQUAD** CPUMultiply(const RGBQUAD* const* colorMap, int width, int height,
                          const RGBQUAD* const* other);
    RGBQUAD** CPUMultiply(const RGBQUAD* const* colorMap, int width, int height,
                          int constant);
    // Point-wise Division
    RGBQUAD** CPUDivision(const RGBQUAD* const* colorMap, int width, int height,
                          const RGBQUAD* const* other);
    RGBQUAD** CPUDivision(const RGBQUAD* const* colorMap, int width, int height,
                          int constant);
    // Matrix Multiplication
    RGBQUAD** CPUMatrixMult(const RGBQUAD* const* colorMap, int width, int height,
                            const RGBQUAD* const* other, int otherWidth, int otherHeight);
}

#endif //WRAPPER_H