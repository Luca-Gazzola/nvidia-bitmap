#ifndef WRAPPER_H
#define WRAPPER_H
// File includes
#include "../include/Kernel.cuh"
#include "../resource/bitmap_resources.h"

namespace Kernel
{
    void CudaSetup();

    RGBQUAD** StaticGeneration(Processor_Type hardware, int width, int height);
    RGBQUAD** StaticGeneration(Processor_Type hardware, int width, int height, double performance[]);
    RGBQUAD** RandomGeneration(Processor_Type hardware, int width, int height);
    RGBQUAD** RandomGeneration(Processor_Type hardware, int width, int height, double performance[]);
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

    unsigned char* CPUGenerateMap(int width, int height, const unsigned char& type);
    uchar4* CPUGenerateMap(int width, int height, const uchar4& type);
    unsigned char* CPUGenerateStaticMap(int width, int height);
    uchar4* CPUGenerateRandomMap(int width, int height);
}

#endif //WRAPPER_H