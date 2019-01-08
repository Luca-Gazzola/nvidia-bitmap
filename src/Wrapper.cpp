// System includes
#include <random>
#include <chrono>
#include <iomanip>
#include <iostream>
// File includes
#include "../include/Wrapper.h"
#include "../include/BitmapException.h"
#include "../resource/cuda_error_check.cuh"

namespace Kernel
{
    // Initializes devices
    void CudaSetup()
    {
        // Device information data
        char deviceName[32];
        int* deviceCount = new int;
        CUdevice device;

        // Obtaining and retrieving device
        CUDA_CALL( cudaGetDeviceCount(deviceCount) );
        CUDA_DRIVER_CALL( cuDeviceGet(&device, 0) );
        CUDA_DRIVER_CALL( cuDeviceGetName(deviceName, 32, device) );
        //std::cout << "[Using device: " << deviceName << "]" << std::endl;
        CUDA_CALL( cudaSetDevice(0) );

        // Freeing allocated memory
        delete deviceCount;
    }

    // Wraps the static generation kernels from Kernel.cu so that
    // it remains C++ compatible without having to conform to using
    // C-style calls and functions
    RGBQUAD** StaticGeneration(Processor_Type hardware, int width, int height)
    {
        // Generate flattened 2D array of either 0's or 255's
        unsigned char* map = nullptr;

        switch(hardware)
        {
            case Processor_Type::GPU :
                map = LaunchStaticGeneration(width, height);
                break;

            case Processor_Type::CPU :
                map = CPUGenerateStaticMap(width, height);
                break;

            case Processor_Type::CPUtoGPU :
            {
                // Run CPU generation and save to a flattened array
                unsigned char *texture = CPUGenerateMap(width, height, *texture);

                // Launching and saving bitmap
                map = LaunchStaticGenerationTexture(width, height, texture);

                delete[] texture;
                break;
            }
        }

        if (map == nullptr)
            throw BitmapException("[GENERATION ERROR] Static Map generation has failed.");

        // Allocate new 2D dynamic array
        RGBQUAD** color = AllocateDynamic2D<RGBQUAD>(width, height);

        // Unflattening array to a 2D array
        for (int row = 0; row < height; ++row)
            for (int col = 0; col < width; ++col)
            {
                color[col][row].rgbBlue     = map[row * width + col];
                color[col][row].rgbGreen    = map[row * width + col];
                color[col][row].rgbRed      = map[row * width + col];
            }

        // Deallocate map
        free(map); // C-style call due to C-style allocation

        return color;
    }

    RGBQUAD** StaticGeneration(Processor_Type hardware, int width, int height,
                               double performance[])
    {
        // Timer --> easier to type
        typedef std::chrono::high_resolution_clock Time;
        typedef std::chrono::duration<double> double_seconds;

        // Generate flattened 2D array of either 0's or 255's
        unsigned char* map = nullptr;

        switch(hardware)
        {
            case Processor_Type::GPU :
                map = LaunchStaticGeneration(width, height, performance);
                break;

            case Processor_Type::CPU :
            {
                // Time measurement for CPU RNG initialization
                auto init_CPU_time_start = Time::now();

                // Run CPU generation with 0/255 values and save to a flattened array
                map = CPUGenerateStaticMap(width, height);

                // Finish timing
                auto init_CPU_time_finish = Time::now();
                double_seconds seconds = init_CPU_time_finish - init_CPU_time_start;
                performance[3] = seconds.count() * 1000; // CPUMapInit

                break;
            }

            case Processor_Type::CPUtoGPU :
            {
                // Time measurement for CPU RNG initialization
                auto init_CPU_time_start = Time::now();

                // Run CPU generation and save to a flattened array
                unsigned char *texture = CPUGenerateMap(width, height, *texture);

                // Finish timing
                auto init_CPU_time_finish = Time::now();
                double_seconds seconds = init_CPU_time_finish - init_CPU_time_start;
                performance[4] = seconds.count() * 1000; // CPUTextureInit

                // Launching and saving bitmap
                map = LaunchStaticGenerationTexture(width, height, texture, performance);
                delete[] texture;
                break;
            }
        }

        if (map == nullptr)
            throw BitmapException("[GENERATION ERROR] Static Map generation has failed.");

        // Time initialization for the RGB map build
        auto bitmap_map_build_start = Time::now();

        // Allocate new 2D dynamic array
        RGBQUAD** color = AllocateDynamic2D<RGBQUAD>(width, height);

        // Unflattening array to a 2D array
        for (int row = 0; row < height; ++row)
            for (int col = 0; col < width; ++col)
            {
                color[col][row].rgbBlue  = map[row * width + col];
                color[col][row].rgbGreen = map[row * width + col];
                color[col][row].rgbRed   = map[row * width + col];
            }

        // Finish timing
        auto bitmap_map_build_finish = Time::now();
        double_seconds seconds = bitmap_map_build_finish - bitmap_map_build_start;
        performance[5] = seconds.count() * 1000; // BitmapMapBuild

        // Deallocate map
        free(map); // C-style call due to C-style allocation

        return color;
    }

    // Wraps the random generation kernels from Kernel.cu so that
    // it remains C++ compatible without having to conform to using
    // C-style calls and functions
    RGBQUAD** RandomGeneration(Processor_Type hardware, int width, int height)
    {
        // Allocate new map array to contain the 3 color channels
        uchar4* maps = nullptr;

        switch (hardware)
        {
            case Processor_Type::GPU :
                maps = LaunchRandomGeneration(width, height);
                break;

            case Processor_Type::CPU :
            {
                // Run CPU generation with 0/255 values and save to a flattened array
                maps = CPUGenerateRandomMap(width, height);
                break;
            }

            case Processor_Type::CPUtoGPU :
            {
                // Run CPU generation and save to a flattened array
                uchar4* texture = nullptr;
                texture = CPUGenerateMap(width, height, *texture);

                // Launching and saving bitmap
                maps = LaunchRandomGenerationTexture(width, height, texture);
                delete[] texture;
                break;
            }
        }

        // Allocate RGBQUAD
        RGBQUAD** colorMap = AllocateDynamic2D<RGBQUAD>(width, height);
        RGBQUAD* colorReader = nullptr;

        // Unflattening array to a 2D array
        for (int row = 0; row < height; ++row)
            for (int col = 0; col < width; ++col)
            {
                colorReader = colorMap[col] + row;
                colorReader->rgbBlue  = maps[row * width + col].x;
                colorReader->rgbGreen = maps[row * width + col].y;
                colorReader->rgbRed   = maps[row * width + col].z;
            }

        // Deallocate C-style malloc
        free(maps);

        return colorMap;
    }

    RGBQUAD** RandomGeneration(Processor_Type hardware, int width, int height, double performance[])
    {
        // Timer --> easier to type
        typedef std::chrono::high_resolution_clock Time;
        typedef std::chrono::duration<double> double_seconds;

        // Allocate new map array to contain the 3 color channels
        uchar4* maps = nullptr;

        switch (hardware)
        {
            case Processor_Type::GPU :
                maps = LaunchRandomGeneration(width, height, performance);
                break;

            case Processor_Type::CPU :
            {
                // Time measurement for CPU RNG initialization
                auto init_CPU_time_start = Time::now();

                // Run CPU generation with 0/255 values and save to a flattened array
                maps = CPUGenerateRandomMap(width, height);

                // Finish timing
                auto init_CPU_time_finish = Time::now();
                double_seconds seconds = init_CPU_time_finish - init_CPU_time_start;
                performance[3] = seconds.count() * 1000; // CPUMapInit

                break;
            }

            case Processor_Type::CPUtoGPU :
            {
                // Time measurement for CPU RNG initialization
                auto init_CPU_time_start = Time::now();

                // Run CPU generation and save to a flattened array
                uchar4* texture = nullptr;
                texture = CPUGenerateMap(width, height, *texture);


                // Finish timing
                auto init_CPU_time_finish = Time::now();
                double_seconds seconds = init_CPU_time_finish - init_CPU_time_start;
                performance[4] = seconds.count() * 1000; // CPUTextureInit

                // Launching and saving bitmap
                maps = LaunchRandomGenerationTexture(width, height, texture, performance);
                delete[] texture;
                break;
            }
        }

        // Allocate RGBQUAD
        RGBQUAD** colorMap = AllocateDynamic2D<RGBQUAD>(width, height);
        RGBQUAD* colorReader = nullptr;

        // Unflattening array to a 2D array
        for (int row = 0; row < height; ++row)
            for (int col = 0; col < width; ++col)
            {
                colorReader = colorMap[col] + row;
                colorReader->rgbBlue  = maps[row * width + col].x;
                colorReader->rgbGreen = maps[row * width + col].y;
                colorReader->rgbRed   = maps[row * width + col].z;
            }

        // Deallocate C-style malloc
        free(maps);

        return colorMap;
    }
}



/**
 * Private member functions
 */
 // Keep these set of functions from being exposed outside
 // of this context
namespace
{
    // 2D matrix of allocated pointers
    template<typename Type>
    Type** AllocateDynamic2D(int width, int height)
    {
        Type** map = new Type* [width];
        for (int i = 0; i < width; ++i)
        {
            map[i] = new Type[height];
        }
        return map;
    }

    template<typename Type>
    void DeallocateDynamic2D(Type** map, int width)
    {
        for (int i = 0; i < width; ++i)
        {
            delete[] map[i];
        }
        delete[] map;
    }

    // Generates a static map with each element being a uchar
    // between 0-255
    unsigned char* CPUGenerateMap(int width, int height, const unsigned char& type)
    {
        // Initialize random number generator and distribution
        int seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937 generator(seed);
        std::uniform_int_distribution<unsigned char> distribution(0, 255);

        unsigned char* texture = new unsigned char[width * height];

        for (int j = 0; j < height; j++)
            for (int i = 0; i < width; i++)
            {
                texture[i + j * width] = distribution(generator);
            }

        return texture;
    }

    // Generates a multi channel map with each element being
    // a component of uchar4, each between 0-255
    uchar4* CPUGenerateMap(int width, int height, const uchar4& type)
    {
        // Initialize random number generator and distribution
        int seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937 generator(seed);
        std::uniform_int_distribution<unsigned char> distribution(0, 255);

        uchar4* texture = new uchar4[width * height];

        for (int j = 0; j < height; j++)
            for (int i = 0; i < width; i++)
            {
                uchar4& texel = texture[i + j * width];
                texel.x = distribution(generator);
                texel.y = distribution(generator);
                texel.z = distribution(generator);
                texel.w = 0;
            }

        return texture;
    }

    unsigned char* CPUGenerateStaticMap(int width, int height)
    {
        // Initialize random number generator and distribution
        int seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937 generator(seed);
        std::uniform_int_distribution<unsigned char> distribution(0, 255);

        unsigned char* texture = new unsigned char[width * height];

        for (int j = 0; j < height; j++)
            for (int i = 0; i < width; i++)
            {
                unsigned char element = distribution(generator);
                texture[i + j * width] = (element < 127) ? (unsigned char)0 : (unsigned char)255;
            }

        return texture;
    }

    uchar4* CPUGenerateRandomMap(int width, int height)
    {
        // Initialize random number generator and distribution
        int seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::mt19937 generator(seed);
        std::uniform_int_distribution<unsigned char> distribution(0, 255);

        uchar4* texture = new uchar4[width*height];

        for (int j = 0; j < height; j++)
            for (int i = 0; i < width; i++)
            {
                texture[i + j * width].x = distribution(generator); // Red
                texture[i + j * width].y = distribution(generator); // Green
                texture[i + j * width].z = distribution(generator); // Blue
                texture[i + j * width].w = 0;                       // Reserved
            }

        return texture;
    }
}
