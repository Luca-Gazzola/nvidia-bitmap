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

            default:
            {
                throw BitmapException("[STATIC_GENERATION] Specified Processor_Type doesn't exist");
            }
        }

        if (map == nullptr)
            throw BitmapException("[GENERATION ERROR] Static Map generation has failed.");

        // Convert back to RGBQUAD (from uchar4)
        RGBQUAD** colorMap = UnflattenCopyToDynamic2D(map, width, height);

        // Deallocate map
        free(map); // C-style call due to C-style allocation

        return colorMap;
    }

    RGBQUAD** StaticGeneration(Processor_Type hardware, int width, int height,
                               double performance[])
    {
        // Timer --> easier to type
        typedef std::chrono::high_resolution_clock Time;
        typedef std::chrono::duration<double> double_seconds;

        // Time initialization for the RGB map build
        auto bitmap_map_build_start = Time::now();

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

            default:
            {
                throw BitmapException("[STATIC_GENERATION] Specified Processor_Type doesn't exist");
            }
        }

        if (map == nullptr)
            throw BitmapException("[GENERATION ERROR] Static Map generation has failed.");

        // Convert back to RGBQUAD (from uchar4)
        RGBQUAD** colorMap = UnflattenCopyToDynamic2D(map, width, height);

        // Deallocate map
        free(map); // C-style call due to C-style allocation

        // Finish timing
        auto bitmap_map_build_finish = Time::now();
        double_seconds seconds = bitmap_map_build_finish - bitmap_map_build_start;
        performance[5] = seconds.count() * 1000; // BitmapMapBuild

        return colorMap;
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

            default:
            {
                throw BitmapException("[RANDOM_GENERATION] Specified Processor_Type doesn't exist");
            }
        }

        // Convert back to RGBQUAD (from uchar4)
        RGBQUAD** colorMap = UnflattenCopyToDynamic2D(maps, width, height);

        // Deallocate C-style malloc
        free(maps);

        return colorMap;
    }

    RGBQUAD** RandomGeneration(Processor_Type hardware, int width, int height, double performance[])
    {
        // Timer --> easier to type
        typedef std::chrono::high_resolution_clock Time;
        typedef std::chrono::duration<double> double_seconds;

        // Time initialization for the RGB map build
        auto bitmap_map_build_start = Time::now();

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

            default:
            {
                throw BitmapException("[RANDOM_GENERATION] Specified Processor_Type doesn't exist");
            }
        }

        // Convert back to RGBQUAD (from uchar4)
        RGBQUAD** colorMap = UnflattenCopyToDynamic2D(maps, width, height);

        // Deallocate C-style malloc
        free(maps);

        // Finish timing
        auto bitmap_map_build_finish = Time::now();
        double_seconds seconds = bitmap_map_build_finish - bitmap_map_build_start;
        performance[5] = seconds.count() * 1000; // BitmapMapBuild

        return colorMap;
    }

    RGBQUAD** Addition(Processor_Type hardware, const RGBQUAD* const* colorMap, int width, int height,
                       const RGBQUAD* const* other, int otherWidth, int otherHeight)
    {
        return nullptr;
    }

    RGBQUAD** Addition(Processor_Type hardware, const RGBQUAD* const* colorMap, int width, int height,
                       const RGBQUAD* const* other, int otherWidth, int otherHeight, double performance[])
    {
        // Timer --> easier to type
        typedef std::chrono::high_resolution_clock Time;
        typedef std::chrono::duration<double> double_seconds;

        // Time initialization for the RGB map build
        auto bitmap_map_build_start = Time::now();

        // Holds newly constructed RGB map
        RGBQUAD** map = nullptr;

        if (width != otherWidth || height != otherHeight)
            throw BitmapException("[ADDITION] Bitmaps are not the same dimensions");

        switch(hardware)
        {
            case Processor_Type::GPU :
            {
                // Create two flat copies of each map to prep for GPU computation
                uchar4* thisMap = FlattenCopyDynamic2D(colorMap, width, height);
                uchar4* otherMap = FlattenCopyDynamic2D(other, otherWidth, otherHeight);

                // Run GPU Matrix Mult
                uchar4* result = LaunchAddition(thisMap, width, height,
                                                otherMap, otherWidth, otherHeight, performance);

                if (!result)
                    throw BitmapException("[MATRIX_MULT] GPU computation failed");

                // Deallocate flattened maps post-GPU use
                delete[] thisMap;
                delete[] otherMap;

                // Convert back to RGBQUAD (from uchar4)
                map = UnflattenCopyToDynamic2D(result, width, height);

                // Deallocate resultant matrix (C-style)
                free(result);

                break;
            }

            case Processor_Type::CPU :
            {
                // Time measurement for CPU RNG initialization
                auto init_CPU_time_start = Time::now();

                // Run CPU generation with 0/255 values and save to a flattened array
                map = CPUAddition(colorMap, width, height, other, otherWidth, otherHeight);

                // Finish timing
                auto init_CPU_time_finish = Time::now();
                double_seconds seconds = init_CPU_time_finish - init_CPU_time_start;
                performance[3] = seconds.count() * 1000; // CPUMapInit

                break;
            }

            case Processor_Type::CPUtoGPU :
            {
                throw BitmapException("[MATRIX_MULT] NOT IMPLEMENTED");
            }

            default:
            {
                throw BitmapException("[MATRIX_MULT] Specified Processor_Type doesn't exist");
            }
        }

        // Finish timing
        auto bitmap_map_build_finish = Time::now();
        double_seconds seconds = bitmap_map_build_finish - bitmap_map_build_start;
        performance[5] = seconds.count() * 1000; // BitmapMapBuild

        return map;
    }

    // Wraps the matrix multiplication kernels from Kernel.cu so
    // that it remains C++ compatible without having to use C-style
    // calls/functions outside of this file
    RGBQUAD** MatrixMult(Processor_Type hardware, const RGBQUAD* const* colorMap, int width, int height,
                         const RGBQUAD* const* other, int otherWidth, int otherHeight)
    {
        // Holds newly constructed RGB map
        RGBQUAD** map = nullptr;

        switch(hardware)
        {
            case Processor_Type::GPU :
            {
                // Create two flat copies of each map to prep for GPU computation
                uchar4* thisMap = FlattenCopyDynamic2D(colorMap, width, height);
                uchar4* otherMap = FlattenCopyDynamic2D(other, otherWidth, otherHeight);

                // Run GPU Matrix Mult
                uchar4* result = LaunchMatrixMult(thisMap, width, height,
                                                  otherMap, otherWidth, otherHeight);

                if (!result)
                    throw BitmapException("[MATRIX_MULT] GPU computation failed");

                // Deallocate flattened maps post-GPU use
                delete[] thisMap;
                delete[] otherMap;

                // Convert back to RGBQUAD (from uchar4)
                map = UnflattenCopyToDynamic2D(result, height, otherWidth);

                // Deallocate resultant matrix (C-style)
                free(result);

                break;
            }

            case Processor_Type::CPU :
            {
                // Run CPU generation with 0/255 values and save to a flattened array
                map = CPUMatrixMult(colorMap, width, height, other, otherWidth, otherHeight);
                break;
            }

            case Processor_Type::CPUtoGPU :
            {
                throw BitmapException("[MATRIX_MULT] NOT IMPLEMENTED");
            }

            default:
            {
                throw BitmapException("[MATRIX_MULT] Specified Processor_Type doesn't exist");
            }
        }

        return map;

    }

    RGBQUAD** MatrixMult(Processor_Type hardware, const RGBQUAD* const* colorMap, int width, int height,
                         const RGBQUAD* const* other, int otherWidth, int otherHeight, double performance[])
    {
        // Timer --> easier to type
        typedef std::chrono::high_resolution_clock Time;
        typedef std::chrono::duration<double> double_seconds;

        // Time initialization for the RGB map build
        auto bitmap_map_build_start = Time::now();

        // Holds newly constructed RGB map
        RGBQUAD** map = nullptr;

        switch(hardware)
        {
            case Processor_Type::GPU :
            {
                // Create two flat copies of each map to prep for GPU computation
                uchar4* thisMap = FlattenCopyDynamic2D(colorMap, width, height);
                uchar4* otherMap = FlattenCopyDynamic2D(other, otherWidth, otherHeight);

                // Run GPU Matrix Mult
                uchar4* result = LaunchMatrixMult(thisMap, width, height,
                                                  otherMap, otherWidth, otherHeight, performance);

                if (!result)
                    throw BitmapException("[MATRIX_MULT] GPU computation failed");

                // Deallocate flattened maps post-GPU use
                delete[] thisMap;
                delete[] otherMap;

                // Convert back to RGBQUAD (from uchar4)
                map = UnflattenCopyToDynamic2D(result, height, otherWidth);

                // Deallocate resultant matrix (C-style)
                free(result);

                break;
            }

            case Processor_Type::CPU :
            {
                // Time measurement for CPU RNG initialization
                auto init_CPU_time_start = Time::now();

                // Run CPU generation with 0/255 values and save to a flattened array
                map = CPUMatrixMult(colorMap, width, height, other, otherWidth, otherHeight);

                // Finish timing
                auto init_CPU_time_finish = Time::now();
                double_seconds seconds = init_CPU_time_finish - init_CPU_time_start;
                performance[3] = seconds.count() * 1000; // CPUMapInit

                break;
            }

            case Processor_Type::CPUtoGPU :
            {
                throw BitmapException("[MATRIX_MULT] NOT IMPLEMENTED");
            }

            default:
            {
                throw BitmapException("[MATRIX_MULT] Specified Processor_Type doesn't exist");
            }
        }

        // Finish timing
        auto bitmap_map_build_finish = Time::now();
        double_seconds seconds = bitmap_map_build_finish - bitmap_map_build_start;
        performance[5] = seconds.count() * 1000; // BitmapMapBuild

        return map;
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

    uchar4* FlattenCopyDynamic2D(const RGBQUAD* const* map, int width, int height)
    {
        uchar4* newMap = new uchar4[width * height];

        for (int j = 0; j < height; ++j)
            for (int i = 0; i < width; ++i)
            {
                uchar4& element = newMap[i + j * width];
                element.x = map[i][j].rgbRed;
                element.y = map[i][j].rgbGreen;
                element.z = map[i][j].rgbBlue;
                element.w = map[i][j].rgbReserved;
            }

        return newMap;
    }

    RGBQUAD** UnflattenCopyToDynamic2D(unsigned char* map, int width, int height)
    {
        // Allocate RGBQUAD
        RGBQUAD** colorMap = AllocateDynamic2D<RGBQUAD>(width, height);
        RGBQUAD* colorReader = nullptr;

        // Unflattening array to a 2D array
        for (int j = 0; j < height; ++j)
            for (int i = 0; i < width; ++i)
            {
                colorReader = colorMap[i] + j;
                colorReader->rgbBlue     = map[i + j * width];
                colorReader->rgbGreen    = map[i + j * width];
                colorReader->rgbRed      = map[i + j * width];
                colorReader->rgbReserved = static_cast<unsigned char>(0);
            }

        return colorMap;
    }

    RGBQUAD** UnflattenCopyToDynamic2D(uchar4* map, int width, int height)
    {
        // Allocate RGBQUAD
        RGBQUAD** colorMap = AllocateDynamic2D<RGBQUAD>(width, height);
        RGBQUAD* colorReader = nullptr;

        // Unflattening array to a 2D array
        for (int j = 0; j < height; ++j)
            for (int i = 0; i < width; ++i)
            {
                colorReader = colorMap[i] + j;
                colorReader->rgbBlue     = map[i + j * width].x;
                colorReader->rgbGreen    = map[i + j * width].y;
                colorReader->rgbRed      = map[i + j * width].z;
                colorReader->rgbReserved = map[i + j * width].w;
            }

        return colorMap;
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
                texture[i + j * width] = (element < 127) ?
                                         static_cast<unsigned char>(0) : static_cast<unsigned char>(255);
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

    // Adds two matrices together using only the CPU, saves it into another
    RGBQUAD** CPUAddition(const RGBQUAD* const* colorMap, int width, int height,
                          const RGBQUAD* const* other, int otherWidth, int otherHeight)
    {
        RGBQUAD** resultant = AllocateDynamic2D<RGBQUAD>(width, height);

        for (int j = 0; j < height; j++)
            for (int i = 0; i < width; i++)
            {
                int tempR = static_cast<int>(colorMap[i][j].rgbRed) + static_cast<int>(other[i][j].rgbRed);
                int tempG = static_cast<int>(colorMap[i][j].rgbGreen) + static_cast<int>(other[i][j].rgbGreen);
                int tempB = static_cast<int>(colorMap[i][j].rgbBlue) + static_cast<int>(other[i][j].rgbBlue);
                resultant[i][j].rgbRed   = (tempR < 255) ? static_cast<unsigned char>(tempR)
                                                      : static_cast<unsigned char>(255);
                resultant[i][j].rgbGreen = (tempG < 255) ? static_cast<unsigned char>(tempG)
                                                      : static_cast<unsigned char>(255);
                resultant[i][j].rgbBlue  = (tempB < 255) ? static_cast<unsigned char>(tempB)
                                                      : static_cast<unsigned char>(255);
            }

        return resultant;
    }

    // Matrix Multiplication between to matrices done on the CPU, saves into another
    RGBQUAD** CPUMatrixMult(const RGBQUAD* const* colorMap, int width, int height,
                            const RGBQUAD* const* other, int otherWidth, int otherHeight)
    {
        // REMINDER: Matrices are defined as "Height x Width" not "Width x Height",
        // hence the slight confusion and bugs that have occurred here

        // Matrix product restriction
        if (height != otherWidth)
            throw BitmapException("[MATRIX_MULT] Dimensions (height and other's width) are not compatible");

        RGBQUAD** result = AllocateDynamic2D<RGBQUAD>(height, otherWidth);

        double heightDownscale = 1.0 / static_cast<double>(height);
        double charDownscale = 1.0 / 255.0;

        // Iterate through resultant
        for (int j = 0; j < height; j++)
            for (int i = 0; i < otherWidth; i++)
            {
                double sumR = 0.0;
                double sumB = 0.0;
                double sumG = 0.0;

                // Iterate through the two input matrices
                for (int k = 0; k < width; k++)
                {
                    sumR += static_cast<double>(colorMap[k][j].rgbRed) * static_cast<double>(other[i][k].rgbRed) *
                            heightDownscale * charDownscale;
                    sumG += static_cast<double>(colorMap[k][j].rgbGreen) * static_cast<double>(other[i][k].rgbGreen) *
                            heightDownscale * charDownscale;
                    sumB += static_cast<double>(colorMap[k][j].rgbBlue) * static_cast<double>(other[i][k].rgbBlue) *
                            heightDownscale * charDownscale;
                }

                result[i][j].rgbRed      = static_cast<unsigned char>(sumR);
                result[i][j].rgbGreen    = static_cast<unsigned char>(sumG);
                result[i][j].rgbBlue     = static_cast<unsigned char>(sumB);
                result[i][j].rgbReserved = static_cast<unsigned char>(0);
            }

        return result;
    }
}
