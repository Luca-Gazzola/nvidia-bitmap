// System includes
#include <fstream>
#include <iostream>
#include <random>
#include <chrono>
#include <sys/stat.h>
// File includes
#include "../include/Bitmap.h"
#include "../include/BitmapException.h"
#include "../include/Wrapper.h"



/**
 * PUBLIC METHODS
 */

/**
 * Constructors
 */
 // Default Constructor
 Bitmap::Bitmap(bool track_performance)
    : m_color(nullptr), m_track_performance(track_performance)
{
    // Setup CUDA for class
    if (!s_cuda_setup)
    {
        Kernel::CudaSetup();
        s_cuda_setup = true;
    }

    // Setup Bitmap structs
    SetBitmapInformation(256, 256);
    SetPerformanceInformation();
    SetDefaultHardware(Processor_Type::CPU);

    // Allocated after info setup to prevent leaks
    m_color = AllocateColorMap(256, 256);
}

// Square Bitmap Constructor
Bitmap::Bitmap(int side, bool track_performance)
    : m_color(nullptr), m_track_performance(track_performance)
{
    // Setup CUDA for class
    if (!s_cuda_setup)
    {
        Kernel::CudaSetup();
        s_cuda_setup = true;
    }

    // Ensure that the pixel amount is scalable and isn't <=0
    if (side % 4 != 0)
        throw BitmapException("Side must be a multiple of 4");
    if (side <= 0)
        throw BitmapException("Side must be an integer greater than 0");

    // Setup Bitmap structs
    SetBitmapInformation(side, side);
    SetPerformanceInformation();
    SetDefaultHardware(Processor_Type::CPU);

    // Allocated after conditional to prevent leaks
    m_color = AllocateColorMap(side, side);
}

// Dimensional Bitmap
Bitmap::Bitmap(int width, int height, bool track_performance)
    : m_color(nullptr), m_track_performance(track_performance)
{
    // Setup CUDA for class
    if (!s_cuda_setup)
    {
        Kernel::CudaSetup();
        s_cuda_setup = true;
    }

    // Ensure that the pixel amount is scalable and isn't <=0
    if (width % 4 != 0 || height % 4 != 0)
        throw BitmapException("Width/Height must be a multiple of 4");
    if (width <= 0 || height <= 0)
        throw BitmapException("Width/Height must be an integer greater than 0");

    // Setup Bitmap structs
    SetBitmapInformation(width, height);
    SetPerformanceInformation();
    SetDefaultHardware(Processor_Type::CPU);

    // Allocated after conditional to prevent leaks
    m_color = AllocateColorMap(width, height);
}

// Copy Constructor
Bitmap::Bitmap(const Bitmap& other)
    : m_color(nullptr), m_track_performance(other.m_track_performance),
      m_bmpHead(other.m_bmpHead), m_bmpInfo(other.m_bmpInfo),
      m_defaultHardware(other.m_defaultHardware)
{
    SetPerformanceInformation();

    // No CUDA setup necessary since it is copy
    // constructing from another constructed Bitmap
    m_color = CopyColorMap(other.m_color, other.m_bmpInfo.biWidth, other.m_bmpInfo.biWidth);
}


/**
 * Destructor
 */
Bitmap::~Bitmap()
{
    if (m_color)
        DeleteColorMap();
}


/**
 * Operator Overloads
 */
 // Assignment
void Bitmap::operator= (const Bitmap& other)
{
    DeleteColorMap();
    m_track_performance = other.m_track_performance;
    m_bmpHead = other.m_bmpHead;
    m_bmpInfo = other.m_bmpInfo;

    SetPerformanceInformation();

    // No CUDA setup necessary since it is copy
    // constructing from another constructed Bitmap
    m_color = CopyColorMap(other.m_color, other.m_bmpInfo.biWidth, other.m_bmpInfo.biWidth);
}

// Addition by Bitmap and by constant
Bitmap Bitmap::operator+ (const Bitmap& other)
{
    Bitmap clone = Clone();
    clone.MakeBitmap(Bitmap_Modify_Type::Add, other, m_defaultHardware);
    return clone;
}

Bitmap Bitmap::operator+ (int constant)
{
    Bitmap clone = Clone();
    clone.MakeBitmap(Bitmap_Modify_Type::Add, constant, m_defaultHardware);
    return clone;
}

Bitmap Bitmap::operator- (const Bitmap& other)
{
    Bitmap clone = Clone();
    clone.MakeBitmap(Bitmap_Modify_Type::Subtract, other, m_defaultHardware);
    return clone;
}

Bitmap Bitmap::operator- (int constant)
{
    Bitmap clone = Clone();
    clone.MakeBitmap(Bitmap_Modify_Type::Subtract, constant, m_defaultHardware);
    return clone;
}

Bitmap Bitmap::operator* (const Bitmap& other)
{
    Bitmap clone = Clone();
    clone.MakeBitmap(Bitmap_Modify_Type::Multiply, other, m_defaultHardware);
    return clone;
}

Bitmap Bitmap::operator* (int constant)
{
    Bitmap clone = Clone();
    clone.MakeBitmap(Bitmap_Modify_Type::Multiply, constant, m_defaultHardware);
    return clone;
}

Bitmap Bitmap::operator/ (const Bitmap& other)
{
    Bitmap clone = Clone();
    clone.MakeBitmap(Bitmap_Modify_Type::Divide, other, m_defaultHardware);
    return clone;
}

Bitmap Bitmap::operator/ (int constant)
{
    Bitmap clone = Clone();
    clone.MakeBitmap(Bitmap_Modify_Type::Divide, constant, m_defaultHardware);
    return clone;
}



/**
* Bitmap Generator
*/
// Generates Bitmap with a specified generation
// Returns the filepath of the generated Bitmap
void Bitmap::MakeBitmap(Bitmap_Type gen, Processor_Type hardware)
{
    // Reset performance
    SetPerformanceInformation();

    // Generates map depending on the Bitmap_Type
    switch (gen)
    {
        case Bitmap_Type::Static    : GenerateStatic(hardware);     return;
        case Bitmap_Type::Random    : GenerateRandom(hardware);     return;
        default: throw BitmapException("[MAKE_BITMAP] Invalid Bitmap_Type");
    }
}

// Takes another bitmap and modifies it with
// this bitmap in various ways
// Returns the filepath of the generated Bitmap
void Bitmap::MakeBitmap(Bitmap_Modify_Type mod, const Bitmap& other,
                        Processor_Type hardware)
{
    // Reset performance
    SetPerformanceInformation();

    switch (mod)
    {
        case Bitmap_Modify_Type::Add       : GenerateAdd(hardware, other);         return;
        case Bitmap_Modify_Type::Subtract  : GenerateSubtract(hardware, other);    return;
        case Bitmap_Modify_Type::Multiply  : GenerateMultiply(hardware, other);    return;
        case Bitmap_Modify_Type::Divide    : GenerateDivide(hardware, other);      return;
        case Bitmap_Modify_Type::MatrixMult: GenerateMatrixMult(hardware, other);  return;
        default: throw BitmapException("[MAKE_BITMAP] Invalid Bitmap_Modify_Type");
    }
}

void Bitmap::MakeBitmap(Bitmap_Modify_Type mod, int constant,
                        Processor_Type hardware)
{
    // Reset performance
    SetPerformanceInformation();

    switch (mod)
    {
        case Bitmap_Modify_Type::Add       : GenerateAdd(hardware, constant);      return;
        case Bitmap_Modify_Type::Subtract  : GenerateSubtract(hardware, constant); return;
        case Bitmap_Modify_Type::Multiply  : GenerateMultiply(hardware, constant); return;
        case Bitmap_Modify_Type::Divide    : GenerateDivide(hardware, constant);   return;
        default: throw BitmapException("[MAKE_BITMAP] Invalid Bitmap_Modify_Type");
    }
}

/**
 * Public Methods
 */
std::string Bitmap::DrawBitmap(std::string fileName)
{
    // Output color map to file output (.bmp file)
    const std::string filePath("./bitmaps/" + fileName + ".bmp");
    std::ofstream file = CreateFile(filePath, fileName);

    // Write to opened file
    if (!m_track_performance)
        WriteData(file);
    else
        WriteData(file, m_performance.performanceArray);

    // Clean-up
    file.close();

    return filePath;
}

const int Bitmap::Width() noexcept { return m_bmpInfo.biWidth; }
const int Bitmap::Height() noexcept { return m_bmpInfo.biHeight; }

// Set default hardware
void Bitmap::SetDefaultHardware(const Processor_Type hardware) noexcept
{
    m_defaultHardware = hardware;
}

// Performance timings
const double& Bitmap::Performance(Bitmap_Performance::GPU timeType) noexcept
{
    switch(timeType)
    {
        case Bitmap_Performance::GPU::KernelInit      : return m_performance.gpuKernelInit;
        case Bitmap_Performance::GPU::KernelLaunch    : return m_performance.gpuKernelLaunch;
        case Bitmap_Performance::GPU::MapCopy         : return m_performance.gpuMapCopy;
        default:
            throw BitmapException("[PERFORMANCE] Invalid Bitmap_Performance::GPU type: " + static_cast<int>(timeType));
    }
}

const double& Bitmap::Performance(Bitmap_Performance::CPU timeType) noexcept
{
    switch(timeType)
    {
        case Bitmap_Performance::CPU::MapInit         : return m_performance.cpuMapInit;
        case Bitmap_Performance::CPU::TextureInit     : return m_performance.cpuTextureInit;
        default:
            throw BitmapException("[PERFORMANCE] Invalid Bitmap_Performance::CPU type: " + static_cast<int>(timeType));
    }
}

const double& Bitmap::Performance(Bitmap_Performance::Bitmap timeType) noexcept
{
    switch(timeType)
    {
        case Bitmap_Performance::Bitmap::MapBuild     : return m_performance.bitmapMapBuild;
        case Bitmap_Performance::Bitmap::FileBuild    : return m_performance.bitmapFileBuild;
        default:
            throw BitmapException("[PERFORMANCE] Invalid Bitmap_Performance::Bitmap type: " + static_cast<int>(timeType));
    }
}


const double* Bitmap::Performance() noexcept
{
    return m_performance.performanceArray;
}



/**
 * PRIVATE
 */

/**
 * Initializing statics
 */
 bool Bitmap::s_cuda_setup = false;


/**
 * Member functions
 */
void Bitmap::GenerateStatic(Processor_Type hardware)
{
    DeleteColorMap();
    if (!m_track_performance)
        m_color = Kernel::StaticGeneration(hardware, m_bmpInfo.biWidth,
                                           m_bmpInfo.biHeight);
    else
        m_color = Kernel::StaticGeneration(hardware, m_bmpInfo.biWidth,
                                           m_bmpInfo.biHeight,
                                           m_performance.performanceArray);
}

void Bitmap::GenerateRandom(Processor_Type hardware)
{
    DeleteColorMap();
    if (!m_track_performance)
        m_color = Kernel::RandomGeneration(hardware, m_bmpInfo.biWidth,
                                           m_bmpInfo.biHeight);
    else
        m_color = Kernel::RandomGeneration(hardware, m_bmpInfo.biWidth,
                                           m_bmpInfo.biHeight,
                                           m_performance.performanceArray);
}

void Bitmap::GenerateAdd(Processor_Type hardware, const Bitmap& other)
{
    if (!m_color || !other.m_color)
        throw BitmapException("[ADDITION] This object's bitmap or the other's bitmap is missing (nullptr)");

    RGBQUAD** resultant = nullptr;

    if (!m_track_performance)
        resultant = Kernel::Addition(hardware, m_color, m_bmpInfo.biWidth, m_bmpInfo.biHeight,
                                     other.m_color, other.m_bmpInfo.biWidth, other.m_bmpInfo.biHeight);
    else
        resultant = Kernel::Addition(hardware, m_color, m_bmpInfo.biWidth, m_bmpInfo.biHeight,
                                     other.m_color, other.m_bmpInfo.biWidth, other.m_bmpInfo.biHeight,
                                     m_performance.performanceArray);

    DeleteColorMap();
    SetBitmapInformation(m_bmpInfo.biWidth, m_bmpInfo.biHeight);
    m_color = resultant;
}

void Bitmap::GenerateAdd(Processor_Type hardware, int constant)
{
    if (!m_color)
        throw BitmapException("[ADDITION] This object's bitmap is missing (nullptr)");

    // If you try to add with a negative number, GenerateSubtract
    // will be called instead, with the positive variant of the
    // constant number. This is more of a safety catch than
    // anything, and I highly doubt I need this.
    if (constant < 0)
    {
        GenerateSubtract(hardware, -constant);
        return;
    }

    RGBQUAD** resultant = nullptr;

    if (!m_track_performance)
        resultant = Kernel::Addition(hardware, m_color, m_bmpInfo.biWidth, m_bmpInfo.biHeight,
                                     constant);
    else
        resultant = Kernel::Addition(hardware, m_color, m_bmpInfo.biWidth, m_bmpInfo.biHeight,
                                     constant, m_performance.performanceArray);

    DeleteColorMap();
    SetBitmapInformation(m_bmpInfo.biWidth, m_bmpInfo.biHeight);
    m_color = resultant;
}

void Bitmap::GenerateSubtract(Processor_Type hardware, const Bitmap& other)
{
    if (!m_color || !other.m_color)
        throw BitmapException("[ADDITION] This object's bitmap or the other's bitmap is missing (nullptr)");

    RGBQUAD** resultant = nullptr;

    if (!m_track_performance)
        resultant = Kernel::Subtract(hardware, m_color, m_bmpInfo.biWidth, m_bmpInfo.biHeight,
                                     other.m_color, other.m_bmpInfo.biWidth, other.m_bmpInfo.biHeight);
    else
        resultant = Kernel::Subtract(hardware, m_color, m_bmpInfo.biWidth, m_bmpInfo.biHeight,
                                     other.m_color, other.m_bmpInfo.biWidth, other.m_bmpInfo.biHeight,
                                     m_performance.performanceArray);

    DeleteColorMap();
    SetBitmapInformation(m_bmpInfo.biWidth, m_bmpInfo.biHeight);
    m_color = resultant;
}

void Bitmap::GenerateSubtract(Processor_Type hardware, int constant)
{
    if (!m_color)
        throw BitmapException("[ADDITION] This object's bitmap is missing (nullptr)");

    // If you try to add with a negative number, GenerateSubtract
    // will be called instead, with the positive variant of the
    // constant number. This is more of a safety catch than
    // anything, and I highly doubt I need this.
    if (constant < 0)
    {
        GenerateAdd(hardware, -constant);
        return;
    }

    RGBQUAD** resultant = nullptr;

    if (!m_track_performance)
        resultant = Kernel::Subtract(hardware, m_color, m_bmpInfo.biWidth, m_bmpInfo.biHeight,
                                     constant);
    else
        resultant = Kernel::Subtract(hardware, m_color, m_bmpInfo.biWidth, m_bmpInfo.biHeight,
                                     constant, m_performance.performanceArray);

    DeleteColorMap();
    SetBitmapInformation(m_bmpInfo.biWidth, m_bmpInfo.biHeight);
    m_color = resultant;
}

void Bitmap::GenerateMultiply(Processor_Type hardware, const Bitmap& other)
{
    if (!m_color || !other.m_color)
        throw BitmapException("[MULTIPLY] This object's bitmap or the other's bitmap is missing (nullptr)");

    RGBQUAD** resultant = nullptr;

    if (!m_track_performance)
        resultant = Kernel::Multiply(hardware, m_color, m_bmpInfo.biWidth, m_bmpInfo.biHeight,
                                     other.m_color, other.m_bmpInfo.biWidth, other.m_bmpInfo.biHeight);
    else
        resultant = Kernel::Multiply(hardware, m_color, m_bmpInfo.biWidth, m_bmpInfo.biHeight,
                                     other.m_color, other.m_bmpInfo.biWidth, other.m_bmpInfo.biHeight,
                                     m_performance.performanceArray);

    DeleteColorMap();
    SetBitmapInformation(m_bmpInfo.biWidth, m_bmpInfo.biHeight);
    m_color = resultant;
}

void Bitmap::GenerateMultiply(Processor_Type hardware, int constant)
{
    if (!m_color)
        throw BitmapException("[MULTIPLY] This object's bitmap is missing (nullptr)");

    RGBQUAD** resultant = nullptr;

    if (!m_track_performance)
        resultant = Kernel::Multiply(hardware, m_color, m_bmpInfo.biWidth, m_bmpInfo.biHeight,
                                     constant);
    else
        resultant = Kernel::Multiply(hardware, m_color, m_bmpInfo.biWidth, m_bmpInfo.biHeight,
                                     constant, m_performance.performanceArray);

    DeleteColorMap();
    SetBitmapInformation(m_bmpInfo.biWidth, m_bmpInfo.biHeight);
    m_color = resultant;
}

void Bitmap::GenerateDivide(Processor_Type hardware, const Bitmap& other)
{
    if (!m_color || !other.m_color)
        throw BitmapException("[DIVISION] This object's bitmap or the other's bitmap is missing (nullptr)");

    RGBQUAD** resultant = nullptr;

    if (!m_track_performance)
        resultant = Kernel::Division(hardware, m_color, m_bmpInfo.biWidth, m_bmpInfo.biHeight,
                                     other.m_color, other.m_bmpInfo.biWidth, other.m_bmpInfo.biHeight);
    else
        resultant = Kernel::Division(hardware, m_color, m_bmpInfo.biWidth, m_bmpInfo.biHeight,
                                     other.m_color, other.m_bmpInfo.biWidth, other.m_bmpInfo.biHeight,
                                     m_performance.performanceArray);

    DeleteColorMap();
    SetBitmapInformation(m_bmpInfo.biWidth, m_bmpInfo.biHeight);
    m_color = resultant;
}

void Bitmap::GenerateDivide(Processor_Type hardware, int constant)
{
    if (!m_color)
        throw BitmapException("[DIVISION] This object's bitmap is missing (nullptr)");

    RGBQUAD** resultant = nullptr;

    if (!m_track_performance)
        resultant = Kernel::Division(hardware, m_color, m_bmpInfo.biWidth, m_bmpInfo.biHeight,
                                     constant);
    else
        resultant = Kernel::Division(hardware, m_color, m_bmpInfo.biWidth, m_bmpInfo.biHeight,
                                     constant, m_performance.performanceArray);

    DeleteColorMap();
    SetBitmapInformation(m_bmpInfo.biWidth, m_bmpInfo.biHeight);
    m_color = resultant;
}

void Bitmap::GenerateMatrixMult(Processor_Type hardware, const Bitmap& other)
{
    if (!m_color || !other.m_color)
        throw BitmapException("[MATRIX_MULT] This object's bitmap or the other's bitmap is missing (nullptr)");

    RGBQUAD** resultant = nullptr;

    if (!m_track_performance)
        resultant = Kernel::MatrixMult(hardware, m_color, m_bmpInfo.biWidth, m_bmpInfo.biHeight,
                                       other.m_color, other.m_bmpInfo.biWidth, other.m_bmpInfo.biHeight);
    else
        resultant = Kernel::MatrixMult(hardware, m_color, m_bmpInfo.biWidth, m_bmpInfo.biHeight,
                                     other.m_color, other.m_bmpInfo.biWidth, other.m_bmpInfo.biHeight,
                                     m_performance.performanceArray);

    DeleteColorMap();
    SetBitmapInformation(m_bmpInfo.biHeight, other.m_bmpInfo.biWidth);
    m_color = resultant;
}


/**
 * File write
 */
std::ofstream Bitmap::CreateFile(std::string filePath,
                                 const std::string& fileName)
{
    std::ofstream file(filePath, std::ofstream::out | std::ofstream::binary);
    if (file.fail())
    {
        std::string path = filePath.erase(filePath.find(fileName + ".bmp"),
                                          fileName.length() + 4);
        mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
        file.open(filePath, std::ofstream::out | std::ofstream::binary);
    }
    return file;
}

void Bitmap::WriteData(std::ofstream& file)
{
    if (!m_color)
        return;

    // Write to opened file
    file.write((char *)(&m_bmpHead), 14);
    file.write((char *)(&m_bmpInfo), 40);

    // Reader
    const RGBQUAD* reader = nullptr;

    for (int row = 0; row < m_bmpInfo.biHeight; ++row)//{
        for (int col = 0; col < m_bmpInfo.biWidth; ++col)
        {
            reader = m_color[col] + row; //printf("[%d %d %d] ", reader->rgbBlue, reader->rgbGreen, reader->rgbRed);
            file << reader->rgbBlue;    // Blue channel
            file << reader->rgbGreen;   // Green channel
            file << reader->rgbRed;     // Red channel
        }//printf("\n");}
}

void Bitmap::WriteData(std::ofstream& file, double performance[])
{
    if (!m_color)
        return;

    // Timer --> easier to type
    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::duration<double> double_seconds;
    // Time initialization for file build
    auto bitmap_file_build_start = Time::now();

    // Write to opened file
    file.write((char *)(&m_bmpHead), 14);
    file.write((char *)(&m_bmpInfo), 40);

    // Reader
    const RGBQUAD* reader = nullptr;

    for (int row = 0; row < m_bmpInfo.biHeight; ++row)//{
        for (int col = 0; col < m_bmpInfo.biWidth; ++col)
        {
            reader = m_color[col] + row; //printf("[%d %d %d] ", reader->rgbBlue, reader->rgbGreen, reader->rgbRed);
            file << reader->rgbBlue;    // Blue channel
            file << reader->rgbGreen;   // Green channel
            file << reader->rgbRed;     // Red channel
        }//printf("\n");}

    // Finish timing
    auto bitmap_file_build_finish = Time::now();
    double_seconds seconds = bitmap_file_build_finish - bitmap_file_build_start;
    performance[6] = seconds.count() * 1000; // BitmapFileBuild
}



/**
 * Data retrieval
 */
BITMAPFILEHEADER Bitmap::GetFileHeader() noexcept { return m_bmpHead; }
BITMAPINFOHEADER Bitmap::GetInfoHeader() noexcept { return m_bmpInfo; }


/**
 * Data setup and removal
 */
inline void Bitmap::SetBitmapInformation(int width, int height) noexcept
{
    // Size of image information
    int size = width * height * 3;

    // Bitmap Header
    m_bmpHead.bfType = 0x4d42;
    m_bmpHead.bfSize = size + sizeof(BITMAPFILEHEADER)
                       + sizeof(BITMAPINFOHEADER);
    m_bmpHead.bfReserved1 = 0;
    m_bmpHead.bfReserved2 = 0;
    m_bmpHead.bfOffBits = m_bmpHead.bfSize - size;

    // Bitmap Information
    m_bmpInfo.biSize = 40;
    m_bmpInfo.biWidth = width;
    m_bmpInfo.biHeight = height;
    m_bmpInfo.biPlanes = 1;
    m_bmpInfo.biBitCount = 24;
    m_bmpInfo.biCompress = 0;
    m_bmpInfo.biSizeImage = size / 3;
    m_bmpInfo.biXPelsPerMeter = 0;
    m_bmpInfo.biYPelsPerMeter = 0;
    m_bmpInfo.biClrUsed = 0;
    m_bmpInfo.biClrImportant = 0;
}

inline void Bitmap::SetPerformanceInformation() noexcept
{
    const double init = (!m_track_performance) ? 0.0 : -1.0;

    m_performance.gpuKernelInit     = \
    m_performance.gpuKernelLaunch   = \
    m_performance.gpuMapCopy        = \
    m_performance.cpuMapInit        = \
    m_performance.cpuTextureInit    = \
    m_performance.bitmapMapBuild    = \
    m_performance.bitmapFileBuild   = init;
}

RGBQUAD** Bitmap::AllocateColorMap(int width, int height)
{
    // Allocates the width of the array
    RGBQUAD** map = new RGBQUAD*[width];

    // Allocation of columns
    for (int i = 0; i < width; ++i)
    {
        map[i] = new RGBQUAD[height];
    }

    return map;
}

RGBQUAD** Bitmap::CopyColorMap(RGBQUAD** map, int width, int height)
{
    // Allocates a new RGBQUAD matrix
    RGBQUAD** newMap = AllocateColorMap(width, height);

    for (int j = 0; j < height; ++j)
        for (int i = 0; i < width; ++i)
            newMap[i][j] = map[i][j];

    return newMap;
}

Bitmap Bitmap::Clone()
{
    Bitmap clone(m_bmpInfo.biWidth, m_bmpInfo.biHeight, m_track_performance);
    clone.DeleteColorMap();
    clone.m_color = CopyColorMap(m_color, m_bmpInfo.biWidth, m_bmpInfo.biHeight);
    return clone;
}

void Bitmap::DeleteColorMap()
{
    if (!m_color)
        return;

    // Deallocate the columns
    for (int i = 0; i < m_bmpInfo.biWidth; ++i)
    {
        delete[] m_color[i];
    }
    // Deallocate the width
    delete[] m_color;
    m_color = nullptr;
}
