#ifndef BITMAP_RESOURCES_H
#define BITMAP_RESOURCES_H

/**
 * Bitmap Structures
 *
 * ======= Data Types =======
 * unsigned char    = 1 bit
 * unsigned short   = 2 bits
 * unsigned int     = 4 bits
 * int              = 4 bits
 */
#pragma pack(1)
struct BITMAPFILEHEADER
{
    unsigned short bfType;          // Always 'BM' (0x4d42)
    int bfSize;                     // File size in bytes
    unsigned short bfReserved1;     // Always 0
    unsigned short bfReserved2;     // Always 0
    int bfOffBits;                  // Offset from the beginning of file
};
#pragma pack()

struct BITMAPINFOHEADER
{
    int biSize;                     // Size of this structure in bytes (40)
    int biWidth;                    // Width of image in pixels
    int biHeight;                   // Height of image in pixels
    unsigned short biPlanes;        // The number of planes of device (1)
    unsigned short biBitCount;      // Number of bits per pixel (24 for color)
    int biCompress;                 // Compression (usually 0)
    int biSizeImage;                // Height * Width * (3 color channels)
    int biXPelsPerMeter;            // Horizontal pixels per meter (0)
    int biYPelsPerMeter;            // Vertical pixels per meter (0)
    int biClrUsed;                  // Number of colors used (0)
    int biClrImportant;             // Number of colors important (0)
};

struct RGBQUAD
{
    unsigned char rgbBlue;          // Blue part of the color
    unsigned char rgbGreen;         // Green part of the color
    unsigned char rgbRed;           // Red part of the color
    unsigned char rgbReserved;      // Always 0
};



/**
 * Additional structures
 */
enum class Processor_Type : int
{
    GPU,        // GPU only for generation and modification
    CPU,        // CPU only for generation and modification
    CPUtoGPU    // CPU for generation, GPU for modification
};

enum class Bitmap_Type : int
{
    Static,
    Random,
    Add,
    Subtract,
    Multiply,
    Divide,
    MatrixMult
};

struct Bitmap_Performance
{
    // ** Maybe separate time depending on category (GPU, CPU, Bitmap Class) ** //
    enum class GPU : int
    {
        KernelInit      = 0,
        KernelLaunch    = 1,
        MapCopy         = 2
    };

    enum class CPU : int
    {
        MapInit         = 3,
        TextureInit     = 4
    };

    enum class Bitmap : int
    {
        MapBuild        = 5,
        FileBuild       = 6
    };

    // ** Don't forget to add the shit here (from the enum) ** //
    // ** Oh yeah, and the macros too, those are nice ** //
    double performanceArray[7];

    double& gpuKernelInit       = performanceArray[static_cast<int>(GPU::KernelInit)];
    double& gpuKernelLaunch     = performanceArray[static_cast<int>(GPU::KernelLaunch)];
    double& gpuMapCopy          = performanceArray[static_cast<int>(GPU::MapCopy)];
    double& cpuMapInit          = performanceArray[static_cast<int>(CPU::MapInit)];
    double& cpuTextureInit      = performanceArray[static_cast<int>(CPU::TextureInit)];
    double& bitmapMapBuild      = performanceArray[static_cast<int>(Bitmap::MapBuild)];
    double& bitmapFileBuild     = performanceArray[static_cast<int>(Bitmap::FileBuild)];
};

/**
 * Preprocessor Macros for easy variable names
 */
// Processor Type
#define BitmapProcessorGPU          Processor_Type::GPU
#define BitmapProcessorCPU          Processor_Type::CPU
#define BitmapProcessorCPUtoGPU     Processor_Type::CPUtoGPU
// Bitmap Type
#define BitmapTypeStatic            Bitmap_Type::Static
#define BitmapTypeRandom            Bitmap_Type::Random
#define BitmapTypeAdd               Bitmap_Type::Add
#define BitmapTypeSubtract          Bitmap_Type::Subtract
#define BitmapTypeMultiply          Bitmap_Type::Multiply
#define BitmapTypeDivide            Bitmap_Type::Divide
#define BitmapTypeMatrixMult        Bitmap_Type::MatrixMult
// Bitmap Performance
#define BitmapPerfGPUKernelInit     Bitmap_Performance::GPU::KernelInit
#define BitmapPerfGPUKernelLaunch   Bitmap_Performance::GPU::KernelLaunch
#define BitmapPerfGPUMapCopy        Bitmap_Performance::GPU::MapCopy
#define BitmapPerfCPUMapInit        Bitmap_Performance::CPU::MapInit
#define BitmapPerfCPUTextureInit    Bitmap_Performance::CPU::TextureInit
#define BitmapPerfMapBuild          Bitmap_Performance::Bitmap::MapBuild
#define BitmapPerfFileBuild         Bitmap_Performance::Bitmap::FileBuild

#endif //BITMAP_RESOURCES_H
