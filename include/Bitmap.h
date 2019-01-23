#ifndef BITMAP_H
#define BITMAP_H

// System includes
#include <string>
// File includes
#include "../resource/bitmap_resources.h"



class Bitmap
{
public:
    /**
     * Constructors
     */
     // Parameterized Constructors
    explicit Bitmap(bool track_performance = false);
    explicit Bitmap(int side, bool track_performance = false);
    explicit Bitmap(int width, int height, bool track_performance = false);
    // Copy Constructor
    Bitmap(const Bitmap& other);

    /**
     * Destructor
     */
    ~Bitmap();

    /**
     * Bitmap Generators - Methods
     */
    const std::string MakeBitmap(std::string fileName);
    const std::string MakeBitmap(Bitmap_Type gen, std::string fileName,
                                 Processor_Type hardware = Processor_Type::GPU);
    const std::string MakeBitmap(Bitmap_Type gen, std::string fileName, const Bitmap& other,
                                 Processor_Type hardware = Processor_Type::GPU);

    /**
     * Public Methods
     */
    const int Width() noexcept;
    const int Height() noexcept;
    const double& Performance(Bitmap_Performance::GPU timeType) noexcept;
    const double& Performance(Bitmap_Performance::CPU timeType) noexcept;
    const double& Performance(Bitmap_Performance::Bitmap timeType) noexcept;
    const double* Performance() noexcept;

private:
    /**
    * Data Members
    */
    // Map
    RGBQUAD** m_color;
    // Bitmap Information
    BITMAPFILEHEADER m_bmpHead;
    BITMAPINFOHEADER m_bmpInfo;
    // CUDA Setup Flag
    static bool s_cuda_setup;
    // Performance info
    bool m_track_performance;
    Bitmap_Performance m_performance;

    /**
     * Member functions
     */
     // Map generators
    void GenerateStatic(Processor_Type hardware);
    void GenerateRandom(Processor_Type hardware);
    void GenerateAdd(Processor_Type hardware, const Bitmap& other);
    void GenerateSubtract();
    void GenerateMultiply();
    void GenerateDivide();
    void GenerateMatrixMult(Processor_Type hardware, const Bitmap& other);

    // File write
    void WriteData(std::ofstream& file);
    void WriteData(std::ofstream& file, double performance[]);

    // Data retrieval
    BITMAPFILEHEADER GetFileHeader() noexcept;
    BITMAPINFOHEADER GetInfoHeader() noexcept;

    // Data setup and removal
    inline void SetBitmapInformation(int width, int height) noexcept;
    void SetPerformanceInformation() noexcept;
    RGBQUAD** AllocateColorMap(int width, int height);
    RGBQUAD** CopyColorMap(RGBQUAD** map, int width, int height);
    void DeleteColorMap();
};

#endif //BITMAP_H
