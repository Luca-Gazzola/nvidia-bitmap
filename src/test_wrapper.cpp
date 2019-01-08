// System includes
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <functional>
// CUDA includes
#include <cuda.h>
#include <cuda_runtime.h>
// File includes
#include "../include/Bitmap.h"
#include "../include/BitmapException.h"
#include "../include/Wrapper.h"



// Strings for Bitmap_Type and Processor_Type that reflect enums
std::string bitmapGen[] = {"Static", "Random", "Add", "Subtract", "Multiply", "Divide"};
std::string procType[]  = {"GPU", "CPU", "CPUtoGPU"};
// Create date/time
auto datetime = std::time(nullptr);
auto currentTime = *std::localtime(&datetime);

void test_ExpectPass(const std::string& testName, unsigned int& testNumber, std::function<void()>& lambda)
{
    std::cout << "\033[0;33m[" << testName << " " << testNumber++ << "]\033[0m" << std::endl;

    try
    {
        lambda();
    }
    catch (const std::runtime_error& e)
    {
        std::cout << " [EXCEPTION CAUGHT] " << e.what() << std::endl;
        std::cout << "\033[0;31m" << "\tFAIL" << "\033[0m" << std::endl;
        throw e;
    }
    catch (const std::exception& e)
    {
        std::cout << " [UNEXPECTED EXCEPTION CAUGHT] " << e.what() << std::endl;
        std::cout << "\033[0;31m" << "\tFAIL" << "\033[0m" << std::endl;
        throw e;
    }
    catch (...)
    {
        std::cout << " [UNKNOWN EXCEPTION CAUGHT]" << std::endl;
        std::cout << "\033[0;31m" << "\tFAIL" << "\033[0m" << std::endl;
        throw std::runtime_error("Unknown Exception");
    }

    std::cout << "\033[0;32m" << "\tPASS" << "\033[0m" << std::endl;
}

void test_ExpectFail(const std::string& testName, unsigned int& testNumber, std::function<void()>& lambda)
{
    std::cout << "\033[0;33m[" << testName << " " << testNumber++ << "]\033[0m";

    try
    {
        lambda();
    }
    catch (const std::runtime_error& e)
    {
        std::cout << " [EXCEPTION CAUGHT] " << e.what() << std::endl;
        std::cout << "\033[0;32m" << "\tPASS" << "\033[0m" << std::endl;
        return;
    }
    catch (const std::exception& e)
    {
        std::cout << " [UNEXPECTED EXCEPTION CAUGHT] " << e.what() << std::endl;
        std::cout << "\033[0;31m" << "\tFAIL" << "\033[0m" << std::endl;
        throw e;
    }
    catch (...)
    {
        std::cout << " [UNKNOWN EXCEPTION CAUGHT]" << std::endl;
        std::cout << "\033[0;31m" << "\tFAIL" << "\033[0m" << std::endl;
        throw std::runtime_error("Unknown Exception");
    }

    std::cout << "\033[0;31m" << "\tFAIL" << "\033[0m" << std::endl;
    throw std::runtime_error("[EXCEPTION] Exception has not been thrown!");
}

void test_LogPerformance(Bitmap& bitmap)
{
    std::cout << std::setprecision(2) << std::fixed;
    std::cout << "GPU Kernel Init:   \t" << bitmap.Performance(BitmapPerfGPUKernelInit)     << " ms" << std::endl;
    std::cout << "GPU Kernel Launch: \t" << bitmap.Performance(BitmapPerfGPUKernelLaunch)   << " ms" << std::endl;
    std::cout << "GPU Map Copy:      \t" << bitmap.Performance(BitmapPerfGPUMapCopy)        << " ms" << std::endl;
    std::cout << "CPU Map Init:      \t" << bitmap.Performance(BitmapPerfCPUMapInit)        << " ms" << std::endl;
    std::cout << "CPU Texture Init:  \t" << bitmap.Performance(BitmapPerfCPUTextureInit)    << " ms" << std::endl;
    std::cout << "Bitmap Map Build:  \t" << bitmap.Performance(BitmapPerfMapBuild)          << " ms" << std::endl;
    std::cout << "Bitmap File Build: \t" << bitmap.Performance(BitmapPerfFileBuild)         << " ms" << std::endl;
}

void test_LogPerformanceToFileHeader(std::ofstream& file, Bitmap_Type gen, Processor_Type proc)
{
    // Outlining info
    file << std::put_time(&currentTime, "%d-%m-%Y %H:%M:%S") << "," << bitmapGen[(int)gen] << ',';
    file << procType[(int)proc] << std::endl;
    // Headers
    file << "GPU_Kernel_Init"  << ',' << "GPU_Kernel_Launch" << ',' << "GPU_Map_Copy" << ',';
    file << "CPU_Map_Init"     << ',' << "CPU_Texture_Init"  << ',';
    file << "Bitmap_Map_Build" << ',' << "Bitmap_File_Build" << std::endl;
}

void test_LogPerformanceToFile(Bitmap& bitmap, std::ofstream& file)
{
    // Data
    file << bitmap.Performance(BitmapPerfGPUKernelInit)   << ',';
    file << bitmap.Performance(BitmapPerfGPUKernelLaunch) << ',';
    file << bitmap.Performance(BitmapPerfGPUMapCopy)      << ',';
    file << bitmap.Performance(BitmapPerfCPUMapInit)      << ',';
    file << bitmap.Performance(BitmapPerfCPUTextureInit)  << ',';
    file << bitmap.Performance(BitmapPerfMapBuild)        << ',';
    file << bitmap.Performance(BitmapPerfFileBuild)       << std::endl;
}



int main()
{
    //<editor-fold desc="Constructor Tests">

    // Setup timeString
    std::ostringstream oss;
    oss << std::put_time(&currentTime, "%d-%m-%Y %H.%M.%S");
    auto timeString = oss.str();

    // Initialize environment, set device
    std::function<void()> test_lambda;
    unsigned int test_construct_number = 1;

    // Basic construction tests
    test_lambda = [](){ Bitmap test; };
    test_ExpectPass("Construction Test", test_construct_number, test_lambda);

    test_lambda = [](){ Bitmap test(true); };
    test_ExpectPass("Construction Test", test_construct_number, test_lambda);

    test_lambda = [](){ Bitmap test(false); };
    test_ExpectPass("Construction Test", test_construct_number, test_lambda);

    test_lambda = [](){ Bitmap test(5,5); };
    test_ExpectFail("Construction Test", test_construct_number, test_lambda);

    test_lambda = [](){ Bitmap test(1,4); };
    test_ExpectFail("Construction Test", test_construct_number, test_lambda);

    test_lambda = [](){ Bitmap test(4,1); };
    test_ExpectFail("Construction Test", test_construct_number, test_lambda);

    test_lambda = [](){ Bitmap test(0,0); };
    test_ExpectFail("Construction Test", test_construct_number, test_lambda);

    test_lambda = [](){ Bitmap test(5,5,true); };
    test_ExpectFail("Construction Test", test_construct_number, test_lambda);

    test_lambda = [](){ Bitmap test(-4,4); };
    test_ExpectFail("Construction Test", test_construct_number, test_lambda);

    test_lambda = [](){ Bitmap test(4,-4); };
    test_ExpectFail("Construction Test", test_construct_number, test_lambda);

    test_lambda = [](){ Bitmap test(-4,-4,true); };
    test_ExpectFail("Construction Test", test_construct_number, test_lambda);

    test_lambda = [](){ Bitmap test(4); };
    test_ExpectPass("Construction Test", test_construct_number, test_lambda);

    test_lambda = [](){ Bitmap test(1028); };
    test_ExpectPass("Construction Test", test_construct_number, test_lambda);

    test_lambda = [](){ Bitmap test(1028, true); };
    test_ExpectPass("Construction Test", test_construct_number, test_lambda);

    test_lambda = [](){ Bitmap test(1024, 1024, true); };
    test_ExpectPass("Construction Test", test_construct_number, test_lambda);

    test_lambda = [](){ Bitmap test(400, 800, false); };
    test_ExpectPass("Construction Test", test_construct_number, test_lambda);

    std::cout << std::endl;

    //</editor-fold>

    /**
     * Test generations
     */
    bool run_prolonged = true;
    bool have_perf = true;
    auto bitmap_gen_type = BitmapTypeRandom;
    auto bitmap_hardware = BitmapProcessorCPUtoGPU;
    int test_iterations = 50;
    std::string location;

    Bitmap test_write(have_perf);
    std::cout << "Creating Bitmap: " << test_write.Width() << "x" << test_write.Height() << std::endl;
    location = test_write.MakeBitmap(bitmap_gen_type, "test_write", bitmap_hardware);
    test_LogPerformance(test_write);
    std::cout << "Saved at " << location << std::endl << std::endl;

    Bitmap test_write2(1920, 1080, have_perf);
    std::cout << "Creating Bitmap: " << test_write2.Width() << "x" << test_write2.Height() << std::endl;
    location = test_write2.MakeBitmap(bitmap_gen_type, "test_write2", bitmap_hardware);
    test_LogPerformance(test_write2);
    std::cout << "Saved at " << location << std::endl << std::endl;

    //<editor-fold desc="Prolonged Tests">
    if (run_prolonged)
    {
        std::string fileName(timeString + "__" + bitmapGen[(int)bitmap_gen_type] + "_" +
                             procType[(int)bitmap_hardware] + "_default.csv");
        std::string fileName2(timeString + "__" + bitmapGen[(int)bitmap_gen_type] + "_" +
                              procType[(int)bitmap_hardware] + "_1080p.csv");
        std::string filePath("./testlogs/" + fileName);
        std::string filePath2("./testlogs/" + fileName2);
        std::ofstream file(filePath, std::ofstream::out);
        std::ofstream file2(filePath2, std::ofstream::out);

        test_LogPerformanceToFileHeader(file, bitmap_gen_type, bitmap_hardware);
        test_LogPerformanceToFileHeader(file2, bitmap_gen_type, bitmap_hardware);

        std::cout << "=RUNNING PROLONGED TEST=" << std::endl;
        for (int i = 0; i < test_iterations; i++)
        {
            std::cout << "Iteration:\t" << i << std::endl;

            Bitmap test_write(have_perf);
            test_write.MakeBitmap(bitmap_gen_type, "test_write", bitmap_hardware);
            test_LogPerformanceToFile(test_write, file);

            Bitmap test_write2(1920, 1080, have_perf);
            test_write2.MakeBitmap(bitmap_gen_type, "test_write2", bitmap_hardware);
            test_LogPerformanceToFile(test_write2, file2);
        }
    }
    //</editor-fold>

    return 0;
};