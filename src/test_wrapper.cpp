// System includes
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <functional>
#include <string>
#include <vector>
#include <stdlib.h>
// CUDA includes
#include <cuda.h>
#include <cuda_runtime.h>
// File includes
#include "../include/Bitmap.h"
#include "../include/BitmapException.h"
#include "../include/Wrapper.h"



// Strings for Bitmap_Type and Processor_Type that reflect enums
std::string bitmapGen[] = {"Static", "Random", "Add", "Subtract", "Multiply", "Divide", "MatrixMult"};
std::string procType[]  = {"GPU", "CPU", "CPUtoGPU"};
// Create date/time
auto datetime = std::time(nullptr);
auto currentTime = *std::localtime(&datetime);
// Create command struct
struct Command_Args
{
    bool performance_track      = false;
    Bitmap_Type bmp_gen         = BitmapTypeStatic;
    Processor_Type bmp_hardware = BitmapProcessorCPU;
    bool long_test_flag         = false;
    int  long_test_iter         = 0;
};

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
    std::cout << std::endl;
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

void test_InterpretFlag(Command_Args& comm, char* arg)
{
    std::string temp = arg;
    // Performance flag
    if (temp == std::string("performance_track=true"))
    { comm.performance_track = true; return; }
    else if (temp == std::string("performance_track=false"))
    { comm.performance_track = false; return; }
        // Bitmap Generation Type
    else if (temp == std::string("bmp_gen=BitmapTypeStatic"))
    { comm.bmp_gen = BitmapTypeStatic; return; }
    else if (temp == std::string("bmp_gen=BitmapTypeRandom"))
    { comm.bmp_gen = BitmapTypeRandom; return; }
    else if (temp == std::string("bmp_gen=BitmapTypeMatrixMult"))
    { comm.bmp_gen = BitmapTypeMatrixMult; return; }
    else if (temp == std::string("bmp_gen=BitmapTypeAdd"))
    { comm.bmp_gen = BitmapTypeAdd; return; }
    else if (temp == std::string("bmp_gen=BitmapTypeSubtract"))
        throw std::runtime_error(temp + std::string(" is not implemented"));
    else if (temp == std::string("bmp_gen=BitmapTypeMultiply"))
        throw std::runtime_error(temp + std::string(" is not implemented"));
    else if (temp == std::string("bmp_gen=BitmapTypeDivide"))
        throw std::runtime_error(temp + std::string(" is not implemented"));
        // Bitmap Processor Type
    else if (temp == std::string("bmp_hardware=BitmapProcessorGPU"))
    { comm.bmp_hardware = BitmapProcessorGPU; return; }
    else if (temp == std::string("bmp_hardware=BitmapProcessorCPU"))
    { comm.bmp_hardware = BitmapProcessorCPU; return; }
    else if (temp == std::string("bmp_hardware=BitmapProcessorCPUtoGPU"))
    { comm.bmp_hardware = BitmapProcessorCPUtoGPU; return; }
        // Long Test bool
    else if (temp == std::string("long_test_flag=true"))
    { comm.long_test_flag = true; return; }
    else if (temp == std::string("long_test_flag=false"))
    { comm.long_test_flag = false; return; }
    // Long Test Iterations
    int temp2 = std::stoi(temp);
    if (temp2 != 0)
    { comm.long_test_iter = temp2;  return; }

    throw std::runtime_error(std::string("Unknown Flag: ") + temp);
}

Command_Args test_InterpretCommands(int argc, char** argv)
{
    /**
     * struct Command_Args
     * {
     * bool performance_track = false;
     * auto bmp_gen           = BitmapTypeStatic;
     * auto bmp_hardware      = BitmapProcessorCPU;
     * bool long_test_flag    = false;
     * int  long_test_iter    = 0;
     * };
     */
     Command_Args comm;

    if (argc <= 1)
    {
        std::cout << "=Using default variables (count: " << argc << ")=" << std::endl;
        std::cout << " >Performance Tracking  : " << (comm.performance_track ? "True" : "False") << std::endl;
        std::cout << " >Bitmap Generate Type  : " << bitmapGen[static_cast<int>(comm.bmp_gen)] << std::endl;
        std::cout << " >Bitmap Processor Type : " << procType[static_cast<int>(comm.bmp_hardware)] << std::endl;
        std::cout << " >Long Test Execution   : " << (comm.long_test_flag ? "True" : "False") << std::endl;
        std::cout << " >Long Test Iterations  : " << comm.long_test_iter << std::endl << std::endl;
        return comm;
    }
    else if (argc > 6)
    {
        std::cout << "Too many arguments" << std::endl;
        throw std::runtime_error("Too many arguments");
    }

    for (int i = 1; i < argc; i++)
        test_InterpretFlag(comm, argv[i]);

    std::cout << "=Using specified variables (count: " << argc << ")=" << std::endl;
    std::cout << " >Performance Tracking  : " << (comm.performance_track ? "True" : "False") << std::endl;
    std::cout << " >Bitmap Generate Type  : " << bitmapGen[static_cast<int>(comm.bmp_gen)] << std::endl;
    std::cout << " >Bitmap Processor Type : " << procType[static_cast<int>(comm.bmp_hardware)] << std::endl;
    std::cout << " >Long Test Execution   : " << (comm.long_test_flag ? "True" : "False") << std::endl;
    std::cout << " >Long Test Iterations  : " << comm.long_test_iter << std::endl << std::endl;

    return comm;
}



int main(int argc, char** argv)
{
    //test_ModifyArgs(argc, argv, ' '); // Use because Makefile has args as one large string

    Command_Args commands = test_InterpretCommands(argc, argv);

    bool have_perf                 = commands.performance_track;
    Bitmap_Type bitmap_gen_type    = commands.bmp_gen;
    Processor_Type bitmap_hardware = commands.bmp_hardware;
    bool run_prolonged             = commands.long_test_flag;
    int test_iterations            = commands.long_test_iter;


    //<editor-fold desc="Constructor Tests">

    // Setup timeString
    std::ostringstream oss;
    oss << std::put_time(&currentTime, "%d-%m-%Y %H.%M.%S");
    auto timeString = oss.str();
/*
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
    */

    /**
     * Test generations
     */
    std::string location;


    Bitmap first(1920, 1080, have_perf);
    Bitmap second(1920, 1080, have_perf);
    first.MakeBitmap(bitmap_gen_type, "first");
    test_LogPerformance(first);
    second.MakeBitmap(bitmap_gen_type, "second");
    test_LogPerformance(second);
    std::cout << "Creating Bitmap: " << first.Width() << "x" << first.Height() << std::endl;
    location = first.MakeBitmap(BitmapTypeSubtract, "matrixSub", second, bitmap_hardware);
    test_LogPerformance(first);
    std::cout << "Saved at " << location << std::endl << std::endl;

/*
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
*/

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

        datetime = std::time(nullptr);
        currentTime = *std::localtime(&datetime);
        std::ostringstream timeStart;
        timeStart << std::put_time(&currentTime, "%d-%m-%Y %H:%M:%S");
        auto timeStartString = timeStart.str();

        std::cout << "========================" << std::endl;
        std::cout << "=RUNNING PROLONGED TEST=" << std::endl;
        std::cout << "========================" << std::endl;
        if (commands.bmp_gen == BitmapTypeMatrixMult)
        {
            Bitmap_Type secondary_gen = BitmapTypeStatic;
            for (int i = 0; i < test_iterations; i++)
            {
                std::cout << "Iteration:\t" << i << std::endl;

                Bitmap test_writeLeft(have_perf);
                Bitmap test_writeRight(have_perf);
                test_writeLeft.MakeBitmap(secondary_gen, "test_matrixMultStatic", bitmap_hardware);
                test_writeRight.MakeBitmap(secondary_gen, "test_matrixMultStatic", bitmap_hardware);
                test_writeLeft.MakeBitmap(bitmap_gen_type, "test_matrixMultStatic",
                                          test_writeRight, bitmap_hardware);
                test_LogPerformanceToFile(test_writeLeft, file);
                test_writeLeft.~Bitmap();
                test_writeRight.~Bitmap();

                Bitmap test_write2Left(1920, 1080, have_perf);
                Bitmap test_write2Right(1080, 1920, have_perf);
                test_write2Left.MakeBitmap(secondary_gen, "test_matrixMultStaticHD", bitmap_hardware);
                test_write2Right.MakeBitmap(secondary_gen, "test_matrixMultStaticHD", bitmap_hardware);
                test_write2Left.MakeBitmap(bitmap_gen_type, "test_matrixMultStaticHD",
                                           test_write2Right, bitmap_hardware);
                test_LogPerformanceToFile(test_write2Left, file2);
                test_write2Left.~Bitmap();
                test_write2Right.~Bitmap();
            }
        }
        else if (commands.bmp_gen == BitmapTypeAdd)
        {
            Bitmap_Type secondary_gen = BitmapTypeStatic;
            for (int i = 0; i < test_iterations; i++)
            {
                std::cout << "Iteration:\t" << i << std::endl;

                Bitmap test_writeLeft(have_perf);
                Bitmap test_writeRight(have_perf);
                test_writeLeft.MakeBitmap(secondary_gen, "test_addRandom", bitmap_hardware);
                test_writeRight.MakeBitmap(secondary_gen, "test_addRandom", bitmap_hardware);
                test_writeLeft.MakeBitmap(bitmap_gen_type, "test_addRandom",
                                          test_writeRight, bitmap_hardware);
                test_LogPerformanceToFile(test_writeLeft, file);

                Bitmap test_write2Left(1920, 1080, have_perf);
                Bitmap test_write2Right(1920, 1080, have_perf);
                test_write2Left.MakeBitmap(secondary_gen, "test_addRandomHD", bitmap_hardware);
                test_write2Right.MakeBitmap(secondary_gen, "test_addRandomHD", bitmap_hardware);
                test_write2Left.MakeBitmap(bitmap_gen_type, "test_addRandomHD",
                                           test_write2Right, bitmap_hardware);
                test_LogPerformanceToFile(test_write2Left, file2);
            }
        }
        else
        {
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

        std::cout << std::endl;

        datetime = std::time(nullptr);
        currentTime = *std::localtime(&datetime);
        std::cout << "Test started at " << timeStartString << std::endl;
        std::cout << "Test ended at   " << std::put_time(&currentTime, "%d-%m-%Y %H:%M:%S") << std::endl;

        std::cout << "=======================" << std::endl;
        std::cout << "=END OF PROLONGED TEST=" << std::endl;
        std::cout << "=======================" << std::endl;
    }
    //</editor-fold>

    return 0;
};