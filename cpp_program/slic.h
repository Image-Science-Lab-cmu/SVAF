#ifndef SLIC_H
#define SLIC_H

#include <CL/cl.h>
#include <opencv2/opencv.hpp>
// #include <opencv2/ximgproc.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <stdexcept>
#include <random>

// OpenCL kernel source code
extern const char* kernelSource1;
extern const char* kernelSource2;
extern const char* kernelSource3;
extern const char* kernelSource4;
extern const char* kernelSource5;
extern const char* kernelSource6;
extern const char* kernelSource7;
extern const char* kernelSource8;
extern const char* kernelSource9;

class SLICSuperpixels {
private:
    // Member variables
    int width;
    int height;
    int numSuperpixels;
    int numPixels;
    int regionSize;
    size_t globalSize;
    size_t globalSize2[2];
    size_t globalSize3;
    float compactness;
    int iterations;
    bool processTwoMaps;

    // OpenCL variables
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel initKernel, assignKernel, updateKernel, accumulateKernel, finalizeKernel;
    // Kernels for single map case
    cl_kernel averageKernel;
    cl_kernel computeKernel;
    cl_kernel assignAvgKernel;
    // Kernels for two-map case
    cl_kernel averageKernel2;
    cl_kernel computeKernel2;
    cl_kernel assignAvgKernel2;

    // Device buffers
    cl_mem d_clusters = nullptr;
    cl_mem d_accum = nullptr;
    cl_mem d_counts = nullptr;
    // Buffers for single map case
    cl_mem d_pixel_counts_per_superpixel = nullptr;
    cl_mem d_value_sum_per_superpixel = nullptr;
    cl_mem d_averages = nullptr;
    // Buffers for two-map case
    cl_mem d_value_sum_per_superpixel1 = nullptr;
    cl_mem d_value_sum_per_superpixel2 = nullptr;
    cl_mem d_pixel_counts_per_superpixel1 = nullptr;
    cl_mem d_pixel_counts_per_superpixel2 = nullptr;
    cl_mem d_averages1 = nullptr;
    cl_mem d_averages2 = nullptr;
    cl_device_id device;

    // Helper function to load OpenCL kernel from string
    std::string loadKernelSource();
    
    // Initialize OpenCL
    void initOpenCL();
    
    // Clean up OpenCL resources
    void CleanupOpenCL();
    
public:
    SLICSuperpixels(cl_context context, cl_command_queue queue, cl_device_id device, float compactness = 13.0f, int numSuperpixels = 1000, int iterations = 1);
    
    ~SLICSuperpixels();

    // Allocate all needed buffers except d_image
    void allocateBuffers(int width_, int height_, bool processTwoMaps_);

    // Set all kernel arguments, using external d_image
    void setKernelArgs(cl_mem d_image, cl_mem d_labels, cl_mem d_opticalFlowMap1, cl_mem d_opticalFlowMap2, cl_mem d_avgImage1, cl_mem d_avgImage2);

    // Release device buffers
    void releaseBuffers();

    // Run kernels and process image, including superpixel averaging and assignment
    void processImage();

    void processTwoImages();
};

#endif // SLIC_H 