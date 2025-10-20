#ifndef DPFLOW_H
#define DPFLOW_H

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include "slic.h"

extern const char* kernelSource_dpflow;

class DPflow {
public:
    DPflow(cl_context context_, cl_command_queue queue_, cl_device_id device_, int window_size, float lambda_reg, bool enableSuperpixels, float compactness, int numSuperpixels, bool green, int grid_spacing = 64, int square_size = 40, int disparity_range = 6);
    ~DPflow();
    void allocateBuffers(int width, int height, int channels, int topMargin=0, int bottomMargin=0);
    void setKernelArgs(const cl_mem clInputL, const cl_mem clInputR, const cl_mem clInputTotalImage, const cl_mem clMask, const cl_mem clFinalOutput, int flag);
    void releaseBuffers();
    void compute_optical_flow_x();

private:
    void buildKernels();
    cl_context context;
    cl_command_queue queue;
    cl_device_id device;
    cl_program program = nullptr;
    cl_kernel flow_kernel = nullptr;
    cl_kernel compute_flow_kernel = nullptr;
    cl_kernel numerator_integral_row_kernel = nullptr;
    cl_kernel numerator_integral_col_kernel = nullptr;
    cl_kernel box_filter_numerator_integral_kernel = nullptr;
    cl_kernel denominator_integral_row_kernel = nullptr;
    cl_kernel denominator_integral_col_kernel = nullptr;
    cl_kernel box_filter_denominator_integral_kernel = nullptr;
    cl_mem clNumeratorOutput = nullptr;
    cl_mem clDenominatorOutput = nullptr;
    cl_mem clNumeratorFinalOutput = nullptr;
    cl_mem clDenominatorFinalOutput = nullptr;
    cl_mem clNumeratorIntegralBuffer1 = nullptr;
    cl_mem clNumeratorIntegralBuffer2 = nullptr;
    cl_mem clDenominatorIntegralBuffer1 = nullptr;
    cl_mem clDenominatorIntegralBuffer2 = nullptr;
    cl_mem clLabels = nullptr;
    int width = 0, height = 0, channels = 0;
    int topMargin = 0, bottomMargin = 0;
    size_t global_size[2] = {0, 0};
    size_t row_global[2] = {0, 0};
    size_t col_global[2] = {0, 0};
    int window_size; 
    float lambda_reg;
    SLICSuperpixels* slic = nullptr;
    bool enableSuperpixels;
    float compactness;
    int numSuperpixels;
    bool green;
    int grid_spacing;
    int square_size;
    int disparity_range;
};

#endif // DPFLOW_H