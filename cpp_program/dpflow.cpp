#include "stdafx.h"
#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <stdexcept>
#include <limits>

#include "dpflow.h"

// OpenCL kernel source code
const char* kernelSource_dpflow = R"(
__kernel void compute_optical_flow_x( //_NCC(
    __global const float* img_r,
    __global const float* img_l,
    __global const float* mask,
    __global float* numerator,
    __global float* denominator,
    int width,
    int height,
    int channels,
    int topMargin,
    int bottomMargin,
    float lambda_reg,
    int is_green,
    __global float* output,
    int flag,
    int grid_spacing,
    int square_size,
    int disparity_range)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    int half_grid_spacing = grid_spacing / 2;
    if ((x % grid_spacing != half_grid_spacing) || (y % grid_spacing != half_grid_spacing)) return;

    //if (x < 64) return;
    //if (y < 64) return;

    float sum_num;
    float sum_den_l, sum_den_r;
    float tmp;
    float mean_l;
    float mean_r;
    float ncc;

    float min_val = -1;
    float min_arg = 0;

    for (int itr = 0; itr < 1; itr++) {
    for (int d = -disparity_range; d < disparity_range; d++) {
        ncc = 0;

        for (int c = 0; c < 3; c++) {
            sum_num = 0;
            sum_den_l = 0;
            sum_den_r = 0;
            mean_l = 0;
            mean_r = 0;

            int half_square_size = square_size / 2;
            for (int dx = -half_square_size; dx < half_square_size; dx+=2) {
                for (int dy = -half_square_size; dy < half_square_size; dy+=2) {
                    int idx_l = (y+dy) * width + (x+dx+d);
                    int idx_r = (y+dy) * width + (x+dx);

                    mean_l += img_l[idx_l * channels + c];
                    mean_r += img_r[idx_r * channels + c];
                }
            }

            mean_l = mean_l / (square_size * square_size);
            mean_r = mean_r / (square_size * square_size);

            for (int dx = -half_square_size; dx < half_square_size; dx+=2) {
                for (int dy = -half_square_size; dy < half_square_size; dy+=2) {
                    int idx_l = (y+dy) * width + (x+dx+d);
                    int idx_r = (y+dy) * width + (x+dx);

                    sum_num += (img_l[idx_l * channels + c] - mean_l) * (img_r[idx_r * channels + c] - mean_r);

                    tmp = (img_l[idx_l * channels + c] - mean_l);
                    sum_den_l += tmp * tmp;

                    tmp = (img_r[idx_r * channels + c] - mean_r);
                    sum_den_r += tmp * tmp;
                }
            }

            ncc += sum_num / sqrt(sum_den_l * sum_den_r);
        }

        if (ncc > min_val) {
            min_val = ncc;
            min_arg = d;
        }
    }
    }

    int half_output_size = grid_spacing / 2;
    for (int dx = -half_output_size; dx < half_output_size; dx++) {
        for (int dy = -half_output_size; dy < half_output_size; dy++) {
            int idx = (y+dy) * width + (x+dx);

            if (flag || (x > width/2 - 200 && x < width/2 + 200 && y > height/2 - 200 && y < height/2 + 200))
                output[idx] = min_arg;
        }
    }
}

__kernel void compute_optical_flow_x_SSD(
    __global const float* img_l,
    __global const float* img_r,
    __global const float* mask,
    __global float* numerator,
    __global float* denominator,
    int width,
    int height,
    int channels,
    int topMargin,
    int bottomMargin,
    float lambda_reg,
    int is_green,
    __global float* output)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    if ((x % 64 != 32) || (y % 64 != 32)) return;

    float sum;
    float tmp;

    float min_val = 1e9;
    float min_arg = 0;

    for (int d = -10; d < 10; d++) {
        sum = 0;

        for (int dx = -20; dx < 20; dx++) {
            for (int dy = -20; dy < 20; dy++) {
                int idx_l = (y+dy) * width + (x+dx+d);
                int idx_r = (y+dy) * width + (x+dx);

                for (int c = 0; c < 3; c++) {
                    tmp = img_l[idx_l * channels + c] - img_r[idx_r * channels + c];
                    sum += tmp * tmp;
                }
            }
        }

        if (sum < min_val) {
            min_val = sum;
            min_arg = d;
        }
    }

    for (int dx = -32; dx < 32; dx++) {
        for (int dy = -32; dy < 32; dy++) {
            int idx = (y+dy) * width + (x+dx);

            output[idx] = min_arg;
        }
    }
}

__kernel void compute_optical_flow_x_old(
    __global const float* img_l,
    __global const float* img_r,
    __global const float* mask,
    __global float* numerator,
    __global float* denominator,
    int width,
    int height,
    int channels,
    int topMargin,
    int bottomMargin,
    float lambda_reg,
    int is_green)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    float xdiff = 0.0f;
    float tdiff = 0.0f;
    
    if (is_green) {
        // Compute xdiff for green channel using forward difference
        if (x < width - 1) {
            float diff = img_l[(idx + 1) * channels + 1] - img_l[idx * channels + 1];
            xdiff = diff * mask[idx];
        }
        
        // Compute temporal difference
        tdiff = (img_l[idx * channels + 1] - img_r[idx * channels + 1]) * mask[idx];
    } else {
        // Convert to grayscale
        float gray_l = 0.299f * img_l[idx * channels] + 
                      0.587f * img_l[idx * channels + 1] + 
                      0.114f * img_l[idx * channels + 2];
        float gray_r = 0.299f * img_r[idx * channels] + 
                      0.587f * img_r[idx * channels + 1] + 
                      0.114f * img_r[idx * channels + 2];
        
        // Compute xdiff using forward difference
        if (x < width - 1) {
            float next_gray = 0.299f * img_l[(idx + 1) * channels] + 
                             0.587f * img_l[(idx + 1) * channels + 1] + 
                             0.114f * img_l[(idx + 1) * channels + 2];
            float diff = next_gray - gray_l;
            xdiff = diff * mask[idx];
            if (y < topMargin || y >= height - bottomMargin) {
                xdiff = xdiff * 0.0f;
            }
        }
        
        // Compute temporal difference
        tdiff = (gray_l - gray_r) * mask[idx];
        if (y < topMargin || y >= height - bottomMargin) {
            tdiff = tdiff * 0.0f;
        }
    }
    
    // Store intermediate results
    numerator[idx] = -xdiff * tdiff;
    denominator[idx] = xdiff * xdiff;
}

// Helper function for BORDER_REFLECT_101 (cv2.BORDER_DEFAULT)
inline int reflect101(int p, int len) {
    if (len == 1) return 0;
    while (p < 0 || p >= len) {
        if (p < 0) p = -p;
        if (p >= len) p = 2 * len - p - 1;
    }
    return p;
}

__kernel void box_filter(
    __global const float* input,
    __global float* output,
    int width,
    int height,
    int window_size)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    int half_size = window_size / 2;
    float sum = 0.0f;
    
    for (int dy = -half_size; dy <= half_size; dy++) {
        for (int dx = -half_size; dx <= half_size; dx++) {
            int nx = x + dx;
            int ny = y + dy;
            int rx = reflect101(nx, width);
            int ry = reflect101(ny, height);
            sum += input[ry * width + rx];
        }
    }
    
    // Normalize by window area
    output[y * width + x] = sum / (window_size * window_size);
}

__kernel void compute_flow(
    __global const float* numerator,
    __global const float* denominator,
    __global float* output,
    int width,
    int height,
    int topMargin,
    int bottomMargin,
    float lambda_reg)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= get_global_size(0) || y >= get_global_size(1)) return;
    if (y < topMargin || y >= height - bottomMargin) return;
    
    int idx = y * get_global_size(0) + x;
    float num = numerator[idx];
    float denom = denominator[idx] + lambda_reg; 
    output[idx] = num / denom;
}

__kernel void integral_image_row(
    __global const float* input,
    __global float* output,
    int width,
    int height,
    int window_size,
    int topMargin,
    int bottomMargin)
{
    int y = get_global_id(1);
    if (y >= height) return;
    if (y < topMargin || y >= height - bottomMargin) return;
    
    float sum = 0.0f;
    int half_window_size = window_size / 2;
    int expanded_width = width + window_size;
    
    for (int dx = -half_window_size; dx < width + half_window_size; dx++) {
        int rx = reflect101(dx, width);
        sum += input[y * width + rx];
        output[y * expanded_width + (dx + half_window_size)] = sum;
    }
}

__kernel void integral_image_col(
    __global const float* input,
    __global float* output,
    int width,
    int height,
    int window_size,
    int topMargin,
    int bottomMargin)
{
    int x = get_global_id(0);
    if (x >= width + window_size) return;
    
    float sum = 0.0f;
    int half_window_size = window_size / 2;
    int expanded_width = width + window_size;
    int expanded_height = height + window_size;
    
    for (int dy = -half_window_size; dy < height + half_window_size; dy++) {
        int ry = reflect101(dy, height);
        if (ry >= topMargin && ry < height - bottomMargin) {
            sum += input[ry * expanded_width + x];
        }
        output[(dy + half_window_size) * expanded_width + x] = sum;
    }
}

__kernel void box_filter_integral(
    __global const float* integral,
    __global float* output,
    int width,
    int height,
    int window_size,
    int topMargin,
    int bottomMargin)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) return;
    if (y < topMargin || y >= height - bottomMargin) return;

    int half_size = window_size / 2;
    int expanded_width = width + window_size;

    int x_orig = x;
    int y_orig = y;

    x = x/100; x = x*100 + 50;
    y = y/100; y = y*100 + 50;

    int x0 = x;
    int y0 = y;
    int x1 = x + window_size;
    int y1 = y + window_size;

    float A = integral[y0 * expanded_width + x0];
    float B = integral[y0 * expanded_width + x1];
    float C = integral[y1 * expanded_width + x0];
    float D = integral[y1 * expanded_width + x1];

    float sum = D - B - C + A;
    output[y_orig * width + x_orig] = sum / (window_size * window_size);
}
)";

DPflow::DPflow(cl_context context_, cl_command_queue queue_, cl_device_id device_, int window_size, float lambda_reg, bool enableSuperpixels, float compactness, int numSuperpixels, bool green, int grid_spacing, int square_size, int disparity_range)
    : context(context_), queue(queue_), device(device_), window_size(window_size), lambda_reg(lambda_reg), enableSuperpixels(enableSuperpixels), compactness(compactness), numSuperpixels(numSuperpixels), green(green), grid_spacing(grid_spacing), square_size(square_size), disparity_range(disparity_range) {
    buildKernels();
}

DPflow::~DPflow() {
    releaseBuffers();
    if (flow_kernel) clReleaseKernel(flow_kernel);
    if (compute_flow_kernel) clReleaseKernel(compute_flow_kernel);

    if (enableSuperpixels) {
        if (slic) {
            delete slic;
            slic = nullptr;
        }
    } else {
        if (numerator_integral_row_kernel) clReleaseKernel(numerator_integral_row_kernel);
        if (numerator_integral_col_kernel) clReleaseKernel(numerator_integral_col_kernel);
        if (denominator_integral_row_kernel) clReleaseKernel(denominator_integral_row_kernel);
        if (denominator_integral_col_kernel) clReleaseKernel(denominator_integral_col_kernel);
        if (box_filter_numerator_integral_kernel) clReleaseKernel(box_filter_numerator_integral_kernel);
        if (box_filter_denominator_integral_kernel) clReleaseKernel(box_filter_denominator_integral_kernel);
    }
    if (program) clReleaseProgram(program);
}

void DPflow::buildKernels() {
    cl_int err;
    program = clCreateProgramWithSource(context, 1, &kernelSource_dpflow, NULL, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create program");
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, &log[0], NULL);
        std::cerr << "Build log:\n" << &log[0] << std::endl;
        throw std::runtime_error("Failed to build program");
    }
    flow_kernel = clCreateKernel(program, "compute_optical_flow_x", &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create flow kernel");

    if (enableSuperpixels) {
        slic = new SLICSuperpixels(context, queue, device, compactness, numSuperpixels);
    } else {
        numerator_integral_row_kernel = clCreateKernel(program, "integral_image_row", &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create numerator integral row kernel");
        numerator_integral_col_kernel = clCreateKernel(program, "integral_image_col", &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create numerator integral col kernel");
        box_filter_numerator_integral_kernel = clCreateKernel(program, "box_filter_integral", &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create box filter numeratorintegral kernel");
        denominator_integral_row_kernel = clCreateKernel(program, "integral_image_row", &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create denominator integral row kernel");
        denominator_integral_col_kernel = clCreateKernel(program, "integral_image_col", &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create denominator integral col kernel");
        box_filter_denominator_integral_kernel = clCreateKernel(program, "box_filter_integral", &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create box filter denominator integral kernel");
    }

    compute_flow_kernel = clCreateKernel(program, "compute_flow", &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create compute flow kernel");
}

void DPflow::allocateBuffers(int width_, int height_, int channels_, int topMargin_, int bottomMargin_) {
    releaseBuffers();
    width = width_;
    height = height_;
    channels = channels_;
    topMargin = topMargin_;
    bottomMargin = bottomMargin_;
    size_t output_size = width * height * sizeof(float);
    size_t integral_row_size = (width + window_size) * height * sizeof(float);
    size_t integral_col_size = (width + window_size) * (height + window_size) * sizeof(float);
    global_size[0] = static_cast<size_t>(width);
    global_size[1] = static_cast<size_t>(height);
    row_global[0] = 1;
    row_global[1] = static_cast<size_t>(height);
    col_global[0] = static_cast<size_t>(width + window_size);
    col_global[1] = 1;
    clNumeratorOutput = clCreateBuffer(context, CL_MEM_READ_WRITE, output_size, nullptr, nullptr);
    clDenominatorOutput = clCreateBuffer(context, CL_MEM_READ_WRITE, output_size, nullptr, nullptr);
    clNumeratorFinalOutput = clCreateBuffer(context, CL_MEM_READ_WRITE, output_size, nullptr, nullptr);
    clDenominatorFinalOutput = clCreateBuffer(context, CL_MEM_READ_WRITE, output_size, nullptr, nullptr);
    if (enableSuperpixels) {
        if (slic) {
            cl_int err;
            clLabels = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        width * height * sizeof(int), nullptr, &err);
            if (err != CL_SUCCESS) throw std::runtime_error("Failed to create labels buffer");
            slic->allocateBuffers(width, height, true);
        }
    } else {
        clNumeratorIntegralBuffer1 = clCreateBuffer(context, CL_MEM_READ_WRITE, integral_row_size, nullptr, nullptr);
        clNumeratorIntegralBuffer2 = clCreateBuffer(context, CL_MEM_READ_WRITE, integral_col_size, nullptr, nullptr);
        clDenominatorIntegralBuffer1 = clCreateBuffer(context, CL_MEM_READ_WRITE, integral_row_size, nullptr, nullptr);
        clDenominatorIntegralBuffer2 = clCreateBuffer(context, CL_MEM_READ_WRITE, integral_col_size, nullptr, nullptr);
    }
}

void DPflow::releaseBuffers() {
    if (clNumeratorOutput) clReleaseMemObject(clNumeratorOutput); clNumeratorOutput = nullptr;
    if (clDenominatorOutput) clReleaseMemObject(clDenominatorOutput); clDenominatorOutput = nullptr;
    if (clNumeratorFinalOutput) clReleaseMemObject(clNumeratorFinalOutput); clNumeratorFinalOutput = nullptr;
    if (clDenominatorFinalOutput) clReleaseMemObject(clDenominatorFinalOutput); clDenominatorFinalOutput = nullptr;
    if (enableSuperpixels) {
        if (slic) {
            if (clLabels) clReleaseMemObject(clLabels); clLabels = nullptr;
            slic->releaseBuffers();
        }
    } else {
        if (clNumeratorIntegralBuffer1) clReleaseMemObject(clNumeratorIntegralBuffer1); clNumeratorIntegralBuffer1 = nullptr;
        if (clNumeratorIntegralBuffer2) clReleaseMemObject(clNumeratorIntegralBuffer2); clNumeratorIntegralBuffer2 = nullptr;
        if (clDenominatorIntegralBuffer1) clReleaseMemObject(clDenominatorIntegralBuffer1); clDenominatorIntegralBuffer1 = nullptr;
        if (clDenominatorIntegralBuffer2) clReleaseMemObject(clDenominatorIntegralBuffer2); clDenominatorIntegralBuffer2 = nullptr;
    }
}

void DPflow::setKernelArgs(const cl_mem clInputL, const cl_mem clInputR, const cl_mem clInputTotalImage, const cl_mem clMask, const cl_mem clFinalOutput, int flag) {
    // Set flow_kernel arguments
    clSetKernelArg(flow_kernel, 0, sizeof(cl_mem), &clInputL);
    clSetKernelArg(flow_kernel, 1, sizeof(cl_mem), &clInputR);
    clSetKernelArg(flow_kernel, 2, sizeof(cl_mem), &clMask);
    clSetKernelArg(flow_kernel, 3, sizeof(cl_mem), &clNumeratorOutput);
    clSetKernelArg(flow_kernel, 4, sizeof(cl_mem), &clDenominatorOutput);
    clSetKernelArg(flow_kernel, 5, sizeof(int), &width);
    clSetKernelArg(flow_kernel, 6, sizeof(int), &height);
    clSetKernelArg(flow_kernel, 7, sizeof(int), &channels);
    clSetKernelArg(flow_kernel, 8, sizeof(int), &topMargin);
    clSetKernelArg(flow_kernel, 9, sizeof(int), &bottomMargin);
    clSetKernelArg(flow_kernel, 10, sizeof(float), &lambda_reg);
    clSetKernelArg(flow_kernel, 11, sizeof(int), &green);
    clSetKernelArg(flow_kernel, 12, sizeof(cl_mem), &clFinalOutput);   // MATT's EDIT
    clSetKernelArg(flow_kernel, 13, sizeof(int), &flag);
    clSetKernelArg(flow_kernel, 14, sizeof(int), &grid_spacing);
    clSetKernelArg(flow_kernel, 15, sizeof(int), &square_size);
    clSetKernelArg(flow_kernel, 16, sizeof(int), &disparity_range);

    /*if (enableSuperpixels) {
        if (slic) {
            slic->setKernelArgs(clInputTotalImage, clLabels, clNumeratorOutput, clDenominatorOutput, clNumeratorFinalOutput, clDenominatorFinalOutput);
        }
    } else {
        // Set numerator_integral_row_kernel and numerator_integral_col_kernel arguments
        clSetKernelArg(numerator_integral_row_kernel, 0, sizeof(cl_mem), &clNumeratorOutput);
        clSetKernelArg(numerator_integral_row_kernel, 1, sizeof(cl_mem), &clNumeratorIntegralBuffer1);
        clSetKernelArg(numerator_integral_row_kernel, 2, sizeof(int), &width);
        clSetKernelArg(numerator_integral_row_kernel, 3, sizeof(int), &height);
        clSetKernelArg(numerator_integral_row_kernel, 4, sizeof(int), &window_size);
        clSetKernelArg(numerator_integral_row_kernel, 5, sizeof(int), &topMargin);
        clSetKernelArg(numerator_integral_row_kernel, 6, sizeof(int), &bottomMargin);
        
        clSetKernelArg(numerator_integral_col_kernel, 0, sizeof(cl_mem), &clNumeratorIntegralBuffer1);
        clSetKernelArg(numerator_integral_col_kernel, 1, sizeof(cl_mem), &clNumeratorIntegralBuffer2);
        clSetKernelArg(numerator_integral_col_kernel, 2, sizeof(int), &width);
        clSetKernelArg(numerator_integral_col_kernel, 3, sizeof(int), &height);
        clSetKernelArg(numerator_integral_col_kernel, 4, sizeof(int), &window_size);
        clSetKernelArg(numerator_integral_col_kernel, 5, sizeof(int), &topMargin);
        clSetKernelArg(numerator_integral_col_kernel, 6, sizeof(int), &bottomMargin);
        
        // Set box_filter_numerator_integral_kernel arguments
        clSetKernelArg(box_filter_numerator_integral_kernel, 0, sizeof(cl_mem), &clNumeratorIntegralBuffer2);
        clSetKernelArg(box_filter_numerator_integral_kernel, 1, sizeof(cl_mem), &clNumeratorFinalOutput);
        clSetKernelArg(box_filter_numerator_integral_kernel, 2, sizeof(int), &width);
        clSetKernelArg(box_filter_numerator_integral_kernel, 3, sizeof(int), &height);
        clSetKernelArg(box_filter_numerator_integral_kernel, 4, sizeof(int), &window_size);
        clSetKernelArg(box_filter_numerator_integral_kernel, 5, sizeof(int), &topMargin);
        clSetKernelArg(box_filter_numerator_integral_kernel, 6, sizeof(int), &bottomMargin);
        
        // Set denominator_integral_row_kernel and denominator_integral_col_kernel arguments
        clSetKernelArg(denominator_integral_row_kernel, 0, sizeof(cl_mem), &clDenominatorOutput);
        clSetKernelArg(denominator_integral_row_kernel, 1, sizeof(cl_mem), &clDenominatorIntegralBuffer1);
        clSetKernelArg(denominator_integral_row_kernel, 2, sizeof(int), &width);
        clSetKernelArg(denominator_integral_row_kernel, 3, sizeof(int), &height);
        clSetKernelArg(denominator_integral_row_kernel, 4, sizeof(int), &window_size);
        clSetKernelArg(denominator_integral_row_kernel, 5, sizeof(int), &topMargin);
        clSetKernelArg(denominator_integral_row_kernel, 6, sizeof(int), &bottomMargin);
        
        clSetKernelArg(denominator_integral_col_kernel, 0, sizeof(cl_mem), &clDenominatorIntegralBuffer1);
        clSetKernelArg(denominator_integral_col_kernel, 1, sizeof(cl_mem), &clDenominatorIntegralBuffer2);
        clSetKernelArg(denominator_integral_col_kernel, 2, sizeof(int), &width);
        clSetKernelArg(denominator_integral_col_kernel, 3, sizeof(int), &height);
        clSetKernelArg(denominator_integral_col_kernel, 4, sizeof(int), &window_size);
        clSetKernelArg(denominator_integral_col_kernel, 5, sizeof(int), &topMargin);
        clSetKernelArg(denominator_integral_col_kernel, 6, sizeof(int), &bottomMargin);
        
        // Set box_filter_denominator_integral_kernel arguments
        clSetKernelArg(box_filter_denominator_integral_kernel, 0, sizeof(cl_mem), &clDenominatorIntegralBuffer2);
        clSetKernelArg(box_filter_denominator_integral_kernel, 1, sizeof(cl_mem), &clDenominatorFinalOutput);
        clSetKernelArg(box_filter_denominator_integral_kernel, 2, sizeof(int), &width);
        clSetKernelArg(box_filter_denominator_integral_kernel, 3, sizeof(int), &height);
        clSetKernelArg(box_filter_denominator_integral_kernel, 4, sizeof(int), &window_size);
        clSetKernelArg(box_filter_denominator_integral_kernel, 5, sizeof(int), &topMargin);
        clSetKernelArg(box_filter_denominator_integral_kernel, 6, sizeof(int), &bottomMargin);
    }*/

    // Set compute_flow_kernel arguments
    /*clSetKernelArg(compute_flow_kernel, 0, sizeof(cl_mem), &clNumeratorFinalOutput);
    clSetKernelArg(compute_flow_kernel, 1, sizeof(cl_mem), &clDenominatorFinalOutput);
    clSetKernelArg(compute_flow_kernel, 2, sizeof(cl_mem), &clFinalOutput);
    clSetKernelArg(compute_flow_kernel, 3, sizeof(int), &width);
    clSetKernelArg(compute_flow_kernel, 4, sizeof(int), &height);
    clSetKernelArg(compute_flow_kernel, 5, sizeof(int), &topMargin);
    clSetKernelArg(compute_flow_kernel, 6, sizeof(int), &bottomMargin);
    clSetKernelArg(compute_flow_kernel, 7, sizeof(float), &lambda_reg);*/

}

void DPflow::compute_optical_flow_x() {
    cl_int err;
    cl_event flow_event, numerator_row_event, numerator_col_event, denominator_row_event, denominator_col_event, box_filter_numerator_event, box_filter_denominator_event, final_event;

    // Compute initial optical flow
    err = clEnqueueNDRangeKernel(queue, flow_kernel, 2, NULL, global_size, NULL, 0, NULL, &flow_event);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to execute flow kernel");
    clFinish(queue);

    int tmp = 0;  // MATT's EDIT

    if (enableSuperpixels) {
        if (slic) {
            slic->processTwoImages();
            clFinish(queue);
        }
        // Compute final flow
        err = clEnqueueNDRangeKernel(queue, compute_flow_kernel, 2, NULL, global_size, NULL, 0, NULL, &final_event);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to execute compute flow kernel");
    } else if (tmp) {    
        // Numerator Row-wise integral
        err = clEnqueueNDRangeKernel(queue, numerator_integral_row_kernel, 2, NULL, row_global, NULL, 1, &flow_event, &numerator_row_event);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to execute numerator integral row kernel");

        // Numerator Col-wise integral
        err = clEnqueueNDRangeKernel(queue, numerator_integral_col_kernel, 2, NULL, col_global, NULL, 1, &numerator_row_event, &numerator_col_event);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to execute numerator integral col kernel");

        // Numerator Box filter from integral image
        err = clEnqueueNDRangeKernel(queue, box_filter_numerator_integral_kernel, 2, NULL, global_size, NULL, 1, &numerator_col_event, &box_filter_numerator_event);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to execute numerator box filter integral kernel");

        // Denominator Row-wise integral
        err = clEnqueueNDRangeKernel(queue, denominator_integral_row_kernel, 2, NULL, row_global, NULL, 1, &flow_event, &denominator_row_event);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to execute denominator integral row kernel");

        // Denominator Col-wise integral
        err = clEnqueueNDRangeKernel(queue, denominator_integral_col_kernel, 2, NULL, col_global, NULL, 1, &denominator_row_event, &denominator_col_event);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to execute denominator integral col kernel");

        // Denominator Box filter from integral image
        err = clEnqueueNDRangeKernel(queue, box_filter_denominator_integral_kernel, 2, NULL, global_size, NULL, 1, &denominator_col_event, &box_filter_denominator_event);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to execute denominator box filter integral kernel");

        // Compute final flow
        // Create list of events to wait for before final computation
        cl_event wait_events[2] = {numerator_col_event, denominator_col_event};
        err = clEnqueueNDRangeKernel(queue, compute_flow_kernel, 2, NULL, global_size, NULL, 2, wait_events, &final_event);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to execute compute flow kernel");
    }
    
    // Wait for the final event to complete
    clWaitForEvents(1, &flow_event);

    // Release all events to avoid memory leaks
    clReleaseEvent(flow_event);
    if (enableSuperpixels) {
    } else if(tmp) {
        clReleaseEvent(numerator_row_event);
        clReleaseEvent(numerator_col_event);
        clReleaseEvent(box_filter_numerator_event);
        clReleaseEvent(denominator_row_event);
        clReleaseEvent(denominator_col_event);
        clReleaseEvent(box_filter_denominator_event);
    }
    //clReleaseEvent(final_event);
}

/*
int main() {
    try {
        // Example images
        cv::Mat img_l = cv::imread("left.jpg");
        cv::Mat img_r = cv::imread("right.jpg");
        cv::Mat mask;
        if (img_l.empty() || img_r.empty()) {
            std::cerr << "Failed to load images" << std::endl;
            return -1;
        }
        
        // Check if images are different
        cv::Mat diff;
        cv::absdiff(img_l, img_r, diff);
        double minVal, maxVal;
        cv::minMaxLoc(diff, &minVal, &maxVal);
        std::cout << "Input image difference range: [" << minVal << ", " << maxVal << "]" << std::endl;
        
        img_l.convertTo(img_l, CV_32FC3, 1.0 / 255.0);
        img_r.convertTo(img_r, CV_32FC3, 1.0 / 255.0);
        cv::Mat img_total = (img_l + img_r) / 2.0f;
        mask = cv::Mat::ones(img_l.rows, img_l.cols, CV_32F);
        int width = img_l.cols;
        int height = img_l.rows;
        int channels = 3;
        size_t img_size = height * width * channels * sizeof(float);
        size_t mask_size = height * width * sizeof(float);
        size_t output_size = height * width * sizeof(float);

        // OpenCL setup
        cl_int err;
        cl_platform_id platform;
        err = clGetPlatformIDs(1, &platform, NULL);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to get platform");
        cl_device_id device;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to get device");
        cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create context");
        cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create command queue");

        // Allocate device buffers and upload data
        cl_mem d_img_l = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, img_size, img_l.data, &err);
        cl_mem d_img_r = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, img_size, img_r.data, &err);
        cl_mem d_img_total = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, img_size, img_total.data, &err);
        cl_mem d_mask = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, mask_size, mask.data, &err);
        cl_mem d_flow = clCreateBuffer(context, CL_MEM_READ_WRITE, output_size, NULL, &err);

        // Create DPflow object
        int window_size = 202;
        float lambda_reg = 1e-10f;
        bool enableSuperpixels = true;
        float compactness = 13.0f;
        int numSuperpixels = 500;
        bool green = false;
        DPflow dpflow(context, queue, device, window_size, lambda_reg, enableSuperpixels, compactness, numSuperpixels, green);
        dpflow.allocateBuffers(width, height, channels);
        dpflow.setKernelArgs(d_img_l, d_img_r, d_img_total, d_mask, d_flow);

        // Compute optical flow
        clFinish(queue);
        auto dpflow_start = std::chrono::high_resolution_clock::now();
        dpflow.compute_optical_flow_x();
        clFinish(queue);
        auto dpflow_end = std::chrono::high_resolution_clock::now();
        clFinish(queue);
        double dpflow_ms = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(dpflow_end - dpflow_start).count());
        std::cout << "DPflow computation took " << dpflow_ms << " ms." << std::endl;

        // Download result
        cv::Mat result(height, width, CV_32F);
        clEnqueueReadBuffer(queue, d_flow, CL_TRUE, 0, output_size, result.data, 0, NULL, NULL);

        // Normalize flow for visualization
        double flow_minVal, flow_maxVal;
        cv::minMaxLoc(result, &flow_minVal, &flow_maxVal);
        cv::Mat flow_norm;
        result.convertTo(flow_norm, CV_8UC1, 255.0 / (flow_maxVal - flow_minVal), -flow_minVal * 255.0 / (flow_maxVal - flow_minVal));
        cv::Mat flow_color;
        cv::applyColorMap(flow_norm, flow_color, cv::COLORMAP_TURBO);
        cv::namedWindow("Optical Flow", cv::WINDOW_NORMAL);
        cv::imshow("Optical Flow", flow_color);
        cv::imwrite("optical_flow.jpg", flow_color);
        cv::waitKey(0);

        // Print image shapes and flow value range
        std::cout << "\nImage Shapes:" << std::endl;
        std::cout << "Left image: " << img_l.cols << "x" << img_l.rows << "x" << img_l.channels() << std::endl;
        std::cout << "Right image: " << img_r.cols << "x" << img_r.rows << "x" << img_r.channels() << std::endl;
        std::cout << "Flow result: " << result.cols << "x" << result.rows << std::endl;
        double flowMin, flowMax;
        cv::minMaxLoc(result, &flowMin, &flowMax);
        std::cout << "\nFlow Value Range:" << std::endl;
        std::cout << "Min flow: " << flowMin << std::endl;
        std::cout << "Max flow: " << flowMax << std::endl;

        // Cleanup
        clReleaseMemObject(d_img_l);
        clReleaseMemObject(d_img_r);
        clReleaseMemObject(d_mask);
        clReleaseMemObject(d_flow);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}
*/