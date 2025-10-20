#include <CL/cl.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <stdexcept>

#include "warpPerspective.h"

// OpenCL kernel source code
const char* warpPerspectiveKernel = R"(
__kernel void warpPerspective(
    __global const float* src,
    __global float* dst,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    __constant float* M,
    float border_value)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= dst_width || y >= dst_height) return;
    
    // Calculate source coordinates using perspective transform
    float X0 = M[0] * x + M[1] * y + M[2];
    float Y0 = M[3] * x + M[4] * y + M[5];
    float W = M[6] * x + M[7] * y + M[8];
    
    // Handle division by zero
    W = W != 0.0f ? 1.0f / W : 0.0f;
    
    // Calculate source coordinates
    int sx = convert_int_sat_rte(X0 * W);
    int sy = convert_int_sat_rte(Y0 * W);

    // Calculate destination index
    int dst_idx = y * dst_width + x;

    // Check if source coordinates are within bounds
    if (sx >= 0 && sx < src_width && sy >= 0 && sy < src_height)
    {
        // Calculate source index and copy pixel
        int src_idx = sy * src_width + sx;
        dst[dst_idx] = src[src_idx];
    }
    else
    {
        // Use border value if out of bounds
        dst[dst_idx] = border_value;
    }
}
)";

WarpPerspective::WarpPerspective(cl_context context_, cl_command_queue queue_, cl_device_id device_)
    : context(context_), queue(queue_), device(device_) {
    buildKernels();
}

WarpPerspective::~WarpPerspective() {
    releaseBuffers();
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
}

void WarpPerspective::buildKernels() {
    cl_int err;
    program = clCreateProgramWithSource(context, 1, &warpPerspectiveKernel, NULL, &err);
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
    
    kernel = clCreateKernel(program, "warpPerspective", &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create kernel");
}

void WarpPerspective::allocateBuffers(int src_width_, int src_height_, int dst_width_, int dst_height_) {
    releaseBuffers();
    src_width = src_width_;
    src_height = src_height_;
    dst_width = dst_width_;
    dst_height = dst_height_;
    global_size[0] = static_cast<size_t>(dst_width);
    global_size[1] = static_cast<size_t>(dst_height);
}

void WarpPerspective::releaseBuffers() {
    // No buffers to release in this case
}

void WarpPerspective::setKernelArgs(const cl_mem clSrc, const cl_mem clDst, const cl_mem clM, float border_value) {
    cl_int err;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &clSrc);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to set kernel argument 0");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &clDst);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to set kernel argument 1");
    err = clSetKernelArg(kernel, 2, sizeof(int), &src_width);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to set kernel argument 2");
    err = clSetKernelArg(kernel, 3, sizeof(int), &src_height);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to set kernel argument 3");
    err = clSetKernelArg(kernel, 4, sizeof(int), &dst_width);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to set kernel argument 4");
    err = clSetKernelArg(kernel, 5, sizeof(int), &dst_height);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to set kernel argument 5");
    err = clSetKernelArg(kernel, 6, sizeof(cl_mem), &clM);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to set kernel argument 6");
    err = clSetKernelArg(kernel, 7, sizeof(float), &border_value);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to set kernel argument 7");
}

void WarpPerspective::compute() {
    cl_int err;
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to execute kernel");
    clFinish(queue);
}

/*
int main() {
    try {
        // // Example image
        // cv::Mat src = cv::Mat::zeros(100, 100, CV_8UC1);
        // cv::circle(src, cv::Point(50, 50), 25, cv::Scalar(255), -1);

        // Set up camera and SLM parameters
        const size_t cameraHeight = 1024;
        const size_t cameraWidth = 1224;
        const int SLM_WIDTH = 4000;
        const int SLM_HEIGHT = 2464;
        const double LEFT_FREQ = 1.0/4.0;  // Left 2/3 of image
        const double RIGHT_FREQ = -1.0/5.0; // Right 1/3 of image

        // cv::Mat M = (cv::Mat_<double>(3,3) << 1, 0, 0,
        //                                      0, 1, 0,
        //                                      0, 0, 1);
        cv::Mat M = (cv::Mat_<double>(3,3) << 2.69136089e+00,  2.53363595e-03,  2.65811566e+02,
                                             1.62845106e-02, -2.67265181e+00,  2.55528807e+03,
                                             7.90199867e-06,  1.41308659e-06,  1.00000000e+00);

        cv::Mat M32f;
        M.convertTo(M32f, CV_32F);
        cv::Mat Minv32f;
        cv::invert(M32f, Minv32f);
        
        size_t M_size = M32f.total() * M32f.elemSize();

        // // Destination size
        // cv::Size dsize(200, 200);
        // cv::Mat dst(dsize, src.type());

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

        // Create WarpPerspective object
        WarpPerspective warp(context, queue, device);
        warp.allocateBuffers(cameraWidth, cameraHeight, SLM_WIDTH, SLM_HEIGHT);

        // // Allocate device buffers
        // size_t src_size = src.total() * src.elemSize();
        // size_t dst_size = dst.total() * dst.elemSize();

        // cv::Mat src_contig = src.isContinuous() ? src : src.clone();
        // cv::Mat dst_contig = dst.isContinuous() ? dst : dst.clone();

        // Create spatial frequency map in camera dimensions
        cv::Mat spatialFreqMap(static_cast<int>(cameraHeight), static_cast<int>(cameraWidth), CV_32F);
        int leftWidth = static_cast<int>(static_cast<double>(cameraWidth) * 2.0 / 3.0);
        spatialFreqMap.colRange(0, leftWidth).setTo(LEFT_FREQ);
                spatialFreqMap.colRange(leftWidth, static_cast<int>(cameraWidth)).setTo(RIGHT_FREQ);

        // Allocate OpenCL buffers
        cl_mem clSpatialFreqMap = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                                                cameraWidth * cameraHeight * sizeof(float), 
                                                nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create spatial frequency map buffer");

        cl_mem clWarpedFreqMap = clCreateBuffer(context, CL_MEM_READ_WRITE, 
                                               SLM_WIDTH * SLM_HEIGHT * sizeof(float), 
                                               nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create warped frequency map buffer");

        cl_mem clHomography = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                                            9 * sizeof(float), nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create homography buffer");

        // Write data to buffers
        err = clEnqueueWriteBuffer(queue, clSpatialFreqMap, CL_TRUE, 0, 
                                  cameraWidth * cameraHeight * sizeof(float), 
                                  spatialFreqMap.data, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to write spatial frequency map to buffer");

        err = clEnqueueWriteBuffer(queue, clHomography, CL_TRUE, 0, 
                                  9 * sizeof(float), Minv32f.data, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to write homography matrix to buffer");

        // Set kernel arguments and compute
        warp.setKernelArgs(clSpatialFreqMap, clWarpedFreqMap, clHomography, 0.0);
        auto warp_start = std::chrono::high_resolution_clock::now();
        warp.compute();
        auto warp_end = std::chrono::high_resolution_clock::now();
        double warp_ms = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(warp_end - warp_start).count());
        std::cout << "WarpPerspective computation took " << warp_ms << " ms." << std::endl;

        // Download result
        // Allocate device buffers
        cv::Mat warpedFreqMap(static_cast<int>(SLM_HEIGHT), static_cast<int>(SLM_WIDTH), CV_32F);
        size_t dst_size = warpedFreqMap.total() * warpedFreqMap.elemSize();
        err = clEnqueueReadBuffer(queue, clWarpedFreqMap, CL_TRUE, 0, dst_size, warpedFreqMap.data, 0, NULL, NULL);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to read result");

        // Display results
        cv::namedWindow("Source", cv::WINDOW_NORMAL);
        cv::namedWindow("Result", cv::WINDOW_NORMAL);

        // Normalize the frequency maps to 0-1 range for colormap application
        cv::Mat spatialFreqMapNormalized, warpedFreqMapNormalized;
        cv::normalize(spatialFreqMap, spatialFreqMapNormalized, 0, 1, cv::NORM_MINMAX);
        cv::normalize(warpedFreqMap, warpedFreqMapNormalized, 0, 1, cv::NORM_MINMAX);

        // Convert to 8-bit unsigned char
        cv::Mat spatialFreqMap8U, warpedFreqMap8U;
        spatialFreqMapNormalized.convertTo(spatialFreqMap8U, CV_8UC1, 255.0);
        warpedFreqMapNormalized.convertTo(warpedFreqMap8U, CV_8UC1, 255.0);

        // Apply turbo colormap
        cv::Mat spatialFreqMapColor, warpedFreqMapColor;
        cv::applyColorMap(spatialFreqMap8U, spatialFreqMapColor, cv::COLORMAP_TURBO);
        cv::applyColorMap(warpedFreqMap8U, warpedFreqMapColor, cv::COLORMAP_TURBO);

        cv::imshow("Source", spatialFreqMapColor);
        cv::imshow("Result", warpedFreqMapColor);
        std::cout << "Source frequency map shape: " << spatialFreqMapColor.rows << "x" << spatialFreqMapColor.cols << "x" << spatialFreqMapColor.channels() << std::endl;
        std::cout << "Warped frequency map shape: " << warpedFreqMapColor.rows << "x" << warpedFreqMapColor.cols << "x" << warpedFreqMapColor.channels() << std::endl;
        cv::imwrite("source_image.jpg", spatialFreqMapColor);
        cv::imwrite("warped_result.jpg", warpedFreqMapColor);

        // Compare with OpenCV's warpPerspective
        cv::Mat cvWarpedFreqMap;
        auto cv_warp_start = std::chrono::high_resolution_clock::now();
        cv::warpPerspective(spatialFreqMap, cvWarpedFreqMap, M32f, cv::Size(SLM_WIDTH, SLM_HEIGHT));
        auto cv_warp_end = std::chrono::high_resolution_clock::now();
        double cv_warp_ms = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(cv_warp_end - cv_warp_start).count());
        std::cout << "OpenCV warpPerspective computation took " << cv_warp_ms << " ms." << std::endl;

        // Normalize and convert OpenCV result for display
        cv::Mat cvWarpedFreqMapNormalized;
        cv::normalize(cvWarpedFreqMap, cvWarpedFreqMapNormalized, 0, 1, cv::NORM_MINMAX);
        cv::Mat cvWarpedFreqMap8U;
        cvWarpedFreqMapNormalized.convertTo(cvWarpedFreqMap8U, CV_8UC1, 255.0);
        cv::Mat cvWarpedFreqMapColor;
        cv::applyColorMap(cvWarpedFreqMap8U, cvWarpedFreqMapColor, cv::COLORMAP_TURBO);

        // Display OpenCV result
        cv::namedWindow("OpenCV Result", cv::WINDOW_NORMAL);
        cv::imshow("OpenCV Result", cvWarpedFreqMapColor);
        cv::imwrite("opencv_result.jpg", cvWarpedFreqMapColor);

        // Compare results
        cv::Mat diff;
        cv::absdiff(warpedFreqMap, cvWarpedFreqMap, diff);
        double maxDiff;
        cv::minMaxLoc(diff, nullptr, &maxDiff);
        std::cout << "Maximum difference between OpenCL and OpenCV results: " << maxDiff << std::endl;

        cv::waitKey(0);

        // Cleanup
        clReleaseMemObject(clSpatialFreqMap);
        clReleaseMemObject(clWarpedFreqMap);
        clReleaseMemObject(clHomography);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
} 
*/
