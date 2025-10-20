#include "stdafx.h"
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
#include "slic.h"

// OpenCL kernel source code
const char* kernelSource1 = R"(
// Convert RGB to LAB color space
float3 rgb2lab(float r, float g, float b) {
    float3 lab;
    
    if (r > 0.04045f) r = pow((r + 0.055f) / 1.055f, 2.4f);
    else r = r / 12.92f;
    
    if (g > 0.04045f) g = pow((g + 0.055f) / 1.055f, 2.4f);
    else g = g / 12.92f;
    
    if (b > 0.04045f) b = pow((b + 0.055f) / 1.055f, 2.4f);
    else b = b / 12.92f;
    
    r *= 100.0f;
    g *= 100.0f;
    b *= 100.0f;
    
    float x = r * 0.4124f + g * 0.3576f + b * 0.1805f;
    float y = r * 0.2126f + g * 0.7152f + b * 0.0722f;
    float z = r * 0.0193f + g * 0.1192f + b * 0.9505f;
    
    x = x / 95.047f;
    y = y / 100.0f;
    z = z / 108.883f;
    
    if (x > 0.008856f) x = pow(x, 1.0f / 3.0f);
    else x = (7.787f * x) + (16.0f / 116.0f);
    
    if (y > 0.008856f) y = pow(y, 1.0f / 3.0f);
    else y = (7.787f * y) + (16.0f / 116.0f);
    
    if (z > 0.008856f) z = pow(z, 1.0f / 3.0f);
    else z = (7.787f * z) + (16.0f / 116.0f);
    
    lab.x = (116.0f * y) - 16.0f;     // L
    lab.y = 500.0f * (x - y);         // a
    lab.z = 200.0f * (y - z);         // b
    
    return lab;
}
)";

const char* kernelSource2 = R"(
// Initialize cluster centers
__kernel void initClusters(
    __global const float* input,
    __global float* clusters,
    int width, int height,
    int numSuperpixels) {
    int clusterIdx = get_global_id(0);
    if (clusterIdx >= numSuperpixels) return;
    
    // Calculate grid step
    float stepX = (float)width / sqrt((float)numSuperpixels);
    float stepY = (float)height / sqrt((float)numSuperpixels);
    
    // Calculate grid position
    int gridY = (int)floor(clusterIdx / floor((float)width / stepX));
    int gridX = clusterIdx - (gridY * floor((float)width / stepX));
    
    // Calculate center position
    int centerX = (int)(stepX * (gridX + 0.5f));
    int centerY = (int)(stepY * (gridY + 0.5f));
    
    // Clamp to image boundaries
    centerX = min(max(centerX, 0), width - 1);
    centerY = min(max(centerY, 0), height - 1);
    
    // Get center pixel color
    int pixIdx = (centerY * width + centerX) * 3;
    float r = input[pixIdx];
    float g = input[pixIdx + 1];
    float b = input[pixIdx + 2];
    
    // Convert to LAB
    float3 lab = rgb2lab(r, g, b);
    
    // Store cluster center [x, y, l, a, b]
    clusters[clusterIdx * 5 + 0] = (float)centerX;
    clusters[clusterIdx * 5 + 1] = (float)centerY;
    clusters[clusterIdx * 5 + 2] = lab.x;
    clusters[clusterIdx * 5 + 3] = lab.y;
    clusters[clusterIdx * 5 + 4] = lab.z;
}
)";

const char* kernelSource3 = R"(
// Assign pixels to clusters
__kernel void assignPixels(
    __global const float* input,
    __global const float* clusters,
    __global int* labels,
    int width, int height,
    int numSuperpixels,
    int regionSize,
    float compactness) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    int pixIdx = (y * width + x) * 3;
    float3 lab = rgb2lab(input[pixIdx], input[pixIdx + 1], input[pixIdx + 2]);
    
    // Initialize distance to a large value
    float minDist = 1.0e10f;
    int nearestCluster = -1;
    
    // Calculate grid step
    float stepX = (float)width / sqrt((float)numSuperpixels);
    float stepY = (float)height / sqrt((float)numSuperpixels);
    int searchRegionSize = (int)(max(stepX, stepY) * 2.0f);
    
    // Calculate grid position of the pixel
    int gridX = (int)floor(x / stepX);
    int gridY = (int)floor(y / stepY);
    
    // Search in a 2x2 grid neighborhood
    for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
            int gridIdxX = gridX + i;
            int gridIdxY = gridY + j;
            
            // Skip if outside grid boundaries
            if (gridIdxX < 0 || gridIdxY < 0 || 
                gridIdxX >= (int)ceil((float)width / stepX) || 
                gridIdxY >= (int)ceil((float)height / stepY))
                continue;
            
            // Calculate cluster index
            int clusterId = gridIdxY * (int)floor((float)width / stepX) + gridIdxX;
            if (clusterId >= numSuperpixels) continue;
            
            // Get cluster center
            float cX = clusters[clusterId * 5 + 0];
            float cY = clusters[clusterId * 5 + 1];
            
            // Skip if outside search region
            if (fabs(cX - x) > searchRegionSize || fabs(cY - y) > searchRegionSize)
                continue;
            
            float cL = clusters[clusterId * 5 + 2];
            float cA = clusters[clusterId * 5 + 3];
            float cB = clusters[clusterId * 5 + 4];
            
            // Calculate color distance (CIELAB)
            float distL = cL - lab.x;
            float distA = cA - lab.y;
            float distB = cB - lab.z;
            float distColor = sqrt(distL*distL + distA*distA + distB*distB);
            
            // Calculate spatial distance
            float distX = cX - x;
            float distY = cY - y;
            float distSpace = sqrt(distX*distX + distY*distY);
            
            // SLIC distance metric: D = distColor + (compactness/regionSize) * distSpace
            float dist = distColor + (compactness / (float)regionSize) * distSpace;
            
            // Update nearest cluster
            if (dist < minDist) {
                minDist = dist;
                nearestCluster = clusterId;
            }
        }
    }
    
    // Assign pixel to nearest cluster
    if (nearestCluster >= 0) {
        labels[y * width + x] = nearestCluster;
    }
}
)";

const char* kernelSource4 = R"(
// Update cluster centers
__kernel void updateClusters(
    __global const float* input,
    __global float* clusters,
    __global const int* labels,
    int width, int height,
    int numSuperpixels) {
    int idx = get_global_id(0);
    int x = idx % width;
    int y = idx / width;
    
    if (x >= width || y >= height) return;
    
    int pixIdx = (y * width + x) * 3;
    int label = labels[y * width + x];
    
    if (label >= 0 && label < numSuperpixels) {
        // Convert RGB to LAB
        float3 lab = rgb2lab(input[pixIdx], input[pixIdx + 1], input[pixIdx + 2]);
        
        // Use local memory for accumulation
        __local float localAccum[5];
        if (get_local_id(0) == 0) {
            for (int i = 0; i < 5; i++) {
                localAccum[i] = 0.0f;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Accumulate values
        localAccum[0] += (float)x;
        localAccum[1] += (float)y;
        localAccum[2] += lab.x;
        localAccum[3] += lab.y;
        localAccum[4] += lab.z;
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Write back to global memory
        if (get_local_id(0) == 0) {
            for (int i = 0; i < 5; i++) {
                clusters[label * 5 + i] = localAccum[i];
            }
        }
    }
}
)";

const char* kernelSource5 = R"(
#define SCALE 1000

__kernel void accumulateClusters(
    __global const float* input,
    __global const int* labels,
    __global int* accum,   // [numClusters * 5]
    __global int* counts,  // [numClusters]
    int width, int height)
{
    int idx = get_global_id(0);
    int x = idx % width;
    int y = idx / width;
    int label = labels[idx];
    float3 lab = rgb2lab(input[idx*3], input[idx*3+1], input[idx*3+2]);
    atomic_add(&accum[label*5+0], (int)(x * SCALE));
    atomic_add(&accum[label*5+1], (int)(y * SCALE));
    atomic_add(&accum[label*5+2], (int)(lab.x * SCALE));
    atomic_add(&accum[label*5+3], (int)(lab.y * SCALE));
    atomic_add(&accum[label*5+4], (int)(lab.z * SCALE));
    atomic_inc(&counts[label]);
}
)";

const char* kernelSource6 = R"(
#define SCALE 1000

__kernel void finalizeClusters(
    __global int* accum,
    __global int* counts,
    __global float* clusters,
    int numClusters)
{
    int i = get_global_id(0);
    if (counts[i] > 0) {
        clusters[i*5+0] = (float)accum[i*5+0] / (counts[i] * SCALE);
        clusters[i*5+1] = (float)accum[i*5+1] / (counts[i] * SCALE);
        clusters[i*5+2] = (float)accum[i*5+2] / (counts[i] * SCALE);
        clusters[i*5+3] = (float)accum[i*5+3] / (counts[i] * SCALE);
        clusters[i*5+4] = (float)accum[i*5+4] / (counts[i] * SCALE);
    }
}
)";

const char* kernelSource7 = R"(
float atomic_add_float(__global float* addr, float val) {
    __global uint* address_as_uint = (__global uint*)addr;
    uint old = *address_as_uint;
    uint assumed;
    do {
        assumed = old;
        old = atomic_cmpxchg(
            address_as_uint,
            assumed,
            as_uint(val + as_float(assumed)));
    } while (assumed != old);
    return as_float(old);
}

float atomic_add_float_local(__local float* addr, float val) {
    __local uint* address_as_uint = (__local uint*)addr;
    uint old = *address_as_uint;
    uint assumed;
    do {
        assumed = old;
        old = atomic_cmpxchg(
            address_as_uint,
            assumed,
            as_uint(val + as_float(assumed)));
    } while (assumed != old);
    return as_float(old);
}

__kernel void superpixel_atomic_average(
    __global const int* labels,
    __global const float* image,     // [width*height]
    __global float* sum,             // [numSuperpixels]
    __global int* count,             // [numSuperpixels]
    int numPixels)
{
    int idx = get_global_id(0);
    if (idx >= numPixels) return;
    int label = labels[idx];
    float value = image[idx];
    atomic_add_float(&sum[label], value);
    atomic_inc(&count[label]);
}

__kernel void superpixel_efficient_average(
    __global const int* labels,
    __global const float* image,
    __global float* sum,             // [numSuperpixels]
    __global int* count,             // [numSuperpixels]
    __local float* local_sums,       // Local memory for work group
    __local int* local_counts,       // Local memory for work group
    int numSuperpixels,
    int numPixels)
{
    // Get indices
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);
    
    // Initialize local memory
    for (int i = lid; i < numSuperpixels; i += local_size) {
        local_sums[i] = 0.0f;
        local_counts[i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // First phase: Accumulate in local memory
    if (gid < numPixels) {
        int label = labels[gid];
        float value = image[gid];
        
        // Using atomic operations on local memory (much faster than global)
        atomic_add_float_local(&local_sums[label], value);
        atomic_inc(&local_counts[label]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Second phase: Accumulate from local to global
    // Only one thread per label in the work group will do this
    for (int label = lid; label < numSuperpixels; label += local_size) {
        if (local_counts[label] > 0) {
            atomic_add_float(&sum[label], local_sums[label]);
            atomic_add(&count[label], local_counts[label]);
        }
    }
}

__kernel void compute_averages(
    __global float* sum,             // [numSuperpixels]
    __global const int* count,       // [numSuperpixels]
    __global float* averages,        // [numSuperpixels]
    int numSuperpixels)
{
    int idx = get_global_id(0);
    if (idx >= numSuperpixels) return;
    if (count[idx] > 0) {
        averages[idx] = sum[idx] / count[idx];
    }
}
)";

const char* kernelSource8 = R"(
__kernel void assign_superpixel_average(
    __global const int* labels,      // [width*height]
    __global const float* averages,  // [numSuperpixels]
    __global float* out_image,       // [width*height]
    int numPixels)
{
    int idx = get_global_id(0);
    if (idx >= numPixels) return;
    int label = labels[idx];
    out_image[idx] = averages[label];
}
)";

const char* kernelSource9 = R"(
__kernel void superpixel_efficient_average2(
    __global const int* labels,
    __global const float* image1,
    __global const float* image2,
    __global float* sum1,
    __global float* sum2,
    __global int* count1,
    __global int* count2,
    __local float* local_sums1,
    __local float* local_sums2,
    __local int* local_counts1,
    __local int* local_counts2,
    int numSuperpixels,
    int numPixels)
{
    // Get indices
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);
    
    // Initialize local memory
    for (int i = lid; i < numSuperpixels; i += local_size) {
        local_sums1[i] = 0.0f;
        local_sums2[i] = 0.0f;
        local_counts1[i] = 0;
        local_counts2[i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // First phase: Accumulate in local memory
    if (gid < numPixels) {
        int label = labels[gid];
        float value1 = image1[gid];
        float value2 = image2[gid];
        
        // Using atomic operations on local memory
        atomic_add_float_local(&local_sums1[label], value1);
        atomic_add_float_local(&local_sums2[label], value2);
        atomic_inc(&local_counts1[label]);
        atomic_inc(&local_counts2[label]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Second phase: Accumulate from local to global
    for (int label = lid; label < numSuperpixels; label += local_size) {
        if (local_counts1[label] > 0) {
            atomic_add_float(&sum1[label], local_sums1[label]);
            atomic_add_float(&sum2[label], local_sums2[label]);
            atomic_add(&count1[label], local_counts1[label]);
            atomic_add(&count2[label], local_counts2[label]);
        }
    }
}

__kernel void compute_averages2(
    __global float* sum1,
    __global float* sum2,
    __global const int* count1,
    __global const int* count2,
    __global float* averages1,
    __global float* averages2,
    int numSuperpixels)
{
    int idx = get_global_id(0);
    if (idx >= numSuperpixels) return;
    if (count1[idx] > 0) {
        averages1[idx] = sum1[idx] / count1[idx];
        averages2[idx] = sum2[idx] / count2[idx];
    }
}

__kernel void assign_superpixel_average2(
    __global const int* labels,
    __global const float* averages1,
    __global const float* averages2,
    __global float* out_image1,
    __global float* out_image2,
    int numPixels)
{
    int idx = get_global_id(0);
    if (idx >= numPixels) return;
    int label = labels[idx];
    out_image1[idx] = averages1[label];
    out_image2[idx] = averages2[label];
}
)";

// Helper function to load OpenCL kernel from string
std::string SLICSuperpixels::loadKernelSource() {
    std::string source;
    source += kernelSource1;
    source += kernelSource2;
    source += kernelSource3;
    source += kernelSource4;
    source += kernelSource5;
    source += kernelSource6;
    source += kernelSource7;
    source += kernelSource8;
    source += kernelSource9;
    return source;
}

// Initialize OpenCL
void SLICSuperpixels::initOpenCL() {
    cl_int err;
    std::string kernelSource = loadKernelSource();
    const char* source = kernelSource.c_str();
    size_t sourceSize = kernelSource.length();
    program = clCreateProgramWithSource(context, 1, &source, &sourceSize, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create program");
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "Build log: " << log.data() << std::endl;
        throw std::runtime_error("Failed to build program");
    }
    initKernel = clCreateKernel(program, "initClusters", &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create init kernel");
    assignKernel = clCreateKernel(program, "assignPixels", &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create assign kernel");
    updateKernel = clCreateKernel(program, "updateClusters", &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create update kernel");
    accumulateKernel = clCreateKernel(program, "accumulateClusters", &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create accumulate kernel");
    finalizeKernel = clCreateKernel(program, "finalizeClusters", &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create finalize kernel");
    
    // Create kernels for single map case
    averageKernel = clCreateKernel(program, "superpixel_efficient_average", &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create average kernel");
    computeKernel = clCreateKernel(program, "compute_averages", &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create compute kernel");
    assignAvgKernel = clCreateKernel(program, "assign_superpixel_average", &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create assign average kernel");
    
    // Create kernels for two map case
    averageKernel2 = clCreateKernel(program, "superpixel_efficient_average2", &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create average2 kernel");
    computeKernel2 = clCreateKernel(program, "compute_averages2", &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create compute2 kernel");
    assignAvgKernel2 = clCreateKernel(program, "assign_superpixel_average2", &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create assign average2 kernel");
}

// Clean up OpenCL resources
void SLICSuperpixels::CleanupOpenCL() {
    if (updateKernel) clReleaseKernel(updateKernel);
    if (assignKernel) clReleaseKernel(assignKernel);
    if (initKernel) clReleaseKernel(initKernel);
    if (finalizeKernel) clReleaseKernel(finalizeKernel);
    if (accumulateKernel) clReleaseKernel(accumulateKernel);
    // Release single map kernels
    if (averageKernel) clReleaseKernel(averageKernel);
    if (computeKernel) clReleaseKernel(computeKernel);
    if (assignAvgKernel) clReleaseKernel(assignAvgKernel);
    // Release two-map kernels
    if (averageKernel2) clReleaseKernel(averageKernel2);
    if (computeKernel2) clReleaseKernel(computeKernel2);
    if (assignAvgKernel2) clReleaseKernel(assignAvgKernel2);
    if (program) clReleaseProgram(program);
}

SLICSuperpixels::SLICSuperpixels(cl_context context, cl_command_queue queue, cl_device_id device, float compactness, int numSuperpixels, int iterations)
    : context(context), queue(queue), device(device), compactness(compactness), numSuperpixels(numSuperpixels), iterations(iterations) {
    initOpenCL();
}

SLICSuperpixels::~SLICSuperpixels() {
    releaseBuffers();
    CleanupOpenCL();
}

// Allocate all needed buffers except d_image
void SLICSuperpixels::allocateBuffers(int width_, int height_, bool processTwoMaps_) {
    width = width_;
    height = height_;
    regionSize = static_cast<int>(std::round(std::sqrt((width * height) / (float)numSuperpixels)));
    globalSize = numSuperpixels;
    numPixels = width * height;
    globalSize2[0] = static_cast<size_t>(width);
    globalSize2[1] = static_cast<size_t>(height);
    globalSize3 = width * height;
    processTwoMaps = processTwoMaps_;

    cl_int err;
    d_clusters = clCreateBuffer(context, CL_MEM_READ_WRITE,
                               numSuperpixels * 5 * sizeof(float), nullptr, &err);
    d_accum = clCreateBuffer(context, CL_MEM_READ_WRITE, numSuperpixels * 5 * sizeof(int), nullptr, &err);
    d_counts = clCreateBuffer(context, CL_MEM_READ_WRITE, numSuperpixels * sizeof(int), nullptr, &err);
    
    if (processTwoMaps) {   
        // Buffers for two-map case
        d_value_sum_per_superpixel1 = clCreateBuffer(context, CL_MEM_READ_WRITE, numSuperpixels * sizeof(float), nullptr, &err);
        d_value_sum_per_superpixel2 = clCreateBuffer(context, CL_MEM_READ_WRITE, numSuperpixels * sizeof(float), nullptr, &err);
        d_pixel_counts_per_superpixel1 = clCreateBuffer(context, CL_MEM_READ_WRITE, numSuperpixels * sizeof(int), nullptr, &err);
        d_pixel_counts_per_superpixel2 = clCreateBuffer(context, CL_MEM_READ_WRITE, numSuperpixels * sizeof(int), nullptr, &err);
        d_averages1 = clCreateBuffer(context, CL_MEM_READ_WRITE, numSuperpixels * sizeof(float), nullptr, &err);
        d_averages2 = clCreateBuffer(context, CL_MEM_READ_WRITE, numSuperpixels * sizeof(float), nullptr, &err);
    } else {
        // Buffers for single map case
        d_pixel_counts_per_superpixel = clCreateBuffer(context, CL_MEM_READ_WRITE, numPixels * sizeof(int), nullptr, &err);
        d_value_sum_per_superpixel = clCreateBuffer(context, CL_MEM_READ_WRITE, numSuperpixels * sizeof(float), nullptr, &err);
        d_averages = clCreateBuffer(context, CL_MEM_READ_WRITE, numSuperpixels * sizeof(float), nullptr, &err);
    }
    
    // Zero initialize all buffers
    std::vector<int> zeroAccum(numSuperpixels * 5, 0);
    std::vector<int> zeroCounts(numSuperpixels, 0);
    std::vector<int> zeroPixelCounts(numPixels, 0);
    std::vector<float> zeroValueSum(numSuperpixels, 0.0f);
    std::vector<float> zeroAverages(numSuperpixels, 0.0f);
    
    err = clEnqueueWriteBuffer(queue, d_accum, CL_TRUE, 0, numSuperpixels * 5 * sizeof(int), zeroAccum.data(), 0, nullptr, nullptr);
    err |= clEnqueueWriteBuffer(queue, d_counts, CL_TRUE, 0, numSuperpixels * sizeof(int), zeroCounts.data(), 0, nullptr, nullptr);
    
    if (processTwoMaps) {
        // Initialize two-map buffers
        err |= clEnqueueWriteBuffer(queue, d_value_sum_per_superpixel1, CL_TRUE, 0, numSuperpixels * sizeof(float), zeroValueSum.data(), 0, nullptr, nullptr);
        err |= clEnqueueWriteBuffer(queue, d_value_sum_per_superpixel2, CL_TRUE, 0, numSuperpixels * sizeof(float), zeroValueSum.data(), 0, nullptr, nullptr);
        err |= clEnqueueWriteBuffer(queue, d_pixel_counts_per_superpixel1, CL_TRUE, 0, numSuperpixels * sizeof(int), zeroCounts.data(), 0, nullptr, nullptr);
        err |= clEnqueueWriteBuffer(queue, d_pixel_counts_per_superpixel2, CL_TRUE, 0, numSuperpixels * sizeof(int), zeroCounts.data(), 0, nullptr, nullptr);
        err |= clEnqueueWriteBuffer(queue, d_averages1, CL_TRUE, 0, numSuperpixels * sizeof(float), zeroAverages.data(), 0, nullptr, nullptr);
        err |= clEnqueueWriteBuffer(queue, d_averages2, CL_TRUE, 0, numSuperpixels * sizeof(float), zeroAverages.data(), 0, nullptr, nullptr);
    } else {
        // Initialize single map buffers
        err |= clEnqueueWriteBuffer(queue, d_pixel_counts_per_superpixel, CL_TRUE, 0, numPixels * sizeof(int), zeroPixelCounts.data(), 0, nullptr, nullptr);
        err |= clEnqueueWriteBuffer(queue, d_value_sum_per_superpixel, CL_TRUE, 0, numSuperpixels * sizeof(float), zeroValueSum.data(), 0, nullptr, nullptr);
        err |= clEnqueueWriteBuffer(queue, d_averages, CL_TRUE, 0, numSuperpixels * sizeof(float), zeroAverages.data(), 0, nullptr, nullptr);
    }
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to initialize buffers");
    }
}

// Set all kernel arguments, using external d_image
void SLICSuperpixels::setKernelArgs(cl_mem d_image, cl_mem d_labels, cl_mem d_opticalFlowMap1, cl_mem d_opticalFlowMap2, cl_mem d_avgImage1, cl_mem d_avgImage2) {
    cl_int err = 0;
    // initKernel
    err = clSetKernelArg(initKernel, 0, sizeof(cl_mem), &d_image);
    err |= clSetKernelArg(initKernel, 1, sizeof(cl_mem), &d_clusters);
    err |= clSetKernelArg(initKernel, 2, sizeof(int), &width);
    err |= clSetKernelArg(initKernel, 3, sizeof(int), &height);
    err |= clSetKernelArg(initKernel, 4, sizeof(int), &numSuperpixels);
    // assignKernel
    err = clSetKernelArg(assignKernel, 0, sizeof(cl_mem), &d_image);
    err |= clSetKernelArg(assignKernel, 1, sizeof(cl_mem), &d_clusters);
    err |= clSetKernelArg(assignKernel, 2, sizeof(cl_mem), &d_labels);
    err |= clSetKernelArg(assignKernel, 3, sizeof(int), &width);
    err |= clSetKernelArg(assignKernel, 4, sizeof(int), &height);
    err |= clSetKernelArg(assignKernel, 5, sizeof(int), &numSuperpixels);
    err |= clSetKernelArg(assignKernel, 6, sizeof(int), &regionSize);
    err |= clSetKernelArg(assignKernel, 7, sizeof(float), &compactness);
    // accumulateKernel
    err = clSetKernelArg(accumulateKernel, 0, sizeof(cl_mem), &d_image);
    err |= clSetKernelArg(accumulateKernel, 1, sizeof(cl_mem), &d_labels);
    err |= clSetKernelArg(accumulateKernel, 2, sizeof(cl_mem), &d_accum);
    err |= clSetKernelArg(accumulateKernel, 3, sizeof(cl_mem), &d_counts);
    err |= clSetKernelArg(accumulateKernel, 4, sizeof(int), &width);
    err |= clSetKernelArg(accumulateKernel, 5, sizeof(int), &height);
    // finalizeKernel
    err = clSetKernelArg(finalizeKernel, 0, sizeof(cl_mem), &d_accum);
    err |= clSetKernelArg(finalizeKernel, 1, sizeof(cl_mem), &d_counts);
    err |= clSetKernelArg(finalizeKernel, 2, sizeof(cl_mem), &d_clusters);
    err |= clSetKernelArg(finalizeKernel, 3, sizeof(int), &numSuperpixels);

    if (processTwoMaps) {
        // averageKernel2
        size_t localMemSize1 = numSuperpixels * sizeof(float);
        size_t localMemSize2 = numSuperpixels * sizeof(float);
        size_t localCountSize1 = numSuperpixels * sizeof(int);
        size_t localCountSize2 = numSuperpixels * sizeof(int);
        err = clSetKernelArg(averageKernel2, 0, sizeof(cl_mem), &d_labels);
        err |= clSetKernelArg(averageKernel2, 1, sizeof(cl_mem), &d_opticalFlowMap1);
        err |= clSetKernelArg(averageKernel2, 2, sizeof(cl_mem), &d_opticalFlowMap2);
        err |= clSetKernelArg(averageKernel2, 3, sizeof(cl_mem), &d_value_sum_per_superpixel1);
        err |= clSetKernelArg(averageKernel2, 4, sizeof(cl_mem), &d_value_sum_per_superpixel2);
        err |= clSetKernelArg(averageKernel2, 5, sizeof(cl_mem), &d_pixel_counts_per_superpixel1);
        err |= clSetKernelArg(averageKernel2, 6, sizeof(cl_mem), &d_pixel_counts_per_superpixel2);
        err |= clSetKernelArg(averageKernel2, 7, localMemSize1, nullptr);  // Local memory for sums1
        err |= clSetKernelArg(averageKernel2, 8, localMemSize2, nullptr);  // Local memory for sums2
        err |= clSetKernelArg(averageKernel2, 9, localCountSize1, nullptr); // Local memory for counts1
        err |= clSetKernelArg(averageKernel2, 10, localCountSize2, nullptr); // Local memory for counts2
        err |= clSetKernelArg(averageKernel2, 11, sizeof(int), &numSuperpixels);
        err |= clSetKernelArg(averageKernel2, 12, sizeof(int), &numPixels);
        // computeKernel2
        err |= clSetKernelArg(computeKernel2, 0, sizeof(cl_mem), &d_value_sum_per_superpixel1);
        err |= clSetKernelArg(computeKernel2, 1, sizeof(cl_mem), &d_value_sum_per_superpixel2);
        err |= clSetKernelArg(computeKernel2, 2, sizeof(cl_mem), &d_pixel_counts_per_superpixel1);
        err |= clSetKernelArg(computeKernel2, 3, sizeof(cl_mem), &d_pixel_counts_per_superpixel2);
        err |= clSetKernelArg(computeKernel2, 4, sizeof(cl_mem), &d_averages1);
        err |= clSetKernelArg(computeKernel2, 5, sizeof(cl_mem), &d_averages2);
        err |= clSetKernelArg(computeKernel2, 6, sizeof(int), &numSuperpixels);
        // assignAvgKernel2
        err |= clSetKernelArg(assignAvgKernel2, 0, sizeof(cl_mem), &d_labels);
        err |= clSetKernelArg(assignAvgKernel2, 1, sizeof(cl_mem), &d_averages1);
        err |= clSetKernelArg(assignAvgKernel2, 2, sizeof(cl_mem), &d_averages2);
        err |= clSetKernelArg(assignAvgKernel2, 3, sizeof(cl_mem), &d_avgImage1);
        err |= clSetKernelArg(assignAvgKernel2, 4, sizeof(cl_mem), &d_avgImage2);
        err |= clSetKernelArg(assignAvgKernel2, 5, sizeof(int), &numPixels);
    } else {
        // averageKernel
        size_t localMemSize = numSuperpixels * sizeof(float);
        size_t localCountSize = numSuperpixels * sizeof(int);
        err = clSetKernelArg(averageKernel, 0, sizeof(cl_mem), &d_labels);
        err |= clSetKernelArg(averageKernel, 1, sizeof(cl_mem), &d_opticalFlowMap1);
        err |= clSetKernelArg(averageKernel, 2, sizeof(cl_mem), &d_value_sum_per_superpixel);
        err |= clSetKernelArg(averageKernel, 3, sizeof(cl_mem), &d_pixel_counts_per_superpixel);
        err |= clSetKernelArg(averageKernel, 4, localMemSize, nullptr);  // Local memory for sums
        err |= clSetKernelArg(averageKernel, 5, localCountSize, nullptr); // Local memory for counts
        err |= clSetKernelArg(averageKernel, 6, sizeof(int), &numSuperpixels);
        err |= clSetKernelArg(averageKernel, 7, sizeof(int), &numPixels);
        // computeKernel
        err |= clSetKernelArg(computeKernel, 0, sizeof(cl_mem), &d_value_sum_per_superpixel);
        err |= clSetKernelArg(computeKernel, 1, sizeof(cl_mem), &d_pixel_counts_per_superpixel);
        err |= clSetKernelArg(computeKernel, 2, sizeof(cl_mem), &d_averages);
        err |= clSetKernelArg(computeKernel, 3, sizeof(int), &numSuperpixels);
        // assignAvgKernel
        err |= clSetKernelArg(assignAvgKernel, 0, sizeof(cl_mem), &d_labels);
        err |= clSetKernelArg(assignAvgKernel, 1, sizeof(cl_mem), &d_averages);
        err |= clSetKernelArg(assignAvgKernel, 2, sizeof(cl_mem), &d_avgImage1);
        err |= clSetKernelArg(assignAvgKernel, 3, sizeof(int), &numPixels);
    }

    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to set kernel arguments");
    }
}

// Release device buffers
void SLICSuperpixels::releaseBuffers() {
    if (d_clusters) clReleaseMemObject(d_clusters);
    if (d_accum) clReleaseMemObject(d_accum);
    if (d_counts) clReleaseMemObject(d_counts);
    // Release single map buffers
    if (d_pixel_counts_per_superpixel) clReleaseMemObject(d_pixel_counts_per_superpixel);
    if (d_value_sum_per_superpixel) clReleaseMemObject(d_value_sum_per_superpixel);
    if (d_averages) clReleaseMemObject(d_averages);
    // Release two-map buffers
    if (d_value_sum_per_superpixel1) clReleaseMemObject(d_value_sum_per_superpixel1);
    if (d_value_sum_per_superpixel2) clReleaseMemObject(d_value_sum_per_superpixel2);
    if (d_pixel_counts_per_superpixel1) clReleaseMemObject(d_pixel_counts_per_superpixel1);
    if (d_pixel_counts_per_superpixel2) clReleaseMemObject(d_pixel_counts_per_superpixel2);
    if (d_averages1) clReleaseMemObject(d_averages1);
    if (d_averages2) clReleaseMemObject(d_averages2);
    
    d_clusters = d_accum = d_counts = nullptr;
    d_pixel_counts_per_superpixel = d_value_sum_per_superpixel = d_averages = nullptr;
    d_value_sum_per_superpixel1 = d_value_sum_per_superpixel2 = nullptr;
    d_pixel_counts_per_superpixel1 = d_pixel_counts_per_superpixel2 = nullptr;
    d_averages1 = d_averages2 = nullptr;
}

// Only run kernels and read back results, using external d_image
void SLICSuperpixels::processImage() {
    cl_int err;
    clFinish(queue);
    // Execute initialization kernel
    err = clEnqueueNDRangeKernel(queue, initKernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
    clFinish(queue);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to execute init kernel");

    // assignPixels kernel
    err = clEnqueueNDRangeKernel(queue, assignKernel, 2, nullptr, globalSize2, nullptr, 0, nullptr, nullptr);
    clFinish(queue);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to execute assign kernel");
    
    // accumulateClusters kernel
    err = clEnqueueNDRangeKernel(queue, accumulateKernel, 1, nullptr, &globalSize3, nullptr, 0, nullptr, nullptr);
    clFinish(queue);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to execute accumulate kernel");

    // finalizeClusters kernel
    size_t clusterGlobalSize = numSuperpixels;
    err = clEnqueueNDRangeKernel(queue, finalizeKernel, 1, nullptr, &clusterGlobalSize, nullptr, 0, nullptr, nullptr);
    clFinish(queue);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to execute finalize kernel");

    // Launch accumulation kernel for averaging with appropriate work group size
    size_t localWorkSize = 256;  // Typical work group size
    size_t globalWorkSize = ((numPixels + localWorkSize - 1) / localWorkSize) * localWorkSize;
    err = clEnqueueNDRangeKernel(queue, averageKernel, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr);
    clFinish(queue);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to launch average kernel");

    // Launch compute_averages kernel
    size_t computeGlobalSize = numSuperpixels;
    err = clEnqueueNDRangeKernel(queue, computeKernel, 1, nullptr, &computeGlobalSize, nullptr, 0, nullptr, nullptr);
    clFinish(queue);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to launch compute_averages kernel");

    // Launch assign_superpixel_average kernel
    err = clEnqueueNDRangeKernel(queue, assignAvgKernel, 1, nullptr, &globalSize3, nullptr, 0, nullptr, nullptr);
    clFinish(queue);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to launch assign_superpixel_average kernel");
}

// Only run kernels and read back results, using external d_image
void SLICSuperpixels::processTwoImages() {
    cl_int err;
    clFinish(queue);
    // Execute initialization kernel
    err = clEnqueueNDRangeKernel(queue, initKernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
    clFinish(queue);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to execute init kernel");

    // assignPixels kernel
    err = clEnqueueNDRangeKernel(queue, assignKernel, 2, nullptr, globalSize2, nullptr, 0, nullptr, nullptr);
    clFinish(queue);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to execute assign kernel");
    
    // accumulateClusters kernel
    err = clEnqueueNDRangeKernel(queue, accumulateKernel, 1, nullptr, &globalSize3, nullptr, 0, nullptr, nullptr);
    clFinish(queue);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to execute accumulate kernel");

    // finalizeClusters kernel
    size_t clusterGlobalSize = numSuperpixels;
    err = clEnqueueNDRangeKernel(queue, finalizeKernel, 1, nullptr, &clusterGlobalSize, nullptr, 0, nullptr, nullptr);
    clFinish(queue);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to execute finalize kernel");

    // Launch accumulation kernel for averaging with appropriate work group size
    size_t localWorkSize = 256;  // Typical work group size
    size_t globalWorkSize = ((numPixels + localWorkSize - 1) / localWorkSize) * localWorkSize;
    err = clEnqueueNDRangeKernel(queue, averageKernel2, 1, nullptr, &globalWorkSize, &localWorkSize, 0, nullptr, nullptr);
    clFinish(queue);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to launch average2 kernel");

    // Launch compute_averages kernel
    size_t computeGlobalSize = numSuperpixels;
    err = clEnqueueNDRangeKernel(queue, computeKernel2, 1, nullptr, &computeGlobalSize, nullptr, 0, nullptr, nullptr);
    clFinish(queue);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to launch compute_averages2 kernel");

    // Launch assign_superpixel_average kernel
    err = clEnqueueNDRangeKernel(queue, assignAvgKernel2, 1, nullptr, &globalSize3, nullptr, 0, nullptr, nullptr);
    clFinish(queue);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to launch assign_superpixel_average2 kernel");
}

/*
int main(int argc, char** argv) {
    try {
        // 1. Load and prepare the image
        cv::Mat image;
        if (argc > 1) {
            image = cv::imread(argv[1]);
        }
        if (image.empty()) {
            std::cerr << "Failed to load image." << std::endl;
            return -1;
        }
        int width = image.cols;
        int height = image.rows;
        int numSuperpixels = 1000;
        float compactness = 36.0f;
        int iterations = 1;
        int regionSize = static_cast<int>(std::round(std::sqrt((width * height) / (float)numSuperpixels)));
        if (regionSize < 1) regionSize = 1;

        // 2. Create OpenCL context and command queue
        cl_int err;
        cl_platform_id platform;
        cl_device_id device;
        cl_context context;
        cl_command_queue queue;
        err = clGetPlatformIDs(1, &platform, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to get OpenCL platform");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to get OpenCL device");
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create OpenCL context");
        queue = clCreateCommandQueue(context, device, 0, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create OpenCL command queue");

        // 3. Initialize image buffers on GPU
        cv::Mat imageFloat, imagegray, opticalFlow;
        image.convertTo(imageFloat, CV_32FC3, 1.0 / 255.0);
        cv::cvtColor(image, imagegray, cv::COLOR_BGR2GRAY);
        imagegray.convertTo(opticalFlow, CV_32FC1);

        cl_mem d_image = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        width * height * 3 * sizeof(float), imageFloat.data, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create image buffer");
        cl_mem d_labels = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        width * height * sizeof(int), nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create labels buffer");
        cl_mem d_opticalFlowMap1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            width * height * sizeof(float), opticalFlow.data, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create optical flow map 1 buffer");
        cl_mem d_opticalFlowMap2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            width * height * sizeof(float), opticalFlow.data, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create optical flow map 2 buffer");
        cl_mem d_avgImage1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height * sizeof(float), nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create output buffer 1");
        cl_mem d_avgImage2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * height * sizeof(float), nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create output buffer 2");
        
        // Zero initialize the output buffer
        std::vector<float> zeroImage(width * height, 0.0f);
        err = clEnqueueWriteBuffer(queue, d_avgImage1, CL_TRUE, 0, width * height * sizeof(float), zeroImage.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to initialize output buffer 1");
        err = clEnqueueWriteBuffer(queue, d_avgImage2, CL_TRUE, 0, width * height * sizeof(float), zeroImage.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to initialize output buffer 2");

        clFinish(queue);

        auto start = std::chrono::high_resolution_clock::now();
        clFinish(queue);
        auto end = std::chrono::high_resolution_clock::now();
        double cl_ms = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
        std::cout << "clFinish took " << cl_ms << " ms." << std::endl;

        // 4. Create and configure SLIC object
        bool processTwoMaps = true;
        SLICSuperpixels slic(context, queue, device, compactness, numSuperpixels, iterations);
        slic.allocateBuffers(width, height, processTwoMaps);
        slic.setKernelArgs(d_image, d_labels, d_opticalFlowMap1, d_opticalFlowMap2, d_avgImage1, d_avgImage2);

        // 5. Run SLIC
        clFinish(queue);
        auto slic_start = std::chrono::high_resolution_clock::now();
        // slic.processImage(); // result saves in d_labels and d_avgImage1
        slic.processTwoImages(); // result saves in d_labels and d_avgImage1 and d_avgImage2
        clFinish(queue);
        auto slic_end = std::chrono::high_resolution_clock::now();
        double slic_ms = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(slic_end - slic_start).count());
        std::cout << "SLIC superpixel computation took " << slic_ms << " ms." << std::endl;

        // Read back labels results
        cv::Mat labels(height, width, CV_32S);
        err = clEnqueueReadBuffer(queue, d_labels, CL_TRUE, 0, width * height * sizeof(int), labels.data, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to read labels buffer");
        }

        // 6. Visualize or save results as needed
        cv::Mat result = image.clone();
        for (int y = 1; y < image.rows - 1; y++) {
            for (int x = 1; x < image.cols - 1; x++) {
                int label = labels.at<int>(y, x);
                if (labels.at<int>(y-1, x) != label || labels.at<int>(y+1, x) != label ||
                    labels.at<int>(y, x-1) != label || labels.at<int>(y, x+1) != label) {
                    result.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255); // Red boundary
                }
            }
        }
        cv::imshow("Original Image", image);
        cv::imshow("SLIC Superpixels", result);
        cv::imwrite("original.jpg", image);
        cv::imwrite("superpixels.jpg", result);

        // Read back averaged optical flow results
        cv::Mat avgImage1(height, width, CV_32FC1);
        cv::Mat avgImage2(height, width, CV_32FC1);
        err = clEnqueueReadBuffer(queue, d_avgImage1, CL_TRUE, 0, width * height * sizeof(float), avgImage1.data, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to read output buffer 1");
        err = clEnqueueReadBuffer(queue, d_avgImage2, CL_TRUE, 0, width * height * sizeof(float), avgImage2.data, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to read output buffer 2");

        // Convert to 8-bit by rounding
        cv::Mat avgImage8U1;
        cv::Mat avgImage8U2;
        avgImage1.convertTo(avgImage8U1, CV_8UC1);
        avgImage2.convertTo(avgImage8U2, CV_8UC1);
        
        // Show and save the 8-bit version
        cv::imshow("Superpixel Averaged Optical Flow 1", avgImage8U1);
        cv::imshow("Superpixel Averaged Optical Flow 2", avgImage8U2);
        cv::imwrite("superpixel_averaged_optical_flow_1.jpg", avgImage8U1);
        cv::imwrite("superpixel_averaged_optical_flow_2.jpg", avgImage8U2);

        // // Run OpenCV's SLIC implementation for comparison
        // cv::Mat labels_cv;
        // cv::Mat result_cv = image.clone();
        // cv::Mat avgImage_cv = cv::Mat::zeros(height, width, CV_32FC1);
        // std::vector<cv::Mat> superpixel_means;
        // cv::Mat avgImage8U_cv;
        // auto cv_slic_start = std::chrono::high_resolution_clock::now();
        // cv::Ptr<cv::ximgproc::SuperpixelSLIC> slic_cv = cv::ximgproc::createSuperpixelSLIC(image, cv::ximgproc::SLIC, regionSize, static_cast<float>(compactness*2.0));
        // slic_cv->iterate(iterations);
        // slic_cv->getLabels(labels_cv);
        // auto cv_slic_end = std::chrono::high_resolution_clock::now();
        // double cv_slic_ms = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(cv_slic_end - cv_slic_start).count());
        // std::cout << "OpenCV SLIC superpixel computation took " << cv_slic_ms << " ms." << std::endl;
        // // Visualize OpenCV SLIC result
        // for (int y = 1; y < image.rows - 1; y++) {
        //     for (int x = 1; x < image.cols - 1; x++) {
        //         int label = labels_cv.at<int>(y, x);
        //         if (labels_cv.at<int>(y-1, x) != label || labels_cv.at<int>(y+1, x) != label ||
        //             labels_cv.at<int>(y, x-1) != label || labels_cv.at<int>(y, x+1) != label) {
        //             result_cv.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255); // Green boundary
        //         }
        //     }
        // }
        // cv::imshow("OpenCV SLIC Superpixels", result_cv);
        // cv::imwrite("opencv_superpixels.jpg", result_cv);
        // auto cv_avg_start = std::chrono::high_resolution_clock::now();
        // // Average superpixels using OpenCV
        // cv::Mat result_avg_cv = image.clone();
        // slic_cv->getLabelContourMask(result_avg_cv, false);
        // slic_cv->enforceLabelConnectivity(10);
        // // Calculate average for each superpixel
        // int numLabels = slic_cv->getNumberOfSuperpixels();
        // for (int i = 0; i < numLabels; i++) {
        //     cv::Mat mask = (labels_cv == i);
        //     cv::Scalar mean = cv::mean(opticalFlow, mask);
        //     avgImage_cv.setTo(mean[0], mask);
        // }
        // // Convert to 8-bit for visualization
        // avgImage_cv.convertTo(avgImage8U_cv, CV_8UC1);
        // cv::imshow("OpenCV Superpixel Averaged Optical Flow", avgImage8U_cv);
        // cv::imwrite("opencv_superpixel_averaged_optical_flow.jpg", avgImage8U_cv);
        // auto cv_avg_end = std::chrono::high_resolution_clock::now();
        // double cv_avg_ms = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(cv_avg_end - cv_avg_start).count());
        // double cv_ms = cv_slic_ms + cv_avg_ms;
        // std::cout << "OpenCV SLIC superpixel computation and averaging took in total " << cv_ms << " ms." << std::endl;
        // std::cout << "OpenCL implementation is " << (cv_ms / slic_ms) << "x faster than OpenCV implementation." << std::endl;

        // Cleanup the image buffer and OpenCL resources
        clReleaseMemObject(d_opticalFlowMap1);
        clReleaseMemObject(d_opticalFlowMap2);
        clReleaseMemObject(d_avgImage1);
        clReleaseMemObject(d_avgImage2);
        clReleaseMemObject(d_image);
        clReleaseMemObject(d_labels);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        std::cout << "Results saved to 'original.jpg' and 'superpixels.jpg' and 'opencv_superpixels.jpg' and 'superpixel_averaged_optical_flow.jpg'" << std::endl;
        std::cout << "Press any key to exit..." << std::endl;
        cv::waitKey(0);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
*/
