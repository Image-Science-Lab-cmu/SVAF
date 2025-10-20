#ifndef WARP_PERSPECTIVE_H
#define WARP_PERSPECTIVE_H

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <stdexcept>

// OpenCL kernel source code
extern const char* warpPerspectiveKernel;

class WarpPerspective {
public:
    WarpPerspective(cl_context context_, cl_command_queue queue_, cl_device_id device_);
    ~WarpPerspective();
    void allocateBuffers(int src_width, int src_height, int dst_width, int dst_height);
    void setKernelArgs(const cl_mem clSrc, const cl_mem clDst, const cl_mem clM, float border_value);
    void releaseBuffers();
    void compute();

private:
    void buildKernels();
    cl_context context;
    cl_command_queue queue;
    cl_device_id device;
    cl_program program;
    cl_kernel kernel;
    size_t global_size[2];
    int src_width;
    int src_height;
    int dst_width;
    int dst_height;
};

#endif // WARP_PERSPECTIVE_H 