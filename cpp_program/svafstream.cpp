#include "stdafx.h"
#include "ArenaApi.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <windows.h>
#include <gl/gl.h>
#include <gl/glu.h>
#include <chrono>
#include <cmath>
#include <set>
#include <cnpy.h>
#include <iomanip>
#include <sstream>
#include <CL/cl.h>
#include <CL/cl_gl.h>
#include <map>
#include "warpPerspective.h"  // Add include for WarpPerspective class
#include "dpflow.h"
#include "slic.h"
#include "turbo_colormap_lut.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Maximum packet size
#define TAB1 "  "
#define MAX_PACKET_SIZE true

// SLM parameters
const int cameraWidth = 1224;
const int cameraTopMargin = 45;
const int cameraBottomMargin = 55;
const int cameraHeight = 1024;
const int cameraChannels = 3;
const int SLM_WIDTH = 4000;
const int SLM_HEIGHT = 2464;
const double LEFT_FREQ = 1.0/4.0;  // Left 2/3 of image
const double RIGHT_FREQ = -1.0/5.0; // Right 1/3 of image
const size_t globalCameraSize[2] = {static_cast<size_t>(cameraWidth), static_cast<size_t>(cameraHeight)};
const size_t globalSLMSize[2] = {static_cast<size_t>(SLM_WIDTH), static_cast<size_t>(SLM_HEIGHT)};

// DPflow and SLIC parameters
float flow_scalar_learning_rate = 0.1f;  // optical flow learning rate
int window_size = 202;
bool enableSuperpixels = false;
int numSuperpixels = 500;
float compactness = 13.0f;
float lambda_reg = 1e-6f;
bool green = false;
int devignet = 0;
int iterations = 1;

// Display mode flag: true = corner depth display, false = side-by-side display
bool cornerDepthDisplay = true;

// SVAF status: 1 = OFF (red), 2/3 = ON (green)
int svafStatus = 1;

// Pre-rendered text overlays
cv::Mat svafOffOverlay = cv::Mat::zeros(static_cast<int>(cameraHeight), static_cast<int>(cameraWidth), CV_8UC4);  // BGRA format
cv::Mat svafOnCenterOverlay = cv::Mat::zeros(static_cast<int>(cameraHeight), static_cast<int>(cameraWidth), CV_8UC4);
cv::Mat svafOnOverlay = cv::Mat::zeros(static_cast<int>(cameraHeight), static_cast<int>(cameraWidth), CV_8UC4);

// OpenCL buffers for text overlays (three separate buffers)
cl_mem g_clTextOverlayOff = nullptr;
cl_mem g_clTextOverlayOnCenter = nullptr;
cl_mem g_clTextOverlayOn = nullptr;

// Function to create text overlays at initialization
void CreateTextOverlays() {
    std::cout << "Creating SVAF status text overlays...\n";
    
    // Text properties
    int fontFace = cv::FONT_HERSHEY_PLAIN;
    double fontScale = 2;
    int thickness = 3;
    int baseline = 0;
    
    // Position text in bottom-left corner
    cv::Point textPosition(20, static_cast<int>(cameraHeight) - 66);
    
    // Create "SVAF OFF" overlay (red text)
    cv::Size textSize = cv::getTextSize("SVAF OFF", fontFace, fontScale, thickness, &baseline);
    cv::putText(svafOffOverlay, "SVAF OFF", textPosition, fontFace, fontScale, 
                cv::Scalar(255, 0, 0, 255), thickness);  // Red text
    
    // Create "SVAF ON (CENTER ONLY)" overlay (green text)
    textSize = cv::getTextSize("SVAF ON (CENTER ONLY)", fontFace, fontScale, thickness, &baseline);
    cv::putText(svafOnCenterOverlay, "SVAF ON (CENTER ONLY)", textPosition, fontFace, fontScale, cv::Scalar(0, 255, 0, 255), thickness);  // Green text
    
    // Create "SVAF ON" overlay (green text)
    textSize = cv::getTextSize("SVAF ON", fontFace, fontScale, thickness, &baseline);
    cv::putText(svafOnOverlay, "SVAF ON", textPosition, fontFace, fontScale, 
                cv::Scalar(0, 255, 0, 255), thickness);  // Green text
    
    // Flip all overlays horizontally and vertically to match display coordinate system
    cv::flip(svafOffOverlay, svafOffOverlay, -1);  // -1 flips both horizontally and vertically
    cv::flip(svafOnCenterOverlay, svafOnCenterOverlay, -1);
    cv::flip(svafOnOverlay, svafOnOverlay, -1);
    
    std::cout << "Text overlays created and flipped successfully\n";
    
}


// OpenCL variables
cl_context g_clContext = nullptr;  // Single context
cl_command_queue g_clQueue = nullptr;  // Single queue
cl_device_id g_clDevice = nullptr;
cl_program g_clProgram = nullptr;
cl_program g_clConvertToFloatProgram = nullptr;
cl_kernel g_clKernel = nullptr;
cl_kernel g_clPeriodMapKernel = nullptr;
cl_kernel g_clConvertLeftImageToFloatKernel = nullptr;
cl_kernel g_clConvertRightImageToFloatKernel = nullptr;
cl_kernel g_clConvertTopImageToFloatKernel = nullptr;
cl_kernel g_clConvertBottomImageToFloatKernel = nullptr;
cl_program g_clPeriodMapProgram = nullptr;
cl_mem g_clTempLeftBuffer = nullptr;
cl_mem g_clTempRightBuffer = nullptr;
cl_mem g_clTempTopBuffer = nullptr;
cl_mem g_clTempBottomBuffer = nullptr;
cl_mem g_clLeftImage = nullptr;
cl_mem g_clRightImage = nullptr;
cl_mem g_clTopImage = nullptr;
cl_mem g_clBottomImage = nullptr;
cl_mem g_clMask = nullptr;
cl_mem g_clFlow = nullptr;
cl_mem g_clLeftDevignetMap = nullptr;
cl_mem g_clRightDevignetMap = nullptr;
cl_mem g_clTopDevignetMap = nullptr;
cl_mem g_clBottomDevignetMap = nullptr;
cl_mem g_clSpatialFreqMap = nullptr;
cl_mem g_clLabels = nullptr;
cl_mem g_clSpatialFreqMapSuperpixels = nullptr;
cl_mem g_clWarpedFreqMap = nullptr;
cl_mem g_clHomography = nullptr;
cl_mem g_clPeriodTempMap = nullptr;
cl_mem g_clPeriodMap = nullptr;
cl_mem g_clYGrid = nullptr;
cl_mem g_clSharedTexture = nullptr;
void* hostPtr = nullptr;

// Add new OpenCL variables for flow multiplication
cl_kernel g_clUpdateFlowKernel = nullptr;
cl_program g_clUpdateFlowProgram = nullptr;

// Add WarpPerspective object as global variable
WarpPerspective* g_warp = nullptr;

// OpenGL variables
HWND g_SLMWnd = nullptr;
HWND g_capturedWnd = nullptr;
HDC g_sharedDC = nullptr;
HGLRC g_sharedRC = nullptr;
GLuint g_SLMTextureID = 0;
GLuint g_capturedTextureID = 0;
GLuint g_SLMPBOID = 0;
GLuint g_capturedPBOID = 0;

// Add new OpenCL variables for captured display
cl_kernel g_clCombineImagesKernel = nullptr;
cl_program g_clCombineImagesProgram = nullptr;
cl_mem g_clCombinedDisplay = nullptr;
cl_mem g_clTurboLUT = nullptr;
cl_mem g_clResizedDepthMap = nullptr;
cl_kernel g_clResizeDepthMapKernel = nullptr;
cl_program g_clResizeDepthMapProgram = nullptr;

// Text overlay blending
cl_kernel g_clBlendTextOverlayKernel = nullptr;
cl_program g_clBlendTextOverlayProgram = nullptr;

std::string g_devignet_folder = "C:/Users/matth/Desktop/SVAF/images/measurements/devignet_maps";

// Define OpenGL function pointer types
typedef void (APIENTRY *PFNGLBINDBUFFERPROC) (GLenum target, GLuint buffer);
typedef void (APIENTRY *PFNGLBUFFERDATAPROC) (GLenum target, GLsizeiptr size, const void* data, GLenum usage);
typedef void (APIENTRY *PFNGLGENBUFFERSPROC) (GLsizei n, GLuint* buffers);
typedef void (APIENTRY *PFNGLDELETEBUFFERSPROC) (GLsizei n, const GLuint* buffers);

// Create Y grid for SLM (constant)
cv::Mat CreateYGrid() {
    cv::Mat Y_grid(SLM_HEIGHT, SLM_WIDTH, CV_32F);  // Changed to float (CV_32F)
    for (int y = 0; y < SLM_HEIGHT; y++) {
        for (int x = 0; x < SLM_WIDTH; x++) {
            Y_grid.at<float>(y, x) = static_cast<float>(y + 1.0f);  // Cast to float
        }
    }
    return Y_grid;
}

const cv::Mat Y_GRID = CreateYGrid();

// Add newOpenCL kernel for converting uchar to float and applying devignet map
const char* convertToFloatKernel = R"(
__kernel void convertToFloat(
    __global const uchar* input,
    __global float* output,
    __global const float* devignetMap,
    const int devignet, 
    const int width,
    const int height,
    const int channels)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    int idx = (y * width + x) * channels;
    int devignetIdx = y * width + x;
    for (int c = 0; c < channels; c++) {
        if (devignet) { 
            output[idx + c] = (float)input[idx + c] / 255.0f * devignetMap[devignetIdx];
        } else {
            output[idx + c] = (float)input[idx + c] / 255.0f;
        }
    }
}
)";

// Add new OpenCL kernel for computing SLM pattern
const char* kernelSource = R"(
__kernel void computeSLMPattern(
    __global const float* periodMap,
    __global const float* yGrid,
    __global uchar* output,  // Explicitly unsigned char
    const int width,
    const int height,
    const int flag,
    const float depth)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float period = periodMap[idx]; //round(periodMap[idx]);

    if (flag == 1) {
        period = 0.3*sin(2*depth*3.14159);
    }
    if (flag == 2) {
        if (x >= width/2 - 400 && x < width/2 + 400 && y >= height/2 - 400 && y < height/2 + 400) {
            period = 0.3*sin(2*depth*3.14159) + 0.3;
        } else {
            period = 0;
        }
    }
    
    /*if (period == 0.0f) {
        output[idx] = 0;
        return;
    }
    
    float phase = (period > 0.0f ? yGrid[idx] : -yGrid[idx]);
    float absPeriod = fabs(period);
    float modResult = fmod(phase, absPeriod);
    
    if (fmod(absPeriod, 2.0f) == 0.0f) {
        modResult = fmod(modResult + absPeriod, absPeriod);
    } else if (modResult < 0.0f) {
        modResult += absPeriod;
    }*/
    
    uchar tmp = (uchar) fmod(fabs(yGrid[idx] * 255.0f * period), 255.0f);

    //int iperiod = ceil(1 / fabs(period));

    //uchar tmp = (uchar) 255.0f * fabs(period) * (yGrid[idx] - iperiod * floor(yGrid[idx] / iperiod));

    output[idx] = (period >= 0 ? tmp : 255-tmp); //(uchar)((modResult / (absPeriod - 1.0f)) * 255.0f);  // 0 MATT's EDIT
}
)";

// Add new OpenCL kernel for computing spatial period map
const char* spatialPeriodMapKernel = R"(
__kernel void computeSpatialPeriodMap(
    __global const float* spatialFreqMap,
    __global float* periodMap,
    const int width,
    const int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float freq = spatialFreqMap[idx];
    
    if (freq != 0.0) {
        periodMap[idx] = freq; //1.0 / freq; //round(1.0 / freq);
    } else {
        periodMap[idx] = 0.0;
    }
}
)";

// Add new OpenCL kernel for multiplying flow with scalar
const char* updateFlowKernel = R"(
__kernel void updateFlowWithScalar(
    __global const float* flow,
    __global float* spatialFreqMap,  // Changed to float to match flow type
    const float scalar,              // Changed to float
    const int width,
    const int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float freq = spatialFreqMap[idx] + flow[idx] * 0.015;

    // Clip freq between -1/4 and 1/4
    if (freq < -0.4f) {
        freq = -0.4f;
    } else if (freq > 0.4f) {
        freq = 0.4f;
    }

    // Compute period map as round of 1/spatialFreqMap[idx]
    /*float period = 0.0f;
    if (freq != 0.0f) {
        period = round(1.0f / freq);
        spatialFreqMap[idx] = 1.0f / period;
    } else {
        spatialFreqMap[idx] = 0.0f;
    }*/

    spatialFreqMap[idx] = freq;
}
)";

// Add new OpenCL kernel for resizing depth map
const char* resizeDepthMapKernel = R"(
__kernel void resizeDepthMap(
    __global const float* inputDepthMap,
    __global float* outputDepthMap,
    const int inputWidth,
    const int inputHeight,
    const int outputWidth,
    const int outputHeight)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= outputWidth || y >= outputHeight) return;
    
    // Calculate scaling factors
    float scaleX = (float)inputWidth / (float)outputWidth;
    float scaleY = (float)inputHeight / (float)outputHeight;
    
    // Map output coordinates to input coordinates
    int inputX = (int)(x * scaleX);
    int inputY = (int)(y * scaleY);
    
    // Clamp to valid range
    inputX = max(0, min(inputWidth - 1, inputX));
    inputY = max(0, min(inputHeight - 1, inputY));
    
    // Copy depth value
    int inputIdx = inputY * inputWidth + inputX;
    int outputIdx = y * outputWidth + x;
    outputDepthMap[outputIdx] = inputDepthMap[inputIdx];
}
)";

// Add new OpenCL kernel for combining images and applying turbo colormap
const char* combineImagesKernel = R"(
__kernel void combineImagesWithColormap(
    __global const float* topImage,
    __global const float* bottomImage,
    __global const float* leftImage,
    __global const float* rightImage,
    __global const float* spatialFreqMap,
    __global const float* resizedDepthMap,
    __global uchar* combinedDisplay,
    __constant float* turbo,
    const int cameraTopMargin, 
    const int cameraBottomMargin, 
    const int width,
    const int height,
    const int mode,
    const int cornerDisplay)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    if (y < cameraTopMargin || y >= height - cameraBottomMargin) return;
    
    int idx = y * width + x;
    int combinedIdx, rightIdx;
    
    // Calculate indices based on display mode
    if (cornerDisplay) {
        // Corner display mode: RGB fills entire image, depth in top-right corner
        combinedIdx = y * width + x;
        rightIdx = combinedIdx; // Same position for RGB
    } else {
        // Side-by-side mode: RGB on right, depth on left
        combinedIdx = y * (width * 2) + x;
        rightIdx = combinedIdx + width;
    }

    float r, g, b;

    // Get RGB values from total image
    if (mode == 0) {
        r = 2*(topImage[idx * 3 + 0] + bottomImage[idx * 3 + 2] + leftImage[idx * 3 + 0] + rightImage[idx * 3 + 0])/4;
        g = 2*(topImage[idx * 3 + 1] + bottomImage[idx * 3 + 2] + leftImage[idx * 3 + 1] + rightImage[idx * 3 + 1])/4;
        b = 2*(topImage[idx * 3 + 2] + bottomImage[idx * 3 + 2] + leftImage[idx * 3 + 2] + rightImage[idx * 3 + 2])/4;
    } else if (mode == 1) {
        float tmp = leftImage[idx * 3 + 1] - rightImage[idx * 3 + 1];
        float tmp2 = leftImage[idx * 3 + 1] + rightImage[idx * 3 + 1];
        tmp = 0.5*tmp/tmp2 + 0.5;

        r = tmp;
        g = tmp;
        b = tmp;
    }
    
    // Ensure values are in valid range
    r = max(0.0f, min(1000.0f, r));
    g = max(0.0f, min(1000.0f, g));
    b = max(0.0f, min(1000.0f, b));
    
    // Apply contrast enhancement (stretch the dynamic range)
    float contrast_factor = 1.5f;  // Increase contrast
    r = (r - 0.5f) * contrast_factor + 0.5f;
    g = (g - 0.5f) * contrast_factor + 0.5f;
    b = (b - 0.5f) * contrast_factor + 0.5f;
    
    // Apply gamma correction for better visual appearance
    r = 0.45f * pow(r, 1/1.11f);  // Increased from 0.4f and 1/1.3f
    g = 0.45f * pow(g, 1/1.11f);
    b = 0.45f * pow(b, 1/1.11f);

    // Apply saturation enhancement
    float luminance = 0.299f * r + 0.587f * g + 0.114f * b;  // Standard luminance weights
    float saturation_factor = 1.08f;  // Increase saturation
    r = luminance + (r - luminance) * saturation_factor;
    g = luminance + (g - luminance) * saturation_factor;
    b = luminance + (b - luminance) * saturation_factor;
    
    // Apply vibrance enhancement (selective saturation boost)
    float vibrance_factor = 1.2f;
    float max_channel = fmax(fmax(r, g), b);
    float min_channel = fmin(fmin(r, g), b);
    float saturation = max_channel - min_channel;
    
    if (saturation > 0.1f) {  // Only enhance if there's significant color
        float vibrance_boost = (1.0f - saturation) * vibrance_factor;
        r = r + (r - luminance) * vibrance_boost;
        g = g + (g - luminance) * vibrance_boost;
        b = b + (b - luminance) * vibrance_boost;
    }

    // Clamp values to valid range
    r = max(0.0f, min(1.0f, r));
    g = max(0.0f, min(1.0f, g));
    b = max(0.0f, min(1.0f, b));
    
    // Get spatial frequency value
    float freq = spatialFreqMap[idx];
    
    // Normalize to [-0.5, 0.5] range (assuming typical flow values)
    float min_freq = -0.4f;
    float max_freq = 0.4f;
    float normalized = max(min_freq, min(max_freq, freq));
    normalized = (normalized - min_freq) / (max_freq - min_freq); // Map to [0, 1]

    // Apply turbo colormap
    int turbo_idx = (int)(max(0.0f, min(1.0f, normalized)) * 255.0f);
    float r_depth = turbo[3 * turbo_idx + 0];
    float g_depth = turbo[3 * turbo_idx + 1];
    float b_depth = turbo[3 * turbo_idx + 2];
    
    if (cornerDisplay) {
        // Corner display mode: RGB fills entire image, depth in top-right corner
        // Note: Image is flipped both vertically and horizontally during rendering
        // So we position corner in bottom-left of kernel coordinates to appear top-right after flip
        int cornerWidth = width / 5;
        int cornerHeight = height / 5;
        int cornerStartX = 0;  // Left side in kernel (appears right after flip)
        int cornerStartY = height - cornerHeight - cameraBottomMargin;  // Bottom side in kernel (appears top after flip)
        
        // Check if current pixel is in the corner region
        if (x >= cornerStartX && x < cornerWidth && y >= cornerStartY && y < cornerStartY + cornerHeight) {
            // In corner region - display resized depth map
            int cornerX = x - cornerStartX;
            int cornerY = y - cornerStartY;
            int resizedIdx = cornerY * cornerWidth + cornerX;
            float resized_freq = resizedDepthMap[resizedIdx];
            
            // Normalize to [-0.5, 0.5] range (assuming typical flow values)
            float min_freq = -0.4f;
            float max_freq = 0.4f;
            float normalized = max(min_freq, min(max_freq, resized_freq));
            normalized = (normalized - min_freq) / (max_freq - min_freq); // Map to [0, 1]

            // Apply turbo colormap to resized depth
            int turbo_idx = (int)(max(0.0f, min(1.0f, normalized)) * 255.0f);
            float r_resized_depth = turbo[3 * turbo_idx + 0];
            float g_resized_depth = turbo[3 * turbo_idx + 1];
            float b_resized_depth = turbo[3 * turbo_idx + 2];
            
            combinedDisplay[combinedIdx * 3] = (uchar)(r_resized_depth * 255.0f);     // R
            combinedDisplay[combinedIdx * 3 + 1] = (uchar)(g_resized_depth * 255.0f); // G
            combinedDisplay[combinedIdx * 3 + 2] = (uchar)(b_resized_depth * 255.0f); // B
        } else {
            // Outside corner region - display RGB
            combinedDisplay[combinedIdx * 3] = (uchar)(r * 255.0f);       // R
            combinedDisplay[combinedIdx * 3 + 1] = (uchar)(g * 255.0f);   // G
            combinedDisplay[combinedIdx * 3 + 2] = (uchar)(b * 255.0f);   // B
        }
    } else {
        // Side-by-side mode: RGB on right, depth on left
        // Copy RGB to right half
        combinedDisplay[rightIdx * 3] = (uchar)(r * 255.0f);       // R
        combinedDisplay[rightIdx * 3 + 1] = (uchar)(g * 255.0f);   // G
        combinedDisplay[rightIdx * 3 + 2] = (uchar)(b * 255.0f);   // B
        
        // Write depth to left half
    combinedDisplay[combinedIdx * 3] = (uchar)(r_depth * 255.0f);     // R
    combinedDisplay[combinedIdx * 3 + 1] = (uchar)(g_depth * 255.0f); // G
    combinedDisplay[combinedIdx * 3 + 2] = (uchar)(b_depth * 255.0f); // B
    }
    
}
)";

// OpenCL kernel for blending text overlay
const char* blendTextOverlayKernel = R"(
__kernel void blendTextOverlay(
    __global uchar* combinedDisplay,
    __global const uchar* textOverlay,
    const int width,
    const int height)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    int overlayIdx = idx * 4;  // BGRA format
    int displayIdx = idx * 3;  // BGR format
    
    // Get overlay pixel (BGRA)
    uchar overlayB = textOverlay[overlayIdx];
    uchar overlayG = textOverlay[overlayIdx + 1];
    uchar overlayR = textOverlay[overlayIdx + 2];
    uchar overlayA = textOverlay[overlayIdx + 3];
    
    // Only blend if overlay has alpha > 0
    if (overlayA > 0) {
        float alpha = overlayA / 255.0f;
        
        // Get current display pixel (BGR)
        uchar displayB = combinedDisplay[displayIdx];
        uchar displayG = combinedDisplay[displayIdx + 1];
        uchar displayR = combinedDisplay[displayIdx + 2];
        
        // Blend: result = background * (1 - alpha) + foreground * alpha
        combinedDisplay[displayIdx] = (uchar)(displayB * (1.0f - alpha) + overlayB * alpha);
        combinedDisplay[displayIdx + 1] = (uchar)(displayG * (1.0f - alpha) + overlayG * alpha);
        combinedDisplay[displayIdx + 2] = (uchar)(displayR * (1.0f - alpha) + overlayR * alpha);
    }
}
)";

// Configure for high frame rates
// (1) maximizes packet size
// (2) sets large number of buffers
void SetUpForRapidAcquisition(Arena::IDevice* pDevice)
{
	int64_t deviceStreamChannelPacketSizeInitial;
	if (MAX_PACKET_SIZE)
	{
		deviceStreamChannelPacketSizeInitial = Arena::GetNodeValue<int64_t>(pDevice->GetNodeMap(), "DeviceStreamChannelPacketSize");
	}

	// Set maximum stream channel packet size
	//    Maximizing packet size increases frame rate by reducing the amount of
	//    overhead required between images. This includes both extra
	//    header/trailer data per packet as well as extra time from intra-packet
	//    spacing (the time between packets). In order to grab images at the
	//    maximum packet size, the Ethernet adapter must be configured
	//    appropriately: 'Jumbo packet' must be set to its maximum, 'UDP checksum
	//    offload' must be set to 'Rx & Tx Enabled', and 'Received Buffers' must
	//    be set to its maximum.
	if (MAX_PACKET_SIZE)
	{
		std::cout << TAB1 << "Set maximum device stream channel packet size";

		GenApi::CIntegerPtr pDeviceStreamChannelPacketSize = pDevice->GetNodeMap()->GetNode("DeviceStreamChannelPacketSize");
		if (!pDeviceStreamChannelPacketSize || !GenApi::IsReadable(pDeviceStreamChannelPacketSize) || !GenApi::IsWritable(pDeviceStreamChannelPacketSize))
		{
			throw GenICam::GenericException("DeviceStreamChannelPacketSize node not found/readable/writable", __FILE__, __LINE__);
		}

		std::cout << " (" << pDeviceStreamChannelPacketSize->GetMax() << " " << pDeviceStreamChannelPacketSize->GetUnit() << ")\n";

		pDeviceStreamChannelPacketSize->SetValue(pDeviceStreamChannelPacketSize->GetMax());
	}

	if (!MAX_PACKET_SIZE)
	{
		// enable stream auto negotiate packet size
		Arena::SetNodeValue<bool>(pDevice->GetTLStreamNodeMap(), "StreamAutoNegotiatePacketSize", true);
	}

	// enable stream packet resend
	Arena::SetNodeValue<bool>(pDevice->GetTLStreamNodeMap(), "StreamPacketResendEnable", true);

	if (MAX_PACKET_SIZE)
	{
		Arena::SetNodeValue<int64_t>(pDevice->GetNodeMap(), "DeviceStreamChannelPacketSize", deviceStreamChannelPacketSizeInitial);
	}
}

// Function to read and load image files
void LoadDevignetMaps() {
    // Read the image files
    cv::Mat imgl1 = cv::imread(g_devignet_folder + "/imgl_0d_devignetmap.tiff", cv::IMREAD_UNCHANGED);
    cv::Mat imgr1 = cv::imread(g_devignet_folder + "/imgr_90d_devignetmap.tiff", cv::IMREAD_UNCHANGED);
    cv::Mat imgtop = cv::imread(g_devignet_folder + "/imgtotal_135d_devignetmap.tiff", cv::IMREAD_UNCHANGED);
    cv::Mat imgbottom = cv::imread(g_devignet_folder + "/imgtotal_45d_devignetmap.tiff", cv::IMREAD_UNCHANGED);

    if (imgl1.empty() || imgr1.empty() || imgtop.empty() || imgbottom.empty()) {
        throw std::runtime_error("Failed to load image files");
    }

    // Create OpenCL buffers for the images
    cl_int err;
    g_clLeftDevignetMap = clCreateBuffer(g_clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     imgl1.total() * imgl1.elemSize(), imgl1.data, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create left image buffer");
    }

    g_clRightDevignetMap = clCreateBuffer(g_clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      imgr1.total() * imgr1.elemSize(), imgr1.data, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create right image buffer");
    }

    g_clTopDevignetMap = clCreateBuffer(g_clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      imgtop.total() * imgtop.elemSize(), imgtop.data, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create top image buffer");
    }

    g_clBottomDevignetMap = clCreateBuffer(g_clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        imgbottom.total() * imgbottom.elemSize(), imgbottom.data, &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create bottom image buffer");
    }
}

// Function to initialize OpenCL
bool InitOpenCL() {
    try {
        std::cout << "Starting OpenCL initialization...\n";
        
        // Verify both OpenGL contexts are properly initialized
        if (!g_sharedDC || !g_sharedRC) {
            std::cout << "OpenGL context not properly initialized\n";
            return false;
        }

        // Get platform
        cl_platform_id platform;
        cl_uint numPlatforms;
        cl_int err = clGetPlatformIDs(0, nullptr, &numPlatforms);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to get number of platforms: " << err << std::endl;
            return false;
        }

        if (numPlatforms == 0) {
            std::cout << "No OpenCL platforms found\n";
            return false;
        }
        
        err = clGetPlatformIDs(1, &platform, nullptr);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to get platform: " << err << std::endl;
            return false;
        }

        // Get platform info
        char platformName[1024];
        err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platformName), platformName, nullptr);
        if (err == CL_SUCCESS) {
            std::cout << "OpenCL Platform: " << platformName << "\n";
        }

        // Get device
        cl_uint numDevices;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to get number of devices: " << err << std::endl;
            return false;
        }

        if (numDevices == 0) {
            std::cout << "No GPU devices found\n";
            return false;
        }

        std::cout << "Found " << numDevices << " GPU devices\n";

        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &g_clDevice, nullptr);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to get device: " << err << std::endl;
            return false;
        }

        // Get device info
        char deviceName[1024];
        err = clGetDeviceInfo(g_clDevice, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
        if (err == CL_SUCCESS) {
            std::cout << "OpenCL Device: " << deviceName << "\n";
        }

        // Get device extensions
        size_t extensionsSize;
        err = clGetDeviceInfo(g_clDevice, CL_DEVICE_EXTENSIONS, 0, nullptr, &extensionsSize);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to get device extensions size: " << err << std::endl;
            return false;
        }

        std::vector<char> extensions(extensionsSize);
        err = clGetDeviceInfo(g_clDevice, CL_DEVICE_EXTENSIONS, extensionsSize, extensions.data(), nullptr);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to get device extensions: " << err << std::endl;
            return false;
        }

        std::string extensionsStr(extensions.data());
        std::cout << "Device Extensions: " << extensionsStr << "\n";

        // Check for OpenGL sharing support
        bool hasGLSharing = extensionsStr.find("cl_khr_gl_sharing") != std::string::npos;
        if (!hasGLSharing) {
            std::cout << "Device does not support OpenGL sharing\n";
            return false;
        }

        // Create single context properties for SLM sharing
        cl_context_properties properties[] = {
            CL_GL_CONTEXT_KHR, (cl_context_properties)g_sharedRC,  // OpenGL context
            CL_WGL_HDC_KHR, (cl_context_properties)g_sharedDC,     // OpenGL DC
            CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
            0
        };

        // Create single context
        g_clContext = clCreateContext(properties, 1, &g_clDevice, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create OpenCL context: " << err << std::endl;
            return false;
        }

        // Create single command queue
        g_clQueue = clCreateCommandQueue(g_clContext, g_clDevice, 0, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create command queue: " << err << std::endl;
            return false;
        }

        // Make OpenGL context current
        if (!wglMakeCurrent(g_sharedDC, g_sharedRC)) {
            std::cout << "Failed to make SLM context current for PBO sharing\n";
            return false;
        }
        
        // Create shared buffers in the single context
        g_clSharedTexture = clCreateFromGLBuffer(g_clContext, CL_MEM_WRITE_ONLY, g_SLMPBOID, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create shared texture buffer: " << err << std::endl;
            return false;
        }

        g_clCombinedDisplay = clCreateFromGLBuffer(g_clContext, CL_MEM_WRITE_ONLY, g_capturedPBOID, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create combined display buffer: " << err << std::endl;
            return false;
        }
        int displayWidth = cornerDepthDisplay ? cameraWidth : cameraWidth * 2;
        std::cout << "Combined display buffer created successfully (size: " << displayWidth << "x" << cameraHeight << ")\n";


        // Create program and kernel for spatial period map computation
        g_clPeriodMapProgram = clCreateProgramWithSource(g_clContext, 1, &spatialPeriodMapKernel, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create period map program: " << err << std::endl;
            return false;
        }
        err = clBuildProgram(g_clPeriodMapProgram, 1, &g_clDevice, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to build period map program: " << err << std::endl;
            return false;
        }

        g_clPeriodMapKernel = clCreateKernel(g_clPeriodMapProgram, "computeSpatialPeriodMap", &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create period map kernel: " << err << std::endl;
            return false;
        }

        // Create program for SLM context
        g_clProgram = clCreateProgramWithSource(g_clContext, 1, &kernelSource, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create program: " << err << std::endl;
            return false;
        }
    
        // Build program
        err = clBuildProgram(g_clProgram, 1, &g_clDevice, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            char buildLog[4096];
            clGetProgramBuildInfo(g_clProgram, g_clDevice, CL_PROGRAM_BUILD_LOG, sizeof(buildLog), buildLog, nullptr);
            std::cout << "OpenCL build error:\n" << buildLog << std::endl;
            return false;
        }

        // Create kernel for SLM context
        g_clKernel = clCreateKernel(g_clProgram, "computeSLMPattern", &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create kernel: " << err << std::endl;
            return false;
        }
    
        // Create program and kernel for flow multiplication
        g_clUpdateFlowProgram = clCreateProgramWithSource(g_clContext, 1, &updateFlowKernel, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create flow multiply program: " << err << std::endl;
            return false;
        }

        // Build program with detailed error checking
        err = clBuildProgram(g_clUpdateFlowProgram, 1, &g_clDevice, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to build flow multiply program. Error code: " << err << std::endl;
            return false;
        }

        g_clUpdateFlowKernel = clCreateKernel(g_clUpdateFlowProgram, "updateFlowWithScalar", &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create flow multiply kernel: " << err << std::endl;
            return false;
        }
    
        // Load devignet maps
        if (0) {//devignet) {
            LoadDevignetMaps();
        }

        // Create buffers in captured display context
        size_t leftImageSize = cameraWidth * cameraHeight * cameraChannels * sizeof(float);
        std::cout << "Creating left image buffer of size: " << leftImageSize << " bytes\n";
        
        // Create a zero-initialized host buffer
        std::vector<float> leftImageData(leftImageSize, 0);
        g_clLeftImage = clCreateBuffer(g_clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, leftImageSize, leftImageData.data(), &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create left image buffer. Error code: " << err << std::endl;
            return false;
        }
        std::cout << "Left image buffer created successfully\n";

        size_t rightImageSize = cameraWidth * cameraHeight * cameraChannels * sizeof(float);
        std::cout << "Creating right image buffer of size: " << rightImageSize << " bytes\n";
        
        // Create a zero-initialized host buffer
        std::vector<float> rightImageData(rightImageSize, 0);
        g_clRightImage = clCreateBuffer(g_clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, rightImageSize, rightImageData.data(), &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create right image buffer: " << err << std::endl;
            return false;
        }
        std::cout << "Right image buffer created successfully\n";

        size_t topImageSize = cameraWidth * cameraHeight * cameraChannels * sizeof(float);
        std::cout << "Creating top image buffer of size: " << topImageSize << " bytes\n";
        
        // Create a zero-initialized host buffer
        std::vector<float> topImageData(topImageSize, 0);
        g_clTopImage = clCreateBuffer(g_clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, topImageSize, topImageData.data(), &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create top image buffer: " << err << std::endl;
            return false;
        }
        std::cout << "Top image buffer created successfully\n";

        size_t bottomImageSize = cameraWidth * cameraHeight * cameraChannels * sizeof(float);
        std::cout << "Creating bottom image buffer of size: " << bottomImageSize << " bytes\n";

        // Create a zero-initialized host buffer
        std::vector<float> bottomImageData(bottomImageSize, 0);
        g_clBottomImage = clCreateBuffer(g_clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, bottomImageSize, bottomImageData.data(), &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create bottom image buffer: " << err << std::endl;
            return false;
        }
        std::cout << "Bottom image buffer created successfully\n";

        cv::Mat mask = cv::Mat::ones(cameraHeight, cameraWidth, CV_32F);
        // Set top and bottom margins to 0
        for (int y = 0; y < cameraTopMargin; y++) {
            for (int x = 0; x < cameraWidth; x++) {
                mask.at<float>(y, x) = 0.0f;
            }
        }
        for (int y = cameraHeight - cameraBottomMargin; y < cameraHeight; y++) {
            for (int x = 0; x < cameraWidth; x++) {
                mask.at<float>(y, x) = 0.0f;
            }
        }
        g_clMask = clCreateBuffer(g_clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, cameraWidth * cameraHeight * sizeof(float), mask.data, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create mask buffer: " << err << std::endl;
            return false;
        }

        g_clFlow = clCreateBuffer(g_clContext, CL_MEM_READ_WRITE, cameraWidth * cameraHeight * sizeof(float), nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create flow buffer: " << err << std::endl;
            return false;
        }

        // Create a zero-initialized host buffer for spatial frequency map
        std::vector<float> zeroImage(cameraWidth * cameraHeight, 0.0f);

        // Create the spatial frequency map buffer with proper flags and host pointer
        g_clSpatialFreqMap = clCreateBuffer(g_clContext, 
                                          CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                          cameraWidth * cameraHeight * sizeof(float), 
                                          zeroImage.data(),
                                          &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create spatial frequency map buffer. Error code: " << err << std::endl;
            return false;
        }
        
        g_clSpatialFreqMapSuperpixels = clCreateBuffer(g_clContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
                                                cameraWidth * cameraHeight * sizeof(float), 
                                                zeroImage.data(), &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create spatial frequency map superpixels buffer: " << err << std::endl;
            return false;
        }
        std::cout << "Spatial frequency map superpixels buffer created successfully\n";

        g_clLabels = clCreateBuffer(g_clContext, CL_MEM_READ_WRITE, 
                                                cameraWidth * cameraHeight * sizeof(int), 
                                                nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create labels buffer: " << err << std::endl;
            return false;
        }
        
        g_clWarpedFreqMap = clCreateBuffer(g_clContext, CL_MEM_READ_WRITE, 
                                                SLM_WIDTH * SLM_HEIGHT * sizeof(float), 
                                                nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create warped frequency map buffer: " << err << std::endl;
            return false;
        }

        g_clHomography = clCreateBuffer(g_clContext, CL_MEM_READ_WRITE, 
                                            9 * sizeof(float), nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create homography buffer: " << err << std::endl;
            return false;
        }
        
        // Create temporary buffer in captured context for period map computation
        g_clPeriodTempMap = clCreateBuffer(g_clContext, CL_MEM_READ_WRITE, 
                                            SLM_WIDTH * SLM_HEIGHT * sizeof(float), nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create temporary period map buffer");

        // Create period map buffer in SLM context instead of captured context
        g_clPeriodMap = clCreateBuffer(g_clContext, CL_MEM_READ_WRITE, SLM_WIDTH * SLM_HEIGHT * sizeof(float), nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create period map buffer: " << err << std::endl;
            return false;
        }

        g_clYGrid = clCreateBuffer(g_clContext, CL_MEM_READ_WRITE, SLM_WIDTH * SLM_HEIGHT * sizeof(float), nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create Y grid buffer: " << err << std::endl;
            return false;
        }

        // Copy Y_GRID to GPU once during initialization using SLM queue
        const float* yGridPtr = Y_GRID.ptr<float>();  // Changed from double to float
        err = clEnqueueWriteBuffer(g_clQueue, g_clYGrid, CL_TRUE, 0, 
                                  SLM_WIDTH * SLM_HEIGHT * sizeof(float), 
                                  yGridPtr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to write Y grid to buffer: " << err << std::endl;
            return false;
        }
    
        // Set g_clUpdateFlowKernel arguments
        err = clSetKernelArg(g_clUpdateFlowKernel, 0, sizeof(cl_mem), &g_clFlow);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set flow multiply kernel arg 0");
        
        err = clSetKernelArg(g_clUpdateFlowKernel, 1, sizeof(cl_mem), &g_clSpatialFreqMap);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set flow multiply kernel arg 1");
        
        err = clSetKernelArg(g_clUpdateFlowKernel, 2, sizeof(float), &flow_scalar_learning_rate);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set flow multiply kernel arg 2");
        
        err = clSetKernelArg(g_clUpdateFlowKernel, 3, sizeof(int), &cameraWidth);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set flow multiply kernel arg 3");
        
        err = clSetKernelArg(g_clUpdateFlowKernel, 4, sizeof(int), &cameraHeight);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set flow multiply kernel arg 4");
        
        // Set g_clPeriodMapKernel arguments to compute the period map
        err = clSetKernelArg(g_clPeriodMapKernel, 0, sizeof(cl_mem), &g_clWarpedFreqMap);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set g_clPeriodMapKernel arg 0");
        err = clSetKernelArg(g_clPeriodMapKernel, 1, sizeof(cl_mem), &g_clPeriodMap);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set g_clPeriodMapKernel arg 1");
        err = clSetKernelArg(g_clPeriodMapKernel, 2, sizeof(int), &SLM_WIDTH);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set g_clPeriodMapKernel arg 2");
        err = clSetKernelArg(g_clPeriodMapKernel, 3, sizeof(int), &SLM_HEIGHT);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set g_clPeriodMapKernel arg 3");

        // Set g_clKernel arguments to compute the SLM pattern
        err = clSetKernelArg(g_clKernel, 0, sizeof(cl_mem), &g_clPeriodMap);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set g_clKernel arg 0");
        
        err = clSetKernelArg(g_clKernel, 1, sizeof(cl_mem), &g_clYGrid);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set g_clKernel arg 1");

        err = clSetKernelArg(g_clKernel, 2, sizeof(cl_mem), &g_clSharedTexture);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set g_clKernel arg 2");
        
        err = clSetKernelArg(g_clKernel, 3, sizeof(int), &SLM_WIDTH);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set g_clKernel arg 3");
        
        err = clSetKernelArg(g_clKernel, 4, sizeof(int), &SLM_HEIGHT);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set g_clKernel arg 4");

        int flag = 0;
        err = clSetKernelArg(g_clKernel, 5, sizeof(int), &flag);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set g_clKernel arg 5");

        float depth = 0;
        err = clSetKernelArg(g_clKernel, 6, sizeof(float), &depth);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set g_clKernel arg 6");

        // Create program and kernel for combining images
        g_clCombineImagesProgram = clCreateProgramWithSource(g_clContext, 1, &combineImagesKernel, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create combine images program: " << err << std::endl;
            return false;
        }
        
        // Build program with detailed error checking
        err = clBuildProgram(g_clCombineImagesProgram, 1, &g_clDevice, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            // Get build log
            size_t logSize;
            clGetProgramBuildInfo(g_clCombineImagesProgram, g_clDevice, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
            std::vector<char> buildLog(logSize);
            clGetProgramBuildInfo(g_clCombineImagesProgram, g_clDevice, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);
            
            std::cout << "Failed to build combine images program. Error code: " << err << std::endl;
            std::cout << "Build log:" << std::endl << buildLog.data() << std::endl;
            return false;
        }
        
        g_clCombineImagesKernel = clCreateKernel(g_clCombineImagesProgram, "combineImagesWithColormap", &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create combine images kernel: " << err << std::endl;
            return false;
        }

        // Create the Turbo LUT buffer
        g_clTurboLUT = clCreateBuffer(g_clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * 256 * 3, (void*)turbo_colormap_lut, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create turbo LUT buffer. Error code: " << err << std::endl;
            return false;
        }

        // Create resized depth map buffer for corner display mode
        int cornerWidth = cameraWidth / 5;
        int cornerHeight = cameraHeight / 5;
        g_clResizedDepthMap = clCreateBuffer(g_clContext, CL_MEM_READ_WRITE, cornerWidth * cornerHeight * sizeof(float), nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create resized depth map buffer. Error code: " << err << std::endl;
            return false;
        }

        // Create program and kernel for resizing depth map
        g_clResizeDepthMapProgram = clCreateProgramWithSource(g_clContext, 1, &resizeDepthMapKernel, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create resize depth map program: " << err << std::endl;
            return false;
        }
        
        err = clBuildProgram(g_clResizeDepthMapProgram, 1, &g_clDevice, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to build resize depth map program. Error code: " << err << std::endl;
            return false;
        }
        
        g_clResizeDepthMapKernel = clCreateKernel(g_clResizeDepthMapProgram, "resizeDepthMap", &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create resize depth map kernel: " << err << std::endl;
            return false;
        }

        // Create program and kernel for text overlay blending
        g_clBlendTextOverlayProgram = clCreateProgramWithSource(g_clContext, 1, &blendTextOverlayKernel, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create blend text overlay program: " << err << std::endl;
            return false;
        }
        
        err = clBuildProgram(g_clBlendTextOverlayProgram, 1, &g_clDevice, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to build blend text overlay program. Error code: " << err << std::endl;
            return false;
        }
        
        g_clBlendTextOverlayKernel = clCreateKernel(g_clBlendTextOverlayProgram, "blendTextOverlay", &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create blend text overlay kernel: " << err << std::endl;
            return false;
        }

        // Set combine images kernel arguments after kernel creation
        err = clSetKernelArg(g_clCombineImagesKernel, 0, sizeof(cl_mem), &g_clTopImage);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set combine images kernel arg 0");

        err = clSetKernelArg(g_clCombineImagesKernel, 1, sizeof(cl_mem), &g_clBottomImage);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set combine images kernel arg 1");

        err = clSetKernelArg(g_clCombineImagesKernel, 2, sizeof(cl_mem), &g_clLeftImage);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set combine images kernel arg 2");

        err = clSetKernelArg(g_clCombineImagesKernel, 3, sizeof(cl_mem), &g_clRightImage);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set combine images kernel arg 3");
        
        err = clSetKernelArg(g_clCombineImagesKernel, 4, sizeof(cl_mem), &g_clSpatialFreqMap);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set combine images kernel arg 4");
        
        err = clSetKernelArg(g_clCombineImagesKernel, 5, sizeof(cl_mem), &g_clResizedDepthMap);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set combine images kernel arg 5");
        
        err = clSetKernelArg(g_clCombineImagesKernel, 6, sizeof(cl_mem), &g_clCombinedDisplay);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set combine images kernel arg 6");
        
        err = clSetKernelArg(g_clCombineImagesKernel, 7, sizeof(cl_mem), &g_clTurboLUT);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set combine images kernel arg 7 (turbo LUT)");
        
        err = clSetKernelArg(g_clCombineImagesKernel, 8, sizeof(int), &cameraTopMargin);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set combine images kernel arg 8");
        
        err = clSetKernelArg(g_clCombineImagesKernel, 9, sizeof(int), &cameraBottomMargin);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set combine images kernel arg 9");
        
        err = clSetKernelArg(g_clCombineImagesKernel, 10, sizeof(int), &cameraWidth);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set combine images kernel arg 10");

        err = clSetKernelArg(g_clCombineImagesKernel, 11, sizeof(int), &cameraHeight);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set combine images kernel arg 11");

        int mode = 0;
        err = clSetKernelArg(g_clCombineImagesKernel, 12, sizeof(int), &mode);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set combine images kernel arg 12");
        
        int cornerDisplay = cornerDepthDisplay ? 1 : 0;
        err = clSetKernelArg(g_clCombineImagesKernel, 13, sizeof(int), &cornerDisplay);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set combine images kernel arg 13");
        
        // Create program and kernel for uchar to float conversion
        g_clConvertToFloatProgram = clCreateProgramWithSource(g_clContext, 1, &convertToFloatKernel, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create convert to float program: " << err << std::endl;
            return false;
        }
        
        err = clBuildProgram(g_clConvertToFloatProgram, 1, &g_clDevice, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to build convert to float program: " << err << std::endl;
            return false;
        }
        
        g_clConvertLeftImageToFloatKernel = clCreateKernel(g_clConvertToFloatProgram, "convertToFloat", &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create convert to float kernel: " << err << std::endl;
            return false;
        }

        g_clConvertRightImageToFloatKernel = clCreateKernel(g_clConvertToFloatProgram, "convertToFloat", &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create convert to float kernel: " << err << std::endl;
            return false;
        }

        g_clConvertTopImageToFloatKernel = clCreateKernel(g_clConvertToFloatProgram, "convertToFloat", &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create convert to float kernel: " << err << std::endl;
            return false;
        }

        g_clConvertBottomImageToFloatKernel = clCreateKernel(g_clConvertToFloatProgram, "convertToFloat", &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create convert to float kernel: " << err << std::endl;
            return false;
        }

        // Create temporary buffers for uchar data
        g_clTempLeftBuffer = clCreateBuffer(g_clContext, CL_MEM_READ_ONLY,
                                          cameraWidth * cameraHeight * cameraChannels * sizeof(uchar),
                                          nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create temporary left image buffer: " << err << std::endl;
            return false;
        }

        g_clTempRightBuffer = clCreateBuffer(g_clContext, CL_MEM_READ_ONLY,
                                           cameraWidth * cameraHeight * cameraChannels * sizeof(uchar),
                                           nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create temporary right image buffer: " << err << std::endl;
            return false;
        }

        g_clTempTopBuffer = clCreateBuffer(g_clContext, CL_MEM_READ_ONLY,
                                           cameraWidth * cameraHeight * cameraChannels * sizeof(uchar),
                                           nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create temporary top image buffer: " << err << std::endl;
            return false;
        }

        g_clTempBottomBuffer = clCreateBuffer(g_clContext, CL_MEM_READ_ONLY,
            cameraWidth * cameraHeight * cameraChannels * sizeof(uchar),
            nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create temporary top image buffer: " << err << std::endl;
            return false;
        }

        // Set kernel arguments for left image conversion (these won't change during execution)
        err = clSetKernelArg(g_clConvertLeftImageToFloatKernel, 0, sizeof(cl_mem), &g_clTempLeftBuffer);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set convert kernel arg 0");
        err = clSetKernelArg(g_clConvertLeftImageToFloatKernel, 1, sizeof(cl_mem), &g_clLeftImage);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set convert kernel arg 1");
        err = clSetKernelArg(g_clConvertLeftImageToFloatKernel, 2, sizeof(cl_mem), &g_clLeftDevignetMap);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set convert kernel arg 2");
        err = clSetKernelArg(g_clConvertLeftImageToFloatKernel, 3, sizeof(int), &devignet);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set convert kernel arg 3");
        err = clSetKernelArg(g_clConvertLeftImageToFloatKernel, 4, sizeof(int), &cameraWidth);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set convert kernel arg 4");
        err = clSetKernelArg(g_clConvertLeftImageToFloatKernel, 5, sizeof(int), &cameraHeight);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set convert kernel arg 5");
        err = clSetKernelArg(g_clConvertLeftImageToFloatKernel, 6, sizeof(int), &cameraChannels);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set convert kernel arg 6");

        // Set kernel arguments for right image conversion (these won't change during execution)
        err = clSetKernelArg(g_clConvertRightImageToFloatKernel, 0, sizeof(cl_mem), &g_clTempRightBuffer);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set convert kernel arg 0");
        err = clSetKernelArg(g_clConvertRightImageToFloatKernel, 1, sizeof(cl_mem), &g_clRightImage);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set convert kernel arg 1");
        err = clSetKernelArg(g_clConvertRightImageToFloatKernel, 2, sizeof(cl_mem), &g_clRightDevignetMap);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set convert kernel arg 2");
        err = clSetKernelArg(g_clConvertRightImageToFloatKernel, 3, sizeof(int), &devignet);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set convert kernel arg 3");
        err = clSetKernelArg(g_clConvertRightImageToFloatKernel, 4, sizeof(int), &cameraWidth);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set convert kernel arg 4");
        err = clSetKernelArg(g_clConvertRightImageToFloatKernel, 5, sizeof(int), &cameraHeight);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set convert kernel arg 5");
        err = clSetKernelArg(g_clConvertRightImageToFloatKernel, 6, sizeof(int), &cameraChannels);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set convert kernel arg 6");

        // Set kernel arguments for total image conversion (these won't change during execution)
        err = clSetKernelArg(g_clConvertTopImageToFloatKernel, 0, sizeof(cl_mem), &g_clTempTopBuffer);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set convert kernel arg 0");
        err = clSetKernelArg(g_clConvertTopImageToFloatKernel, 1, sizeof(cl_mem), &g_clTopImage);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set convert kernel arg 1");
        err = clSetKernelArg(g_clConvertTopImageToFloatKernel, 2, sizeof(cl_mem), &g_clTopDevignetMap);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set convert kernel arg 2");
        err = clSetKernelArg(g_clConvertTopImageToFloatKernel, 3, sizeof(int), &devignet);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set convert kernel arg 3");
        err = clSetKernelArg(g_clConvertTopImageToFloatKernel, 4, sizeof(int), &cameraWidth);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set convert kernel arg 4");
        err = clSetKernelArg(g_clConvertTopImageToFloatKernel, 5, sizeof(int), &cameraHeight);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set convert kernel arg 5");
        err = clSetKernelArg(g_clConvertTopImageToFloatKernel, 6, sizeof(int), &cameraChannels);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set convert kernel arg 6");

        // Set kernel arguments for total image conversion (these won't change during execution)
        err = clSetKernelArg(g_clConvertBottomImageToFloatKernel, 0, sizeof(cl_mem), &g_clTempBottomBuffer);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set convert kernel arg 0");
        err = clSetKernelArg(g_clConvertBottomImageToFloatKernel, 1, sizeof(cl_mem), &g_clBottomImage);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set convert kernel arg 1");
        err = clSetKernelArg(g_clConvertBottomImageToFloatKernel, 2, sizeof(cl_mem), &g_clBottomDevignetMap);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set convert kernel arg 2");
        err = clSetKernelArg(g_clConvertBottomImageToFloatKernel, 3, sizeof(int), &devignet);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set convert kernel arg 3");
        err = clSetKernelArg(g_clConvertBottomImageToFloatKernel, 4, sizeof(int), &cameraWidth);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set convert kernel arg 4");
        err = clSetKernelArg(g_clConvertBottomImageToFloatKernel, 5, sizeof(int), &cameraHeight);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set convert kernel arg 5");
        err = clSetKernelArg(g_clConvertBottomImageToFloatKernel, 6, sizeof(int), &cameraChannels);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set convert kernel arg 6");

        std::cout << "OpenCL initialization complete\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "Exception in InitOpenCL: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cout << "Unknown exception in InitOpenCL\n";
        return false;
    }
}

// Function to clean up OpenCL resources
void CleanupOpenCL() {
    try {
        std::cout << "Starting OpenCL cleanup...\n";
        
        // First, make sure we're in a valid OpenGL context for SLM
        if (g_sharedDC && g_sharedRC) {
            if (!wglMakeCurrent(g_sharedDC, g_sharedRC)) {
                std::cout << "Warning: Failed to make SLM OpenGL context current during cleanup\n";
            }
        }

        // Release shared buffers first
        if (g_clSharedTexture) {
            std::cout << "Releasing shared texture...\n";
            clReleaseMemObject(g_clSharedTexture);
            g_clSharedTexture = nullptr;
        }
        
        if (g_clCombinedDisplay) {
            std::cout << "Releasing combined display buffer...\n";
            clReleaseMemObject(g_clCombinedDisplay);
            g_clCombinedDisplay = nullptr;
        }

        // Release kernels before their programs
        if (g_clKernel) {
            std::cout << "Releasing main kernel...\n";
            clReleaseKernel(g_clKernel);
            g_clKernel = nullptr;
        }

        if (g_clPeriodMapKernel) {
            std::cout << "Releasing period map kernel...\n";
            clReleaseKernel(g_clPeriodMapKernel);
            g_clPeriodMapKernel = nullptr;
        }

        if (g_clUpdateFlowKernel) {
            std::cout << "Releasing flow update kernel...\n";
            clReleaseKernel(g_clUpdateFlowKernel);
            g_clUpdateFlowKernel = nullptr;
        }

        if (g_clCombineImagesKernel) {
            std::cout << "Releasing combine images kernel...\n";
            clReleaseKernel(g_clCombineImagesKernel);
            g_clCombineImagesKernel = nullptr;
        }

        if (g_clResizeDepthMapKernel) {
            std::cout << "Releasing resize depth map kernel...\n";
            clReleaseKernel(g_clResizeDepthMapKernel);
            g_clResizeDepthMapKernel = nullptr;
        }

        if (g_clBlendTextOverlayKernel) {
            std::cout << "Releasing blend text overlay kernel...\n";
            clReleaseKernel(g_clBlendTextOverlayKernel);
            g_clBlendTextOverlayKernel = nullptr;
        }

        if (g_clConvertLeftImageToFloatKernel) {
            std::cout << "Releasing convert left image to float kernel...\n";
            clReleaseKernel(g_clConvertLeftImageToFloatKernel);
            g_clConvertLeftImageToFloatKernel = nullptr;
        }

        if (g_clConvertRightImageToFloatKernel) {
            std::cout << "Releasing convert right image to float kernel...\n";
            clReleaseKernel(g_clConvertRightImageToFloatKernel);
            g_clConvertRightImageToFloatKernel = nullptr;
        }

        if (g_clConvertTopImageToFloatKernel) {
            std::cout << "Releasing convert total image to float kernel...\n";
            clReleaseKernel(g_clConvertTopImageToFloatKernel);
            g_clConvertTopImageToFloatKernel = nullptr;
        }

        if (g_clConvertBottomImageToFloatKernel) {
            std::cout << "Releasing convert bottom image to float kernel...\n";
            clReleaseKernel(g_clConvertBottomImageToFloatKernel);
            g_clConvertBottomImageToFloatKernel = nullptr;
        }

        // Release programs
        if (g_clProgram) {
            std::cout << "Releasing main program...\n";
            clReleaseProgram(g_clProgram);
            g_clProgram = nullptr;
        }

        if (g_clPeriodMapProgram) {
            std::cout << "Releasing period map program...\n";
            clReleaseProgram(g_clPeriodMapProgram);
            g_clPeriodMapProgram = nullptr;
        }

        if (g_clUpdateFlowProgram) {
            std::cout << "Releasing flow update program...\n";
            clReleaseProgram(g_clUpdateFlowProgram);
            g_clUpdateFlowProgram = nullptr;
        }

        if (g_clCombineImagesProgram) {
            std::cout << "Releasing combine images program...\n";
            clReleaseProgram(g_clCombineImagesProgram);
            g_clCombineImagesProgram = nullptr;
        }

        if (g_clResizeDepthMapProgram) {
            std::cout << "Releasing resize depth map program...\n";
            clReleaseProgram(g_clResizeDepthMapProgram);
            g_clResizeDepthMapProgram = nullptr;
        }

        if (g_clBlendTextOverlayProgram) {
            std::cout << "Releasing blend text overlay program...\n";
            clReleaseProgram(g_clBlendTextOverlayProgram);
            g_clBlendTextOverlayProgram = nullptr;
        }

        // Release memory objects
        if (g_clYGrid) {
            std::cout << "Releasing Y grid buffer...\n";
            clReleaseMemObject(g_clYGrid);
            g_clYGrid = nullptr;
        }

        if (g_clPeriodMap) {
            std::cout << "Releasing period map buffer...\n";
            clReleaseMemObject(g_clPeriodMap);
            g_clPeriodMap = nullptr;
        }

        if (g_clPeriodTempMap) {
            std::cout << "Releasing period temp map buffer...\n";
            clReleaseMemObject(g_clPeriodTempMap);
            g_clPeriodTempMap = nullptr;
        }

        if (g_clWarpedFreqMap) {
            std::cout << "Releasing warped frequency map buffer...\n";
            clReleaseMemObject(g_clWarpedFreqMap);
            g_clWarpedFreqMap = nullptr;
        }

        if (g_clSpatialFreqMap) {
            std::cout << "Releasing spatial frequency map buffer...\n";
            clReleaseMemObject(g_clSpatialFreqMap);
            g_clSpatialFreqMap = nullptr;
        }

        if (g_clLabels) {
            std::cout << "Releasing labels buffer...\n";
            clReleaseMemObject(g_clLabels);
            g_clLabels = nullptr;
        }

        if (g_clSpatialFreqMapSuperpixels) {
            std::cout << "Releasing spatial frequency map superpixels buffer...\n";
            clReleaseMemObject(g_clSpatialFreqMapSuperpixels);
            g_clSpatialFreqMapSuperpixels = nullptr;
        }

        if (g_clTopImage) {
            std::cout << "Releasing top image buffer...\n";
            clReleaseMemObject(g_clTopImage);
            g_clTopImage = nullptr;
        }

        if (g_clBottomImage) {
            std::cout << "Releasing bottom image buffer...\n";
            clReleaseMemObject(g_clBottomImage);
            g_clBottomImage = nullptr;
        }

        if (g_clRightImage) {
            std::cout << "Releasing right image buffer...\n";
            clReleaseMemObject(g_clRightImage);
            g_clRightImage = nullptr;
        }

        if (g_clLeftImage) {
            std::cout << "Releasing left image buffer...\n";
            clReleaseMemObject(g_clLeftImage);
            g_clLeftImage = nullptr;
        }

        if (g_clMask) {
            std::cout << "Releasing mask buffer...\n";
            clReleaseMemObject(g_clMask);
            g_clMask = nullptr;
        }

        if (g_clFlow) {
            std::cout << "Releasing flow buffer...\n";
            clReleaseMemObject(g_clFlow);
            g_clFlow = nullptr;
        }

        if (g_clTurboLUT) {
            std::cout << "Releasing turbo LUT buffer...\n";
            clReleaseMemObject(g_clTurboLUT);
            g_clTurboLUT = nullptr;
        }

        if (g_clResizedDepthMap) {
            std::cout << "Releasing resized depth map buffer...\n";
            clReleaseMemObject(g_clResizedDepthMap);
            g_clResizedDepthMap = nullptr;
        }

        if (g_clTextOverlayOff) {
            std::cout << "Releasing SVAF OFF text overlay buffer...\n";
            clReleaseMemObject(g_clTextOverlayOff);
            g_clTextOverlayOff = nullptr;
        }

        if (g_clTextOverlayOnCenter) {
            std::cout << "Releasing SVAF ON CENTER text overlay buffer...\n";
            clReleaseMemObject(g_clTextOverlayOnCenter);
            g_clTextOverlayOnCenter = nullptr;
        }

        if (g_clTextOverlayOn) {
            std::cout << "Releasing SVAF ON text overlay buffer...\n";
            clReleaseMemObject(g_clTextOverlayOn);
            g_clTextOverlayOn = nullptr;
        }

        // Release command queue
        if (g_clQueue) {
            std::cout << "Releasing command queue...\n";
            clReleaseCommandQueue(g_clQueue);
            g_clQueue = nullptr;
        }

        // Release device
        if (g_clDevice) {
            std::cout << "Releasing device...\n";
            clReleaseDevice(g_clDevice);
            g_clDevice = nullptr;
        }

        // Finally release contexts
        if (g_clContext) {
            std::cout << "Releasing context...\n";
            clReleaseContext(g_clContext);
            g_clContext = nullptr;
        }

        // Clean up WarpPerspective object
        if (g_warp) {
            std::cout << "Cleaning up WarpPerspective object...\n";
            delete g_warp;
            g_warp = nullptr;
        }

        if (g_clTempLeftBuffer) {
            std::cout << "Releasing temporary left image buffer...\n";
            clReleaseMemObject(g_clTempLeftBuffer);
            g_clTempLeftBuffer = nullptr;
        }

        if (g_clTempRightBuffer) {
            std::cout << "Releasing temporary right image buffer...\n";
            clReleaseMemObject(g_clTempRightBuffer);
            g_clTempRightBuffer = nullptr;
        }

        if (g_clTempTopBuffer) {
            std::cout << "Releasing temporary top image buffer...\n";
            clReleaseMemObject(g_clTempTopBuffer);
            g_clTempTopBuffer = nullptr;
        }

        if (g_clTempBottomBuffer) {
            std::cout << "Releasing temporary bottom image buffer...\n";
            clReleaseMemObject(g_clTempBottomBuffer);
            g_clTempBottomBuffer = nullptr;
        }

        std::cout << "OpenCL cleanup complete\n";
    } catch (const std::exception& e) {
        std::cout << "Exception during OpenCL cleanup: " << e.what() << std::endl;
    } catch (...) {
        std::cout << "Unknown exception during OpenCL cleanup\n";
    }
}

// Host functions -> CPU execution (DRAM)

// Device functions (Kernel) -> GPU execution (VRAM)

// Function to compute SLM pattern from period map using OpenCL
void ComputeSLMPatternFromPeriodMap() {    
    // Execute kernel
    size_t globalSizeSLM[2] = {static_cast<size_t>(SLM_WIDTH), static_cast<size_t>(SLM_HEIGHT)};
    cl_int err = clEnqueueNDRangeKernel(g_clQueue, g_clKernel, 2, nullptr, globalSizeSLM, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to execute g_clKernel: " << err << std::endl;
        throw std::runtime_error("Failed to execute g_clKernel");
    }
    err = clFinish(g_clQueue);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to synchronize after compute SLM pattern kernel execution");
    }
}

// Add new function to multiply flow with scalar
void UpdateSpatialFreqMapWithScalar() {
    cl_int err;
    err = clEnqueueNDRangeKernel(g_clQueue, g_clUpdateFlowKernel, 2, nullptr, globalCameraSize, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to execute flow multiply kernel");
}

// Function to read the SLM pattern from GPU
void ReadSLMPattern(cv::Mat& slmPattern) {
    cl_int err = clEnqueueReadBuffer(g_clQueue, g_clSharedTexture, CL_TRUE, 0, 
                                    SLM_WIDTH * SLM_HEIGHT * sizeof(uchar), 
                                    slmPattern.data, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cout << "Failed to read SLM pattern: " << err << std::endl;
        return;
    }
}

LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
        case WM_SIZE: {
            if (hwnd == g_capturedWnd) {
                // Get the new window size
                int newWidth = LOWORD(lParam);
                int newHeight = HIWORD(lParam);
                
                // Make the OpenGL context current
                if (wglMakeCurrent(g_sharedDC, g_sharedRC)) {
                    // Update viewport to match new window size
                    glViewport(0, 0, newWidth, newHeight);
                    
                    // Update orthographic projection to maintain aspect ratio
                    glMatrixMode(GL_PROJECTION);
                    glLoadIdentity();
                    // Use the actual image dimensions for the orthographic projection
                    int projectionWidth = cornerDepthDisplay ? cameraWidth : cameraWidth * 2;
                    glOrtho(0, static_cast<double>(projectionWidth), cameraHeight, 0, -1, 1);
                    glMatrixMode(GL_MODELVIEW);
                    glLoadIdentity();
                    
                    // Force a redraw
                    InvalidateRect(hwnd, NULL, FALSE);
                }
            }
            break;
        }
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

// Function to initialize OpenGL for SLM Display
bool InitOpenGL() {
    try {
        std::cout << "Starting OpenGL initialization...\n";
        
        /*********************** Create the SLM and Captured Display Windows ***********************/
        std::cout << "Getting virtual screen metrics...\n";
        // Get virtual screen metrics (accounts for all displays)
        int virtualScreenWidth = GetSystemMetrics(SM_CXVIRTUALSCREEN);
        int virtualScreenHeight = GetSystemMetrics(SM_CYVIRTUALSCREEN);
        int virtualScreenX = GetSystemMetrics(SM_XVIRTUALSCREEN);
        
        std::cout << "Calculating SLM window position...\n";
        // Calculate the position for the SLM window (at the rightmost edge of virtual screen - SLM width)
        int slmX = virtualScreenX + virtualScreenWidth - SLM_WIDTH;
        
        std::cout << "Creating SLM window...\n";
        // Create full resolution SLM window
        g_SLMWnd = CreateWindowEx(
            WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
            L"STATIC",
            L"SLM Display",
            WS_POPUP | WS_VISIBLE | WS_CLIPSIBLINGS | WS_CLIPCHILDREN | WS_DISABLED,
            slmX, 0,  // Position at the rightmost edge minus SLM width
            SLM_WIDTH, SLM_HEIGHT,
            nullptr, nullptr, nullptr, nullptr
        );
        if (!g_SLMWnd) throw std::runtime_error("Failed to create SLM window");
        // Ensure window stays on top
        SetWindowPos(g_SLMWnd, HWND_TOPMOST, 0, 0, SLM_WIDTH, SLM_HEIGHT, 
                    SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE | SWP_NOOWNERZORDER);

        // Hide the window from the taskbar and prevent any interaction
        ShowWindow(g_SLMWnd, SW_SHOW);
        SetWindowLong(g_SLMWnd, GWL_EXSTYLE, GetWindowLong(g_SLMWnd, GWL_EXSTYLE) | 
                    WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE);
        EnableWindow(g_SLMWnd, FALSE);  // Disable the window completely

        std::cout << "Creating captured display window...\n";
        // Calculate window size at 39% of original while maintaining aspect ratio
        double captured_display_scale = 0.6; //0.39;
        int displayWidth = cornerDepthDisplay ? cameraWidth : cameraWidth * 2;
        double aspectRatio = static_cast<double>(displayWidth) / static_cast<double>(cameraHeight - cameraTopMargin - cameraBottomMargin);
        int windowWidth = static_cast<int>(displayWidth * captured_display_scale);
        int windowHeight = static_cast<int>(windowWidth / aspectRatio);
        int screenWidth = GetSystemMetrics(SM_CXSCREEN);
        int screenHeight = GetSystemMetrics(SM_CYSCREEN);
        
        // Set window position to center horizontally but align top with main display
        int windowX = (screenWidth - windowWidth) / 2;
        int windowY = 0;  // Align with top of screen

        // Register window class
        WNDCLASSEX wc = { 0 };
        wc.cbSize = sizeof(WNDCLASSEX);
        wc.style = CS_OWNDC;
        wc.lpfnWndProc = WindowProc;
        wc.hInstance = GetModuleHandle(nullptr);
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        wc.lpszClassName = L"CapturedWindowClass";
        RegisterClassEx(&wc);

        // Create captured display window with proper style for movement
        g_capturedWnd = CreateWindowEx(
            0,
            L"CapturedWindowClass",
            L"Captured Display",
            WS_OVERLAPPEDWINDOW | WS_VISIBLE | WS_CLIPSIBLINGS | WS_CLIPCHILDREN,
            windowX, windowY,
            windowWidth, windowHeight,
            nullptr, nullptr, GetModuleHandle(nullptr), nullptr
        );
        // Adjust window size to account for borders and title bar
        RECT clientRect = { 0, 0, windowWidth, windowHeight};
        AdjustWindowRect(&clientRect, WS_OVERLAPPEDWINDOW, FALSE);
        SetWindowPos(g_capturedWnd, nullptr, windowX, windowY, 
                    clientRect.right - clientRect.left, 
                    clientRect.bottom - clientRect.top,
                    SWP_NOZORDER | SWP_NOACTIVATE);
        if (!g_capturedWnd) throw std::runtime_error("Failed to create captured display window");

        /*********************** Initialize the OpenGL DC and Context ***********************/
        // Create DCs for both windows
        HDC slmDC = GetDC(g_SLMWnd);
        HDC capturedDC = GetDC(g_capturedWnd);
        if (!slmDC || !capturedDC) throw std::runtime_error("Failed to create DCs");

        // Set up pixel format for both DCs
        PIXELFORMATDESCRIPTOR pfd = {
            sizeof(PIXELFORMATDESCRIPTOR),
            1,
            PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,
            PFD_TYPE_RGBA,
            32,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            24,
            0, 0,
            PFD_MAIN_PLANE,
            0,
            0, 0, 0
        };

        // Try to find a suitable pixel format for SLM window
        int pixelFormat = ChoosePixelFormat(slmDC, &pfd);
        if (!pixelFormat) {
            DWORD error = GetLastError();
            std::cout << "Failed to choose pixel format for SLM window. Error code: " << error << std::endl;
            throw std::runtime_error("Failed to choose pixel format for SLM window");
        }

        // Describe the pixel format
        PIXELFORMATDESCRIPTOR pfd2;
        if (!DescribePixelFormat(slmDC, pixelFormat, sizeof(PIXELFORMATDESCRIPTOR), &pfd2)) {
            DWORD error = GetLastError();
            std::cout << "Failed to describe pixel format. Error code: " << error << std::endl;
            throw std::runtime_error("Failed to describe pixel format");
        }

        // Set the pixel format for both windows
        if (!SetPixelFormat(slmDC, pixelFormat, &pfd2)) {
            DWORD error = GetLastError();
            std::cout << "Failed to set pixel format for SLM window. Error code: " << error << std::endl;
            throw std::runtime_error("Failed to set pixel format for SLM window");
        }

        if (!SetPixelFormat(capturedDC, pixelFormat, &pfd2)) {
            DWORD error = GetLastError();
            std::cout << "Failed to set pixel format for captured window. Error code: " << error << std::endl;
            throw std::runtime_error("Failed to set pixel format for captured window");
        }

        // Create a single OpenGL context using the SLM window's DC
        g_sharedRC = wglCreateContext(slmDC);
        if (!g_sharedRC) throw std::runtime_error("Failed to create OpenGL context");

        // Make the context current with the SLM window's DC
        if (!wglMakeCurrent(slmDC, g_sharedRC)) throw std::runtime_error("Failed to make OpenGL context current");

        // Store the SLM window's DC as the shared DC
        g_sharedDC = slmDC;

        // Get OpenGL version info, vendor info, and renderer info
        const GLubyte* version = glGetString(GL_VERSION);
        if (version) std::cout << "OpenGL Version: " << version << "\n";
        const GLubyte* vendor = glGetString(GL_VENDOR);
        if (vendor) std::cout << "OpenGL Vendor: " << vendor << "\n";
        const GLubyte* renderer = glGetString(GL_RENDERER);
        if (renderer) std::cout << "OpenGL Renderer: " << renderer << "\n";
        GLenum error = glGetError();
        if (error != GL_NO_ERROR) {
            std::cout << "OpenGL error after context creation: " << error << "\n";
            return false;
        }

        // Release the captured window's DC since we'll use the shared context
        ReleaseDC(g_capturedWnd, capturedDC);

        /*********************** Start SLM Initialization ***********************/
        // Create and bind texture for SLM (single-channel grayscale)
        glGenTextures(1, &g_SLMTextureID);
        error = glGetError();
        if (error != GL_NO_ERROR) {
            throw std::runtime_error("Failed to generate texture, error: " + std::to_string(error));
        }

        glBindTexture(GL_TEXTURE_2D, g_SLMTextureID);
        error = glGetError();
        if (error != GL_NO_ERROR) {
            throw std::runtime_error("Failed to bind texture, error: " + std::to_string(error));
        }

        // Critical setting for proper luminance
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);  // Ensure 1-byte alignment for luminance
        
        // Set texture parameters for SLM
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

        // Set texture environment mode to replace color
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

        // Allocate texture storage with explicit LUMINANCE8 internal format for SLM
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE8, SLM_WIDTH, SLM_HEIGHT, 0,
                     GL_LUMINANCE, GL_UNSIGNED_BYTE, nullptr);
        error = glGetError();
        if (error != GL_NO_ERROR) {
            throw std::runtime_error("Failed to allocate texture storage, error: " + std::to_string(error));
        }
        glBindTexture(GL_TEXTURE_2D, 0);
        if (g_SLMTextureID == 0) throw std::runtime_error("SLM OpenGL resources not properly initialized");

        /*********************** Start Captured Display Initialization ***********************/
        // Initialize captured display OpenGL resources
        std::cout << "Starting captured OpenGL initialization...\n";
        if (!wglMakeCurrent(slmDC, g_sharedRC)) {
            std::cout << "Captured OpenGL context is not current\n";
            return false;
        }
        
        // Set up viewport to match window size
        std::cout << "Setting up captured display viewport...\n";
        glViewport(0, 0, windowWidth, windowHeight);
        error = glGetError();
        if (error != GL_NO_ERROR) {
            throw std::runtime_error("Error setting captured display viewport: " + std::to_string(error));
        }

        // Set up orthographic projection to match image aspect ratio
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        // Use the actual image dimensions for the orthographic projection
        int projectionWidth = cornerDepthDisplay ? cameraWidth : cameraWidth * 2;
        glOrtho(0, static_cast<double>(projectionWidth), cameraHeight, 0, -1, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        error = glGetError();
        if (error != GL_NO_ERROR) {
            throw std::runtime_error("Error setting up captured display matrices: " + std::to_string(error));
        }
        std::cout << "Captured display viewport and matrices set up successfully\n";

        // Create and initialize captured texture
        std::cout << "Creating captured display texture...\n";
        glGenTextures(1, &g_capturedTextureID);
        error = glGetError();
        if (error != GL_NO_ERROR || g_capturedTextureID == 0) {
            throw std::runtime_error("Failed to generate captured display texture, error: " + std::to_string(error));
        }
        std::cout << "Captured display texture created successfully, ID: " << g_capturedTextureID << "\n";

        std::cout << "Binding texture...\n";
        glBindTexture(GL_TEXTURE_2D, g_capturedTextureID);
        error = glGetError();
        if (error != GL_NO_ERROR) {
            throw std::runtime_error("Failed to bind captured display texture, error: " + std::to_string(error));
        }
        std::cout << "Captured display texture bound successfully\n";

        // Set captured texture parameters
        std::cout << "Setting captured display texture parameters...\n";
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        error = glGetError();
        if (error != GL_NO_ERROR) {
            throw std::runtime_error("Error setting captured display texture parameters: " + std::to_string(error));
        }
        std::cout << "Captured display texture parameters set successfully\n";
        // Set texture environment mode to replace luminance
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

        // Allocate captured texture storage based on display mode
        std::cout << "Allocating captured display texture storage...\n";
        int textureWidth = cornerDepthDisplay ? cameraWidth : cameraWidth * 2;
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, textureWidth, cameraHeight, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        error = glGetError();
        if (error != GL_NO_ERROR) {
            throw std::runtime_error("Failed to allocate captured display texture storage, error: " + std::to_string(error));
        }
        std::cout << "Captured display texture storage allocated successfully (width: " << textureWidth << ")\n";
        glBindTexture(GL_TEXTURE_2D, 0);

        /*********************** Start PBO Initialization ***********************/
        // Verify OpenGL context is current before PBO creation
        if (!wglMakeCurrent(slmDC, g_sharedRC)) {
            throw std::runtime_error("OpenGL context is not current before PBO creation");
        }

        // Get function pointers for glGenBuffers, glBindBuffer, and glBufferData
        PFNGLGENBUFFERSPROC glGenBuffers = (PFNGLGENBUFFERSPROC)wglGetProcAddress("glGenBuffers");
        if (!glGenBuffers) throw std::runtime_error("Failed to get glGenBuffers function pointer");
        PFNGLBINDBUFFERPROC  glBindBuffer = (PFNGLBINDBUFFERPROC )wglGetProcAddress("glBindBuffer");
        if (!glBindBuffer) throw std::runtime_error("Failed to get glBindBuffer function pointer");
        PFNGLBUFFERDATAPROC glBufferData = (PFNGLBUFFERDATAPROC)wglGetProcAddress("glBufferData");
        if (!glBufferData) throw std::runtime_error("Failed to get glBufferData function pointer");

        // Create the SLM PBO
        std::cout << "Creating SLM PBO...\n";
        glBindTexture(GL_TEXTURE_2D, g_SLMTextureID);
        g_SLMPBOID = 0;
        GLuint pbo = 0;
        glGenBuffers(1, &pbo);
        error = glGetError();
        if (error != GL_NO_ERROR) throw std::runtime_error("Failed to generate PBO, error: " + std::to_string(error));
        if (pbo == 0) throw std::runtime_error("PBO ID is 0 after generation");
        std::cout << "Binding PBO...\n";
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        error = glGetError();
        if (glGetError() != GL_NO_ERROR) throw std::runtime_error("Failed to bind PBO, error: " + std::to_string(error));
        std::cout << "Allocating PBO storage...\n";
        size_t size = SLM_WIDTH * SLM_HEIGHT * sizeof(GLubyte);  // Single channel
        std::cout << "PBO size: " << size << " bytes\n";
        glBufferData(GL_PIXEL_UNPACK_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);
        error = glGetError();
        if (error != GL_NO_ERROR) throw std::runtime_error("Failed to allocate PBO storage, error: " + std::to_string(error));

        // Set the SLM PBO pointer if PBO is properly bound and allocated
        GLint boundBuffer = 0;
        glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &boundBuffer);
        if (boundBuffer != pbo) throw std::runtime_error("PBO binding verification failed");
        g_SLMPBOID = pbo;
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
        
        // Create captured display PBO in the same context
        std::cout << "Creating captured display PBO...\n";
        glBindTexture(GL_TEXTURE_2D, g_capturedTextureID);
        g_capturedPBOID = 0;
        GLuint capturedPbo = 0;
        glGenBuffers(1, &capturedPbo);
        error = glGetError();
        if (error != GL_NO_ERROR) throw std::runtime_error("Failed to generate captured PBO, error: " + std::to_string(error));
        if (capturedPbo == 0) throw std::runtime_error("Captured PBO ID is 0 after generation");
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, capturedPbo);
        error = glGetError();
        if (error != GL_NO_ERROR) throw std::runtime_error("Failed to bind captured PBO, error: " + std::to_string(error));
        size_t capturedSize = textureWidth * cameraHeight * 3 * sizeof(GLubyte);
        std::cout << "Captured PBO size: " << capturedSize << " bytes\n";
        glBufferData(GL_PIXEL_UNPACK_BUFFER, capturedSize, nullptr, GL_DYNAMIC_DRAW);
        error = glGetError();
        if (error != GL_NO_ERROR) throw std::runtime_error("Failed to allocate captured PBO storage, error: " + std::to_string(error));

        // Set the captured PBO pointer if PBO is properly bound and allocated
        glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &boundBuffer);
        if (boundBuffer != capturedPbo) throw std::runtime_error("Captured PBO binding verification failed");
        g_capturedPBOID = capturedPbo;
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
        
        std::cout << "OpenGL initialization complete!\n";
        std::cout << "SLM Texture ID: " << g_SLMTextureID << "\n";
        std::cout << "SLM PBO ID: " << g_SLMPBOID << "\n";
        std::cout << "Captured Texture ID: " << g_capturedTextureID << "\n";
        std::cout << "Captured PBO ID: " << g_capturedPBOID << "\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "Exception in InitOpenGL: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cout << "Unknown exception in InitOpenGL\n";
        return false;
    }
}

// Function to clean up OpenGL resources
void CleanupOpenGL() {
    try {
        std::cout << "CleanupOpenGL: Starting cleanup...\n";
        
        // Make sure we have a valid context
        if (g_sharedDC && g_sharedRC) {
            std::cout << "CleanupOpenGL: Attempting to make context current...\n";
            if (!wglMakeCurrent(g_sharedDC, g_sharedRC)) {
                std::cout << "CleanupOpenGL: Warning: Failed to make SLM OpenGL context current during cleanup\n";
                return;  // Exit if we can't make the context current
            }
            std::cout << "CleanupOpenGL: Context made current successfully\n";
        } else {
            std::cout << "CleanupOpenGL: No valid DC or RC found\n";
            return;
        }

        // Verify context is still current
        HGLRC currentContext = wglGetCurrentContext();
        if (currentContext != g_sharedRC) {
            std::cout << "CleanupOpenGL: Warning: Context is not current before PBO deletion\n";
        if (!wglMakeCurrent(g_sharedDC, g_sharedRC)) {
                std::cout << "CleanupOpenGL: Failed to make context current again\n";
                return;
            }
        }

        // Delete OpenGL objects while context is still valid
        PFNGLDELETEBUFFERSPROC glDeleteBuffers = (PFNGLDELETEBUFFERSPROC)wglGetProcAddress("glDeleteBuffers");
        if (!glDeleteBuffers) {
            std::cout << "CleanupOpenGL: Warning: Failed to get glDeleteBuffers function pointer\n";
        }
        PFNGLBINDBUFFERPROC glBindBuffer = (PFNGLBINDBUFFERPROC )wglGetProcAddress("glBindBuffer");
        if (!glBindBuffer) {
            std::cout << "CleanupOpenGL: Warning: Failed to get glBindBuffer function pointer\n";
        }

        if (g_SLMPBOID) {
            std::cout << "CleanupOpenGL: Deleting SLM PBO...\n";

            // Unbind the PBO first
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            
            // Now delete the PBO
            glDeleteBuffers(1, &g_SLMPBOID);
            GLenum error = glGetError();
            if (error != GL_NO_ERROR) {
                std::cout << "CleanupOpenGL: Warning: Error deleting SLM PBO: " << error << "\n";
            }
            g_SLMPBOID = 0;
            std::cout << "CleanupOpenGL: SLM PBO deleted\n";
        }

        if (g_capturedPBOID) {
            std::cout << "CleanupOpenGL: Deleting captured PBO...\n";
            
            // Unbind the PBO first
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            
            // Now delete the PBO
            glDeleteBuffers(1, &g_capturedPBOID);
            GLenum error = glGetError();
            if (error != GL_NO_ERROR) {
                std::cout << "CleanupOpenGL: Warning: Error deleting captured PBO: " << error << "\n";
            }
            g_capturedPBOID = 0;
            std::cout << "CleanupOpenGL: Captured PBO deleted\n";
        }

        if (g_SLMTextureID) {
            std::cout << "CleanupOpenGL: Deleting texture...\n";
            glDeleteTextures(1, &g_SLMTextureID);
            GLenum error = glGetError();
            if (error != GL_NO_ERROR) {
                std::cout << "CleanupOpenGL: Warning: Error deleting SLM texture: " << error << "\n";
            }
            g_SLMTextureID = 0;
            std::cout << "CleanupOpenGL: SLM texture deleted\n";
        }

        if (g_capturedTextureID) {
            std::cout << "CleanupCapturedOpenGL: Deleting texture...\n";
            glDeleteTextures(1, &g_capturedTextureID);
            GLenum error = glGetError();
            if (error != GL_NO_ERROR) {
                std::cout << "CleanupCapturedOpenGL: Warning: Error deleting captured texture: " << error << "\n";
            }
            g_capturedTextureID = 0;
            std::cout << "CleanupCapturedOpenGL: Captured texture deleted\n";
        }

        // Release OpenGL context
        if (g_sharedRC) {
            std::cout << "CleanupOpenGL: Releasing OpenGL context...\n";
            wglMakeCurrent(nullptr, nullptr);
            wglDeleteContext(g_sharedRC);
            g_sharedRC = nullptr;
            std::cout << "CleanupOpenGL: OpenGL context released\n";
        }

        // Release DC last
        if (g_sharedDC) {
            std::cout << "CleanupOpenGL: Releasing DC...\n";
            ReleaseDC(g_SLMWnd, g_sharedDC);
            g_sharedDC = nullptr;
            std::cout << "CleanupOpenGL: DC released\n";
        }
        std::cout << "CleanupOpenGL: Cleanup complete\n";
    } catch (const std::exception& e) {
        std::cout << "CleanupOpenGL: Exception during cleanup: " << e.what() << std::endl;
    } catch (...) {
        std::cout << "CleanupOpenGL: Unknown exception during cleanup\n";
    }
}

// Function to update texture with new image data
void UpdateSLMTexture(cv::Mat& slmPattern, bool showBlackPattern) {
    try {
        // Make sure we're using the shared context
        if (!wglMakeCurrent(g_sharedDC, g_sharedRC)) {
            throw std::runtime_error("Failed to make OpenGL context current for SLM texture update");
        }

        // Bind texture and verify
        glBindTexture(GL_TEXTURE_2D, g_SLMTextureID);
        GLenum error = glGetError();
        if (error != GL_NO_ERROR) {
            std::cout << "Failed to bind texture, error: " << error << "\n";
            throw std::runtime_error("Failed to bind texture");
        }

        if (showBlackPattern) {
            // For black pattern, use direct CPU upload with LUMINANCE format
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, SLM_WIDTH, SLM_HEIGHT, 
                           GL_LUMINANCE, GL_UNSIGNED_BYTE, slmPattern.data);
            error = glGetError();
            if (error != GL_NO_ERROR) {
                std::cout << "Failed to update texture with black pattern, error: " << error << "\n";
                throw std::runtime_error("Failed to update texture with black pattern");
            }
        } else {
            if (!g_clQueue || !g_clSharedTexture || !g_clKernel) {
                std::cout << "OpenCL not properly initialized\n";
                throw std::runtime_error("OpenCL not properly initialized");
            }

            // Get PBO function pointers
            PFNGLBINDBUFFERPROC glBindBuffer = (PFNGLBINDBUFFERPROC)wglGetProcAddress("glBindBuffer");
            if (!glBindBuffer) {
                std::cout << "Failed to get glBindBuffer function pointer\n";
                throw std::runtime_error("Failed to get glBindBuffer function pointer");
            }

            // Unbind any existing PBO first
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            error = glGetError();
            if (error != GL_NO_ERROR) {
                std::cout << "Error unbinding existing PBO: " << error << "\n";
                throw std::runtime_error("Failed to unbind existing PBO");
            }

            // Bind the SLM PBO
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_SLMPBOID);
            error = glGetError();
            if (error != GL_NO_ERROR) {
                std::cout << "Failed to bind PBO, error: " << error << "\n";
                throw std::runtime_error("Failed to bind PBO");
            }

            glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // Ensure 1-byte alignment for luminance

            // Verify SLM PBO is properly bound
            GLint boundBuffer = 0;
            glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &boundBuffer);
            if (boundBuffer != g_SLMPBOID) {
                std::cout << "Warning: SLM PBO not bound, binding...\n";
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_SLMPBOID);
                error = glGetError();
                if (error != GL_NO_ERROR) {
                    std::cout << "Error: Failed to bind SLM PBO: " << error << "\n";
                    throw std::runtime_error("Failed to bind SLM PBO");
                }
            }

            // Verify SLM texture
            GLint boundTexture = 0;
            glGetIntegerv(GL_TEXTURE_BINDING_2D, &boundTexture);
            if (boundTexture != g_SLMTextureID) {
                std::cout << "Warning: SLM texture not bound, binding...\n";
                glBindTexture(GL_TEXTURE_2D, g_SLMTextureID);
                error = glGetError();
                if (error != GL_NO_ERROR) {
                    std::cout << "Error: Failed to bind SLM texture: " << error << "\n";
                    throw std::runtime_error("Failed to bind SLM texture");
                }
            }

            // Ensure OpenGL commands are complete before OpenCL operations
            glFinish();

            // Acquire the shared texture from OpenCL
            cl_int err = clEnqueueAcquireGLObjects(g_clQueue, 1, &g_clSharedTexture, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                throw std::runtime_error("Failed to acquire shared texture from OpenCL");
            }

            // Execute the kernel to update the shared texture
            err = clEnqueueNDRangeKernel(g_clQueue, g_clKernel, 2, nullptr, globalSLMSize, nullptr, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                clEnqueueReleaseGLObjects(g_clQueue, 1, &g_clSharedTexture, 0, nullptr, nullptr);
                throw std::runtime_error("Failed to execute kernel");
            }

            // Ensure kernel execution is complete
            err = clFinish(g_clQueue);
            if (err != CL_SUCCESS) {
                clEnqueueReleaseGLObjects(g_clQueue, 1, &g_clSharedTexture, 0, nullptr, nullptr);
                throw std::runtime_error("Failed to synchronize after kernel execution");
            }

            // Release the shared texture back to OpenGL
            err = clEnqueueReleaseGLObjects(g_clQueue, 1, &g_clSharedTexture, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                throw std::runtime_error("Failed to release shared texture to OpenGL");
            }

            // Ensure release is complete
            err = clFinish(g_clQueue);
            if (err != CL_SUCCESS) {
                std::cout << "Failed to synchronize after releasing GL objects: " << err << "\n";
                throw std::runtime_error("Failed to synchronize after releasing GL objects");
            }

            // Make sure OpenGL context is still current after OpenCL operations
            if (!wglMakeCurrent(g_sharedDC, g_sharedRC)) {
                throw std::runtime_error("Failed to make OpenGL context current after OpenCL operations");
            }

            // Update the texture from the PBO using LUMINANCE format
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, SLM_WIDTH, SLM_HEIGHT,
                           GL_LUMINANCE, GL_UNSIGNED_BYTE, nullptr);
            error = glGetError();
            if (error != GL_NO_ERROR) {
                throw std::runtime_error("Failed to update texture from PBO");
            }

            // Ensure all OpenGL commands are complete
            glFinish();

            // Unbind the PBO
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            glFinish();
        }

        // Unbind texture
        glBindTexture(GL_TEXTURE_2D, 0);
        glFinish();
        
        glFlush();  // Ensure texture update is complete
        error = glGetError();
        if (error != GL_NO_ERROR) {
            std::cout << "Error after flush, error: " << error << "\n";
            throw std::runtime_error("Failed to flush SLM texture update");
        }
    } catch (const std::exception& e) {
        std::cout << "Exception in UpdateSLMTexture: " << e.what() << std::endl;
        throw std::runtime_error("Failed to update SLM texture");
    } catch (...) {
        std::cout << "Unknown exception in UpdateSLMTexture\n";
        throw std::runtime_error("Unknown exception in UpdateSLMTexture");
    }
}

// Function to render the texture
void RenderSLM() {
    // Make sure we're using the shared context
    if (!wglMakeCurrent(g_sharedDC, g_sharedRC)) {
        std::cout << "Failed to make OpenGL context current for SLM rendering\n";
        return;
    }

    // Set viewport for SLM window
    glViewport(0, 0, SLM_WIDTH, SLM_HEIGHT);
    
    // Set up orthographic projection for SLM
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, SLM_WIDTH, SLM_HEIGHT, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_BLEND);
    glDisable(GL_COLOR_MATERIAL);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, g_SLMTextureID);
    glColor3f(1.0f, 1.0f, 1.0f);

    // Disable color material to prevent color interpolation
    // glDisable(GL_COLOR_MATERIAL);

    // Make sure texture environment is set to modulate or replace
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(0.0f, 0.0f);
    glTexCoord2f(1, 0); glVertex2f(static_cast<float>(SLM_WIDTH), 0.0f);
    glTexCoord2f(1, 1); glVertex2f(static_cast<float>(SLM_WIDTH), static_cast<float>(SLM_HEIGHT));
    glTexCoord2f(0, 1); glVertex2f(0.0f, static_cast<float>(SLM_HEIGHT));
    glEnd();

    glFinish();
    glFlush();
    SwapBuffers(g_sharedDC);

    glBindTexture(GL_TEXTURE_2D, 0);
    glFinish();
}

void UpdateCapturedTexture() {
    try {
        // Make sure we're using the shared context
        if (!wglMakeCurrent(GetDC(g_capturedWnd), g_sharedRC)) {
            throw std::runtime_error("Failed to make OpenGL context current for captured texture update");
        }

        // Verify context is current
        HGLRC currentContext = wglGetCurrentContext();
        if (currentContext != g_sharedRC) {
            std::cout << "Context verification failed. Current: " << currentContext << ", Expected: " << g_sharedRC << "\n";
            throw std::runtime_error("Context verification failed");
        }

        // Bind the texture in captured context
        glBindTexture(GL_TEXTURE_2D, g_capturedTextureID);
        GLenum error = glGetError();
        if (error != GL_NO_ERROR) {
            std::cout << "Error binding texture: " << error << "\n";
            throw std::runtime_error("Failed to bind texture");
        }

        // Get function pointer for glBindBuffer
        PFNGLBINDBUFFERPROC glBindBuffer = (PFNGLBINDBUFFERPROC)wglGetProcAddress("glBindBuffer");
        if (!glBindBuffer) {
            std::cout << "Failed to get glBindBuffer function pointer\n";
            throw std::runtime_error("Failed to get glBindBuffer function pointer");
        }

        // Get function pointer for glGetBufferParameteriv
        typedef void (APIENTRY *PFNGLGETBUFFERPARAMETERIVPROC) (GLenum target, GLenum pname, GLint* params);
        PFNGLGETBUFFERPARAMETERIVPROC glGetBufferParameteriv = (PFNGLGETBUFFERPARAMETERIVPROC)wglGetProcAddress("glGetBufferParameteriv");
        if (!glGetBufferParameteriv) {
            std::cout << "Failed to get glGetBufferParameteriv function pointer\n";
            throw std::runtime_error("Failed to get glGetBufferParameteriv function pointer");
        }
        
        // Verify PBO ID is valid
        if (g_capturedPBOID == 0) throw std::runtime_error("Invalid PBO ID (0)");

        // Unbind any existing PBO first
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        error = glGetError();
        if (error != GL_NO_ERROR) {
            std::cout << "Error unbinding existing PBO: " << error << "\n";
            throw std::runtime_error("Failed to unbind existing PBO");
        }

        // Bind the captured display PBO
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, g_capturedPBOID);
        error = glGetError();
        if (error != GL_NO_ERROR) {
            std::cout << "Error binding PBO for size check: " << error << "\n";
            throw std::runtime_error("Failed to bind PBO for size check");
        }
        // std::cout << "Successfully bound PBO\n";
        // std::cout << "Captured PBO ID: " << g_capturedPBOID << std::endl;

        // Ensure proper pixel storage alignment
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);

        // Verify PBO is properly bound
        GLint boundBuffer = 0;
        glGetIntegerv(GL_PIXEL_UNPACK_BUFFER_BINDING, &boundBuffer);
        if (boundBuffer != g_capturedPBOID) {
            std::cout << "PBO binding verification failed. Bound buffer: " << boundBuffer << ", Expected: " << g_capturedPBOID << "\n";
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            throw std::runtime_error("Failed to verify captured display PBO");
        }

        // Verify captured display texture
        GLint boundTexture = 0;
        glGetIntegerv(GL_TEXTURE_BINDING_2D, &boundTexture);
        if (boundTexture != g_capturedTextureID) {
            glBindTexture(GL_TEXTURE_2D, g_capturedTextureID);
            error = glGetError();
            if (error != GL_NO_ERROR) {
                std::cout << "Error: Failed to bind captured display texture: " << error << "\n";
                throw std::runtime_error("Failed to bind captured display texture");
            }
        }

        // Ensure all OpenGL operations are complete before OpenCL
        glFinish();

        // Acquire the shared buffer from OpenGL
        cl_int err = clEnqueueAcquireGLObjects(g_clQueue, 1, &g_clCombinedDisplay, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to acquire shared buffer from OpenGL");
        }

        // Wait for acquire to complete
        err = clFinish(g_clQueue);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to synchronize after acquire");
        }

        // Resize depth map for corner display mode if needed
        if (cornerDepthDisplay) {
            int cornerWidth = cameraWidth / 5;
            int cornerHeight = cameraHeight / 5;
            
            // Set resize kernel arguments
            err = clSetKernelArg(g_clResizeDepthMapKernel, 0, sizeof(cl_mem), &g_clSpatialFreqMap);
            if (err != CL_SUCCESS) throw std::runtime_error("Failed to set resize kernel arg 0");
            err = clSetKernelArg(g_clResizeDepthMapKernel, 1, sizeof(cl_mem), &g_clResizedDepthMap);
            if (err != CL_SUCCESS) throw std::runtime_error("Failed to set resize kernel arg 1");
            err = clSetKernelArg(g_clResizeDepthMapKernel, 2, sizeof(int), &cameraWidth);
            if (err != CL_SUCCESS) throw std::runtime_error("Failed to set resize kernel arg 2");
            err = clSetKernelArg(g_clResizeDepthMapKernel, 3, sizeof(int), &cameraHeight);
            if (err != CL_SUCCESS) throw std::runtime_error("Failed to set resize kernel arg 3");
            err = clSetKernelArg(g_clResizeDepthMapKernel, 4, sizeof(int), &cornerWidth);
            if (err != CL_SUCCESS) throw std::runtime_error("Failed to set resize kernel arg 4");
            err = clSetKernelArg(g_clResizeDepthMapKernel, 5, sizeof(int), &cornerHeight);
            if (err != CL_SUCCESS) throw std::runtime_error("Failed to set resize kernel arg 5");
            
            // Execute resize kernel
            size_t globalResizeSize[2] = {static_cast<size_t>(cornerWidth), static_cast<size_t>(cornerHeight)};
            err = clEnqueueNDRangeKernel(g_clQueue, g_clResizeDepthMapKernel, 2, nullptr, globalResizeSize, nullptr, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                clEnqueueReleaseGLObjects(g_clQueue, 1, &g_clCombinedDisplay, 0, nullptr, nullptr);
                throw std::runtime_error("Failed to execute resize depth map kernel");
            }
        }

        // Execute kernel to update the combined display
        size_t displayWidth = cornerDepthDisplay ? cameraWidth : cameraWidth * 2;
        size_t globalDisplaySize[2] = {static_cast<size_t>(displayWidth), static_cast<size_t>(cameraHeight)};
        err = clEnqueueNDRangeKernel(g_clQueue, g_clCombineImagesKernel, 2, nullptr, globalDisplaySize, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            clEnqueueReleaseGLObjects(g_clQueue, 1, &g_clCombinedDisplay, 0, nullptr, nullptr);
            throw std::runtime_error("Failed to execute combine images kernel");
        }

        // Apply text overlay using OpenCL kernel (much more efficient)
        if (svafStatus >= 1 && svafStatus <= 3 && g_clBlendTextOverlayKernel) {
            // Select the appropriate overlay buffer based on status
            cl_mem selectedOverlay = nullptr;
            switch (svafStatus) {
                case 1: selectedOverlay = g_clTextOverlayOff; break;
                case 2: selectedOverlay = g_clTextOverlayOnCenter; break;
                case 3: selectedOverlay = g_clTextOverlayOn; break;
                default: return; // No overlay for other statuses
            }
            
            if (selectedOverlay) {
                // Set kernel arguments
                err = clSetKernelArg(g_clBlendTextOverlayKernel, 0, sizeof(cl_mem), &g_clCombinedDisplay);
                if (err != CL_SUCCESS) {
                    std::cout << "Failed to set blend text overlay kernel arg 0: " << err << std::endl;
                } else {
                    err = clSetKernelArg(g_clBlendTextOverlayKernel, 1, sizeof(cl_mem), &selectedOverlay);
                    if (err != CL_SUCCESS) {
                        std::cout << "Failed to set blend text overlay kernel arg 1: " << err << std::endl;
                    } else {
                        err = clSetKernelArg(g_clBlendTextOverlayKernel, 2, sizeof(int), &cameraWidth);
                        if (err != CL_SUCCESS) {
                            std::cout << "Failed to set blend text overlay kernel arg 2: " << err << std::endl;
                        } else {
                            err = clSetKernelArg(g_clBlendTextOverlayKernel, 3, sizeof(int), &cameraHeight);
                            if (err != CL_SUCCESS) {
                                std::cout << "Failed to set blend text overlay kernel arg 3: " << err << std::endl;
                            } else {
                                // Execute the blending kernel
                                size_t globalSize[2] = {static_cast<size_t>(cameraWidth), static_cast<size_t>(cameraHeight)};
                                err = clEnqueueNDRangeKernel(g_clQueue, g_clBlendTextOverlayKernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
                                if (err != CL_SUCCESS) {
                                    std::cout << "Failed to execute blend text overlay kernel: " << err << std::endl;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Wait for kernel to complete
        err = clFinish(g_clQueue);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to synchronize after combine images kernel execution");
        }

        // Release the shared buffer back to OpenGL
        err = clEnqueueReleaseGLObjects(g_clQueue, 1, &g_clCombinedDisplay, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to release shared buffer to OpenGL");
        }

        // Update texture from PBO
        int textureWidth = cornerDepthDisplay ? cameraWidth : cameraWidth * 2;
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, textureWidth, cameraHeight,
                       GL_RGB, GL_UNSIGNED_BYTE, nullptr);
        error = glGetError();
        if (error != GL_NO_ERROR) {
            std::cout << "Failed to update captured texture: " << error << "\n";
            std::cout << "Texture dimensions: " << textureWidth << "x" << cameraHeight << "\n";
            throw std::runtime_error("Failed to update captured texture");
        }

        // Wait for texture update to complete
        glFinish();

        // Unbind resources
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glFinish();
        
        glFlush();  // Ensure texture update is complete
        error = glGetError();
        if (error != GL_NO_ERROR) {
            std::cout << "Error after flush, error: " << error << "\n";
            throw std::runtime_error("Failed to flush SLM texture update");
        }
    } catch (const std::exception& e) {
        std::cout << "Exception in UpdateCapturedTexture: " << e.what() << std::endl;
        throw std::runtime_error("Failed to update captured texture");
    }
}

int g_buffer[720 * 554 * 3 * 400];
int g_frame = 0;

void RenderCaptured() {
    // Make sure we're using the shared context
    if (!wglMakeCurrent(GetDC(g_capturedWnd), g_sharedRC)) {
        std::cout << "Failed to make OpenGL context current for captured display rendering\n";
        return;
    }

    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_BLEND);

    // Get window dimensions
    RECT clientRect;
    GetClientRect(g_capturedWnd, &clientRect);
    int windowWidth = clientRect.right - clientRect.left;
    int windowHeight = clientRect.bottom - clientRect.top;

    // Set viewport for captured window
    glViewport(0, 0, windowWidth, windowHeight);
    
    // Set up orthographic projection for captured display
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    // Use the actual image dimensions for the orthographic projection
    int projectionWidth = cornerDepthDisplay ? cameraWidth : cameraWidth * 2;
    glOrtho(0, static_cast<double>(projectionWidth), cameraHeight - cameraTopMargin - cameraBottomMargin, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_BLEND);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, g_capturedTextureID);
    glColor3f(1.0f, 1.0f, 1.0f);

    // Calculate texture coordinates for visible portion
    float topTexCoord = static_cast<float>(cameraTopMargin) / static_cast<float>(cameraHeight);
    float bottomTexCoord = 1.0f - (static_cast<float>(cameraBottomMargin) / static_cast<float>(cameraHeight));
    
    // Draw quad with texture - flipped both vertically and horizontally
    glBegin(GL_QUADS);
    // Bottom-left
    glTexCoord2f(1.0f, topTexCoord); 
    glVertex2f(0.0f, static_cast<float>(cameraHeight - cameraTopMargin - cameraBottomMargin));
    
    // Bottom-right
    glTexCoord2f(0.0f, topTexCoord); 
    glVertex2f(static_cast<float>(projectionWidth), static_cast<float>(cameraHeight - cameraTopMargin - cameraBottomMargin));
    
    // Top-right
    glTexCoord2f(0.0f, bottomTexCoord); 
    glVertex2f(static_cast<float>(projectionWidth), 0.0f);
    
    // Top-left
    glTexCoord2f(1.0f, bottomTexCoord); 
    glVertex2f(0.0f, 0.0f);
    glEnd();

    // Ensure rendering is complete before swap
    glFinish();
    glFlush();

    // Swap buffers
    SwapBuffers(GetDC(g_capturedWnd));

    if (0) {
        int N = 400;
        if (g_frame < N) {
            glPixelStorei(GL_PACK_ALIGNMENT, 4);
            glReadBuffer(GL_FRONT);
            glReadPixels(0, 0, 720, windowHeight, GL_RGB, GL_UNSIGNED_BYTE, g_buffer + g_frame * (554 * 720 * 3));
        }
        else if (g_frame == N) {
            char filename[256];

            for (int k = 0; k < N; k++) {
                sprintf_s(filename, "frames/frame_%04d.png", k);
                stbi_write_png(filename, 720, windowHeight, 3, g_buffer + k * (554 * 720 * 3), 720 * 3);
            }
        }

        g_frame++;
    }

    glBindTexture(GL_TEXTURE_2D, 0);
    glFinish();
}


// Function to load homography matrix from .npy file
cv::Mat LoadHomographyMatrix(const std::string& filepath) {
    try {
        cnpy::NpyArray arr = cnpy::npy_load(filepath);
        if (arr.shape.size() != 2 || arr.shape[0] != 3 || arr.shape[1] != 3) {
            throw std::runtime_error("Invalid homography matrix dimensions");
        }
        
        // Create a 3x3 matrix and copy the data
        cv::Mat H(3, 3, CV_64F);
        std::memcpy(H.data, arr.data.data(), 9 * sizeof(double));
        
        return H;
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading homography matrix: " << e.what() << std::endl;
        // Return identity matrix as fallback
        return cv::Mat::eye(3, 3, CV_64F);
    }
}


float depth = 0.0f;
int animate = 0;
int mode = 0;
bool showCheckerboard = false;
cv::Mat checkerboardImage;

// =-=-=-=-=-=-=-=-=-
// =-=- STREAMING -=-=-
// =-=-=-=-=-=-=-=-=-

float exposure = 47178.6f;
float gain = 24.96f; //28;

// Function to load checkerboard image
cv::Mat LoadCheckerboardImage() {
    std::cout << "Loading checkerboard image...\n";
    cv::Mat image = cv::imread("../../../../homography/checkerboard.png", cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cout << "Warning: Could not load checkerboard image from ../../../../homography/checkerboard.png\n";
        std::exit(EXIT_FAILURE);
    }
    return image;
}

// Function to save current RGB and depth images
void SaveCurrentImages() {
    try {
        if (!g_clQueue || !g_clCombinedDisplay) {
            std::cout << "OpenCL not properly initialized for image saving\n";
            return;
        }

        // Make sure we're using the shared context
        if (!wglMakeCurrent(g_sharedDC, g_sharedRC)) {
            std::cout << "Failed to make OpenGL context current for image saving\n";
            return;
        }

        // Ensure all OpenGL operations are complete
        glFinish();

        // Acquire the shared buffer from OpenGL
        cl_int err = clEnqueueAcquireGLObjects(g_clQueue, 1, &g_clCombinedDisplay, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to acquire combined display buffer from OpenGL: " << err << std::endl;
            return;
        }

        // Wait for acquire to complete
        err = clFinish(g_clQueue);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to synchronize after acquire: " << err << std::endl;
            clEnqueueReleaseGLObjects(g_clQueue, 1, &g_clCombinedDisplay, 0, nullptr, nullptr);
            return;
        }

        // Create buffer to read the combined display data
        int displayWidth = cornerDepthDisplay ? cameraWidth : cameraWidth * 2;
        size_t imageSize = displayWidth * cameraHeight * 3 * sizeof(uchar);
        std::vector<uchar> combinedData(imageSize);
        
        // Read the combined display data from OpenCL buffer
        err = clEnqueueReadBuffer(g_clQueue, g_clCombinedDisplay, CL_TRUE, 0, 
                                        imageSize, combinedData.data(), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to read combined display buffer: " << err << std::endl;
            clEnqueueReleaseGLObjects(g_clQueue, 1, &g_clCombinedDisplay, 0, nullptr, nullptr);
            return;
        }

        // Release the shared buffer back to OpenGL
        err = clEnqueueReleaseGLObjects(g_clQueue, 1, &g_clCombinedDisplay, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to release combined display buffer to OpenGL: " << err << std::endl;
        }

        // Wait for release to complete
        err = clFinish(g_clQueue);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to synchronize after release: " << err << std::endl;
        }

        // Create OpenCV Mat from the combined display data
        cv::Mat combinedImage(cameraHeight, displayWidth, CV_8UC3, combinedData.data());
        
        cv::Mat rgbImage, depthImage;
        
        // Side-by-side mode: RGB on right, depth on left
        // Calculate the split point (middle of the image)
        int splitX = cameraWidth;
        
        // Extract left half (depth image)
        cv::Rect leftRect(0, 0, splitX, cameraHeight);
        depthImage = combinedImage(leftRect).clone();
        
        // Extract right half (RGB image)
        cv::Rect rightRect(splitX, 0, splitX, cameraHeight);
        rgbImage = combinedImage(rightRect).clone();
        
        // Convert BGR to RGB
        cv::Mat rgbImageCorrected, depthImageCorrected;
        cv::cvtColor(rgbImage, rgbImageCorrected, cv::COLOR_BGR2RGB);
        cv::cvtColor(depthImage, depthImageCorrected, cv::COLOR_BGR2RGB);

        // Create saved_images directory if it doesn't exist
        CreateDirectoryA("saved_images", NULL);

        // Generate timestamp for unique filenames
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
        
        // Use safer approach for timestamp generation
        char timeBuffer[32];
        struct tm timeinfo;
        if (localtime_s(&timeinfo, &time_t) == 0) {
            strftime(timeBuffer, sizeof(timeBuffer), "%Y%m%d_%H%M%S", &timeinfo);
        } else {
            strcpy_s(timeBuffer, "unknown");
        }
        
        std::stringstream ss;
        ss << timeBuffer << "_" << std::setfill('0') << std::setw(3) << ms.count();
        std::string timestamp = ss.str();

        // Save RGB image (with correct RGB color order)
        std::string rgbFilename = "saved_images/rgb_" + timestamp + ".jpg";
        if (!cv::imwrite(rgbFilename, rgbImageCorrected)) {
            std::cout << "Failed to save RGB image: " << rgbFilename << std::endl;
        } else {
            std::cout << "RGB image saved: " << rgbFilename << std::endl;
        }

        // Save depth image
        std::string depthFilename = "saved_images/depth_" + timestamp + ".jpg";
        if (!cv::imwrite(depthFilename, depthImageCorrected)) {
            std::cout << "Failed to save depth image: " << depthFilename << std::endl;
        } else {
            std::cout << "Depth image saved: " << depthFilename << std::endl;
        }

        std::cout << "Images saved successfully!\n";
        
    } catch (const std::exception& e) {
        std::cout << "Exception in SaveCurrentImages: " << e.what() << std::endl;
    } catch (...) {
        std::cout << "Unknown exception in SaveCurrentImages\n";
    }
}

// streaming captured total image and current depth map in a 1x2 grid
// stops stream when ESC is pressed
// press x to turn off focusing
// press z to turn on focusing
void StreamSplitImages()
{
    // flag to track when an exception has been thrown
    bool exceptionThrown = false;
    cl_int err;
    try
    {
        std::cout << "Opening system...\n";
        // prepare example
        Arena::ISystem* pSystem = Arena::OpenSystem();
        std::cout << "System opened successfully\n";

        std::cout << "Updating devices...\n";
        pSystem->UpdateDevices(100);
        std::vector<Arena::DeviceInfo> devices = pSystem->GetDevices();
        std::cout << "Found " << devices.size() << " devices\n";

        if (devices.size() == 0)
        {
            std::cout << "\nNo camera connected\n";
            return;
        }

        std::cout << "Creating device...\n";
        // get first device
        Arena::IDevice* pDevice = pSystem->CreateDevice(devices[0]);
        std::cout << "Device created successfully\n";

        std::cout << "Configuring stream settings...\n";
        Arena::SetNodeValue<GenICam::gcstring>(pDevice->GetNodeMap(), "PixelFormat", "PolarizedAngles_0d_45d_90d_135d_BayerRG8");
        // Arena::SetNodeValue<bool>(pDevice->GetNodeMap(), "AcquisitionFrameRateEnable", true);
        // Arena::SetNodeValue<double>(pDevice->GetNodeMap(), "AcquisitionFrameRate", 21.020768519297064);
        Arena::SetNodeValue<GenICam::gcstring>(pDevice->GetNodeMap(), "ExposureAuto", "Off");
        Arena::SetNodeValue<double>(pDevice->GetNodeMap(), "ExposureTime", exposure);
        Arena::SetNodeValue<GenICam::gcstring>(pDevice->GetNodeMap(), "GainAuto", "Off");
        Arena::SetNodeValue<double>(pDevice->GetNodeMap(), "Gain", gain); // 18.0
        Arena::SetNodeValue<double>(pDevice->GetNodeMap(), "BlackLevel", 0.0);

        Arena::SetNodeValue<GenICam::gcstring>(pDevice->GetTLStreamNodeMap(), "StreamBufferHandlingMode", "NewestOnly");

        // enable stream auto negotiate packet size
        Arena::SetNodeValue<bool>(pDevice->GetTLStreamNodeMap(), "StreamAutoNegotiatePacketSize", true);

        // enable stream packet resend
        Arena::SetNodeValue<bool>(pDevice->GetTLStreamNodeMap(), "StreamPacketResendEnable", true);

        // Configure for high frame rates
        SetUpForRapidAcquisition(pDevice);

        // Try to stop any existing acquisition
        try {
            std::cout << "Attempting to stop any existing acquisition...\n";
            pDevice->StopStream();
            std::cout << "Existing acquisition stopped successfully\n";
        } catch (...) {
            std::cout << "No existing acquisition to stop\n";
        }

        std::cout << "Starting stream...\n";
        // start stream
        pDevice->StartStream();
        std::cout << "Stream started successfully\n";

        // Get first image to determine dimensions
        std::cout << "Getting first image for dimensions...\n";
        Arena::IImage* pFirstImage = pDevice->GetImage(2000);
        const int actualWidth = static_cast<int>(pFirstImage->GetWidth());
        const int actualHeight = static_cast<int>(pFirstImage->GetHeight());
        const int actualChannels = 3;  // Assuming RGB
        std::cout << "Camera dimensions: " << actualWidth << "x" << actualHeight << "\n";

        /* Initialize DPflow */
        std::cout << "Initializing DPflow...\n";
        int grid_spacing = 64; //64;      // Grid spacing for computation points
        int square_size = 40; //40;       // Size of each square region for correlation
        int disparity_range = 6;    // Disparity search range
        DPflow dpflow(g_clContext, g_clQueue, g_clDevice, window_size, lambda_reg, enableSuperpixels, compactness, numSuperpixels, green, grid_spacing, square_size, disparity_range);
        
        std::cout << "Allocating DPflow buffers...\n";
        try {
            dpflow.allocateBuffers(actualWidth, actualHeight, actualChannels, cameraTopMargin, cameraBottomMargin);
            std::cout << "DPflow buffers allocated successfully\n";
        } catch (const std::exception& e) {
            std::cout << "Failed to allocate DPflow buffers: " << e.what() << std::endl;
            return;
        }

        std::cout << "Setting DPflow kernel arguments...\n";
        try {
            dpflow.setKernelArgs(g_clLeftImage, g_clRightImage, g_clTopImage, g_clMask, g_clFlow, 0);
            std::cout << "DPflow kernel arguments set successfully\n";
        } catch (const std::exception& e) {
            std::cout << "Failed to set DPflow kernel arguments: " << e.what() << std::endl;
            return;
        }

        // Load homography matrix (H_CAM2SLM) from .npy file
        std::string homographyPath = "C:/Users/matth/Desktop/SVAF/homography/50mm_customdpsensor/result_HomographyMatrix_Cam.npy";
        std::cout << "Loading homography matrix from: " << homographyPath << std::endl;
        cv::Mat H_CAM2SLM = LoadHomographyMatrix(homographyPath);
        std::cout << "Loaded homography matrix:\n" << H_CAM2SLM << std::endl;
        
        std::cout << "Converting homography matrix to float...\n";
        cv::Mat H_CAM2SLM_32f;
        H_CAM2SLM.convertTo(H_CAM2SLM_32f, CV_32F);
        std::cout << "Converted matrix:\n" << H_CAM2SLM_32f << std::endl;
        
        std::cout << "Computing inverse of homography matrix...\n";
        cv::Mat H_CAM2SLM_32f_inv;
        cv::invert(H_CAM2SLM_32f, H_CAM2SLM_32f_inv);
        std::cout << "Inverse matrix:\n" << H_CAM2SLM_32f_inv << std::endl;       

        // Initialize spatial frequency map
        cv::Mat spatialFreqMap = cv::Mat::zeros(static_cast<int>(cameraHeight), static_cast<int>(cameraWidth), CV_32F);

        // Write data to buffers
        std::cout << "Writing spatial frequency map to buffer...\n";
        err = clEnqueueWriteBuffer(g_clQueue, g_clSpatialFreqMap, CL_TRUE, 0, 
                                  cameraWidth * cameraHeight * sizeof(float), 
                                  spatialFreqMap.data, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to write spatial frequency map to buffer");
        std::cout << "Spatial frequency map written successfully\n";
        clFinish(g_clQueue);

        std::cout << "Writing homography matrix to buffer...\n";
        err = clEnqueueWriteBuffer(g_clQueue, g_clHomography, CL_TRUE, 0, 
                                  9 * sizeof(float), H_CAM2SLM_32f_inv.data, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to write homography matrix to buffer");
        std::cout << "Homography matrix written successfully\n";
        clFinish(g_clQueue);

        // Create and initialize WarpPerspective object
        std::cout << "Creating WarpPerspective object...\n";
        g_warp = new WarpPerspective(g_clContext, g_clQueue, g_clDevice);
        std::cout << "WarpPerspective object created successfully\n";
        
        std::cout << "Allocating WarpPerspective buffers...\n";
        g_warp->allocateBuffers(static_cast<int>(cameraWidth), static_cast<int>(cameraHeight), SLM_WIDTH, SLM_HEIGHT);
        std::cout << "WarpPerspective buffers allocated successfully\n";
        
        std::cout << "Setting WarpPerspective kernel arguments...\n";
        g_warp->setKernelArgs(g_clSpatialFreqMap, g_clWarpedFreqMap, g_clHomography, 0);
        std::cout << "WarpPerspective kernel arguments set successfully\n";

        // Execute period map kernel
        std::cout << "Executing period map kernel...\n";
        err = clEnqueueNDRangeKernel(g_clQueue, g_clPeriodMapKernel, 2, nullptr, globalSLMSize, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to execute period map kernel");
        std::cout << "Period map kernel executed successfully\n";

        // Add synchronization to ensure kernel completion
        err = clFinish(g_clQueue);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to synchronize after period map kernel");

        // Create black pattern for when 'x' is pressed
        cv::Mat blackPattern = cv::Mat::zeros(SLM_HEIGHT, SLM_WIDTH, CV_8U);
        
        // Flag to track which pattern to display
        bool showBlackPattern = false;

        // Create SLM pattern
        cv::Mat slmPattern = cv::Mat::zeros(SLM_HEIGHT, SLM_WIDTH, CV_8U);

        // Requeue the first image
        pDevice->RequeueBuffer(pFirstImage);

        // FPS calculation variables
        auto startTime = std::chrono::high_resolution_clock::now();
        int frameCount = 0;
        double fps = 0.0;

        bool running = true;
        while (running)
        {
            try {
                // Handle Windows messages
                MSG msg;
                while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                    if (msg.hwnd == g_capturedWnd) {  // Only process messages for the grid window
                        if (msg.message == WM_KEYDOWN) {
                            if (msg.wParam == VK_ESCAPE) {
                                std::cout << "ESC key pressed, preparing to exit...\n";
                                running = false;
                                break;
                            }
                            else if (msg.wParam == '1') {
                                showBlackPattern = true;
                                svafStatus = 1;  // SVAF OFF
                                std::cout << "Showing black pattern - SVAF OFF\n";
                            }
                            else if (msg.wParam == '2') {
                                int flag;
                                flag = 0;
                                err = clSetKernelArg(g_clKernel, 5, sizeof(int), &flag);
                                dpflow.setKernelArgs(g_clLeftImage, g_clRightImage, g_clTopImage, g_clMask, g_clFlow, 0);

                                showBlackPattern = false;
                                svafStatus = 2;  // SVAF ON
                                std::cout << "Showing SLM pattern - SVAF ON\n";
                            }
                            else if (msg.wParam == '3') {
                                int flag;
                                flag = 0;
                                err = clSetKernelArg(g_clKernel, 5, sizeof(int), &flag);
                                dpflow.setKernelArgs(g_clLeftImage, g_clRightImage, g_clTopImage, g_clMask, g_clFlow, 1);

                                showBlackPattern = false;
                                svafStatus = 3;  // SVAF ON
                                std::cout << "Showing SLM pattern - SVAF ON\n";
                            }
                            else if (msg.wParam == '4') {
                                int flag;
                                flag = 1;
                                showBlackPattern = false;
                                err = clSetKernelArg(g_clKernel, 5, sizeof(int), &flag);
                                animate = 1;

                                depth = 0.0f;
                            }
                            else if (msg.wParam == '5') {
                                int flag;
                                flag = 2;
                                showBlackPattern = false;
                                err = clSetKernelArg(g_clKernel, 5, sizeof(int), &flag);
                                animate = 1;

                                depth = 0.0f;
                            }
                            else if (msg.wParam == '7') {
                                exposure -= 100.0f;
                                exposure = std::max(100.0f, std::min(47186.28f, exposure));
                                Arena::SetNodeValue<double>(pDevice->GetNodeMap(), "ExposureTime", exposure);
                            }
                            else if (msg.wParam == '8') {
                                exposure += 100.0f;
                                exposure = std::max(100.0f, std::min(47186.28f, exposure));
                                Arena::SetNodeValue<double>(pDevice->GetNodeMap(), "ExposureTime", exposure);
                            }
                            else if (msg.wParam == '9') {
                                gain -= 1.0f;
                                gain = std::max(0.0f, std::min(48.0f, gain));
                                Arena::SetNodeValue<double>(pDevice->GetNodeMap(), "Gain", gain);
                                //depth = depth + 0.01f;
                            }
                            else if (msg.wParam == '0') {
                                gain += 1.0f;
                                Arena::SetNodeValue<double>(pDevice->GetNodeMap(), "Gain", gain);
                                //depth = depth - 0.01f;
                            }
                            else if (msg.wParam == 'A' || msg.wParam == 'a') {
                                animate = 1-animate;
                            }
                            else if (msg.wParam == 'M' || msg.wParam == 'm') {
                                mode = (mode + 1) % 2;
                                err = clSetKernelArg(g_clCombineImagesKernel, 12, sizeof(int), &mode);
                            }
                            else if (msg.wParam == 'S' || msg.wParam == 's') {
                                std::cout << "Saving current RGB and depth images...\n";
                                SaveCurrentImages();
                            }
                            else if (msg.wParam == 'C' || msg.wParam == 'c') {
                                showCheckerboard = !showCheckerboard;
                                if (showCheckerboard) {
                                    std::cout << "Loading and showing checkerboard pattern on SLM\n";
                                    checkerboardImage = LoadCheckerboardImage();
                                } else {
                                    std::cout << "Returning to normal SLM pattern\n";
                                    // Clear the checkerboard image from memory
                                    checkerboardImage.release();
                                }
                            }
                            else if (msg.wParam == 'D' || msg.wParam == 'd') {
                                std::cout << "Display mode toggle requested. Current mode: " << (cornerDepthDisplay ? "corner depth" : "side-by-side") << "\n";
                                std::cout << "To change display mode, modify the 'cornerDepthDisplay' flag at the top of svafstream.cpp and restart the application.\n";
                                std::cout << "Current setting: cornerDepthDisplay = " << (cornerDepthDisplay ? "true" : "false") << "\n";
                            }
                        }
                    }

                    TranslateMessage(&msg);
                    DispatchMessage(&msg);
                }

                if (depth > 0.5f) {
                    depth = -0.5f;
                }
                if (depth < -0.5f) {
                    depth = 0.5f;
                }

                if (animate) {
                    depth = depth + 0.02f;
                }

                // std::cout << "Depth: " << depth << std::endl;
                    
                err = clSetKernelArg(g_clKernel, 6, sizeof(float), &depth);
 
                if (!running) {
                    std::cout << "Exiting main loop...\n";
                    break;
                }

                // get image
                err = clFinish(g_clQueue);
                Arena::IImage* pImage = pDevice->GetImage(2000);
                if (!pImage) {
                    std::cout << "Failed to get image from camera\n";
                    continue;
                }
                clFinish(g_clQueue);


                // Split into channels
                std::vector<Arena::IImage*> splitImages = Arena::ImageFactory::SplitChannels(pImage);
                if (splitImages.size() != 4) {
                    std::cout << "Error: Expected 4 channels but got " << splitImages.size() << "\n";
                    pDevice->RequeueBuffer(pImage);
                    continue;
                }

                // Get dimensions
                size_t width = splitImages[0]->GetWidth();
                size_t height = splitImages[0]->GetHeight();

                // Create OpenCV Mats for each channel
                std::vector<Arena::IImage*> convertedImages;  // Store converted images to prevent premature destruction
                
                // Write left image (i=0) to buffer
                auto pConvertedLeft = Arena::ImageFactory::Convert(splitImages[0], RGB8);
                if (!pConvertedLeft) {
                    throw std::runtime_error("Error: Failed to convert channel " + std::to_string(0) + " to RGB8");
                }
                convertedImages.push_back(pConvertedLeft);  // Store to prevent premature destruction
                const void* leftImageData = pConvertedLeft->GetData();
                if (!leftImageData) {
                    throw std::runtime_error("Error: No data available for left image");
                }

                // Copy data to left image temporary buffer
                size_t imageSize = width * height * cameraChannels * sizeof(uchar);
                err = clEnqueueWriteBuffer(g_clQueue, g_clTempLeftBuffer, CL_TRUE, 0,
                                         imageSize,
                                         leftImageData, 0, nullptr, nullptr);
                if (err != CL_SUCCESS) {
                    std::cout << "Failed to write to temporary left image buffer. Error: " << err << std::endl;
                    throw std::runtime_error("Failed to write to temporary left image buffer");
                }

                // Execute conversion kernel, result saved in g_clLeftImage
                err = clEnqueueNDRangeKernel(g_clQueue, g_clConvertLeftImageToFloatKernel, 2, nullptr, globalCameraSize, nullptr, 0, nullptr, nullptr);
                if (err != CL_SUCCESS) throw std::runtime_error("Failed to execute convert kernel");

                // Write right image (i=2) to buffer
                auto pConvertedRight = Arena::ImageFactory::Convert(splitImages[2], RGB8);
                if (!pConvertedRight) {
                    throw std::runtime_error("Error: Failed to convert channel " + std::to_string(2) + " to RGB8");
                }
                convertedImages.push_back(pConvertedRight);  // Store to prevent premature destruction
                const void* rightImageData = pConvertedRight->GetData();
                if (!rightImageData) {
                    throw std::runtime_error("Error: No data available for right image");
                }

                // Copy data to right image temporary buffer
                err = clEnqueueWriteBuffer(g_clQueue, g_clTempRightBuffer, CL_TRUE, 0,
                                         imageSize,
                                         rightImageData, 0, nullptr, nullptr);
                if (err != CL_SUCCESS) {
                    std::cout << "Failed to write to temporary right image buffer. Error: " << err << std::endl;
                    throw std::runtime_error("Failed to write to temporary right image buffer");
                }

                // Execute conversion kernel, result saved in g_clRightImage
                err = clEnqueueNDRangeKernel(g_clQueue, g_clConvertRightImageToFloatKernel, 2, nullptr, globalCameraSize, nullptr, 0, nullptr, nullptr);
                if (err != CL_SUCCESS) throw std::runtime_error("Failed to execute convert kernel");

                // Write total image (i=1) to buffer
                auto pConvertedTop = Arena::ImageFactory::Convert(splitImages[1], RGB8);
                if (!pConvertedTop) {
                    throw std::runtime_error("Error: Failed to convert channel " + std::to_string(3) + " to RGB8");
                }
                convertedImages.push_back(pConvertedTop);  // Store to prevent premature destruction
                const void* topImageData = pConvertedTop->GetData();
                if (!topImageData) {
                    throw std::runtime_error("Error: No data available for total image");
                }

                // Copy data to total image temporary buffer
                err = clEnqueueWriteBuffer(g_clQueue, g_clTempTopBuffer, CL_TRUE, 0,
                    imageSize,
                    topImageData, 0, nullptr, nullptr);
                if (err != CL_SUCCESS) {
                    std::cout << "Failed to write to temporary total image buffer. Error: " << err << std::endl;
                    throw std::runtime_error("Failed to write to temporary total image buffer");
                }

                // Execute conversion kernel, result saved in g_clTopImage
                err = clEnqueueNDRangeKernel(g_clQueue, g_clConvertTopImageToFloatKernel, 2, nullptr, globalCameraSize, nullptr, 0, nullptr, nullptr);
                if (err != CL_SUCCESS) throw std::runtime_error("Failed to execute convert kernel");

                // Write total image (i=3) to buffer
                auto pConvertedBottom = Arena::ImageFactory::Convert(splitImages[3], RGB8);
                if (!pConvertedBottom) {
                    throw std::runtime_error("Error: Failed to convert channel " + std::to_string(3) + " to RGB8");
                }
                convertedImages.push_back(pConvertedBottom);  // Store to prevent premature destruction
                const void* bottomImageData = pConvertedBottom->GetData();
                if (!bottomImageData) {
                    throw std::runtime_error("Error: No data available for total image");
                }

                // Copy data to total image temporary buffer
                err = clEnqueueWriteBuffer(g_clQueue, g_clTempBottomBuffer, CL_TRUE, 0,
                                         imageSize,
                                         bottomImageData, 0, nullptr, nullptr);
                if (err != CL_SUCCESS) {
                    std::cout << "Failed to write to temporary total image buffer. Error: " << err << std::endl;
                    throw std::runtime_error("Failed to write to temporary total image buffer");
                }

                // Execute conversion kernel, result saved in g_clTopImage
                err = clEnqueueNDRangeKernel(g_clQueue, g_clConvertBottomImageToFloatKernel, 2, nullptr, globalCameraSize, nullptr, 0, nullptr, nullptr);
                if (err != CL_SUCCESS) throw std::runtime_error("Failed to execute convert kernel");

                // Wait for all transfers to complete
                err = clFinish(g_clQueue);
                if (err != CL_SUCCESS) throw std::runtime_error("Failed to synchronize g_clQueue after image transfers");

                // Switch to captured display context for rendering
                if (!wglMakeCurrent(g_sharedDC, g_sharedRC)) {
                    std::cout << "Error: Failed to make captured display OpenGL context current\n";
                    throw std::runtime_error("Failed to make captured display OpenGL context current");
                }
                // Display the captured image and depth map using OpenGL
                try {
                    clFinish(g_clQueue);
                    glFinish();
                    UpdateCapturedTexture();
                    glFinish();
                    RenderCaptured();
                    glFinish();
                } catch (const std::exception& e) {
                    std::cout << "Error updating/rendering captured display: " << e.what() << "\n";
                    continue;
                }

                if (!showBlackPattern) {
                    /* Compute the optical flow*/

                    dpflow.compute_optical_flow_x(); // result saved in g_clFlow

                    // turn on global autofocus
                    

                    // Synchronize captured context before updating spatial frequency map
                    err = clFinish(g_clQueue);
                    if (err != CL_SUCCESS) throw std::runtime_error("Failed to synchronize captured context after optical flow");

                    if ((frameCount % 2) == 0) {
                        /* Update the spatial frequency map*/
                        UpdateSpatialFreqMapWithScalar(); // result saved in g_clSpatialFreqMap
                    }

                    // Synchronize captured context after spatial frequency map update
                    err = clFinish(g_clQueue);
                    if (err != CL_SUCCESS) throw std::runtime_error("Failed to synchronize captured context after spatial frequency map update");

                    /* Warp frequency map from camera to SLM coordinates and save to g_clWarpedFreqMap */
                    g_warp->compute(); // result saved in g_clWarpedFreqMap

                    // Synchronize SLM context after warping
                    err = clFinish(g_clQueue);
                    if (err != CL_SUCCESS) throw std::runtime_error("Failed to synchronize SLM context after warping");

                    /* Convert warped frequency map g_clWarpedFreqMap to period map g_clPeriodMap */
                    if (!g_clQueue || !g_clPeriodMapKernel || !g_clWarpedFreqMap || !g_clPeriodMap) {
                        throw std::runtime_error("Invalid OpenCL resources");
                    }

                    err = clEnqueueNDRangeKernel(g_clQueue, g_clPeriodMapKernel, 2, nullptr, globalSLMSize, nullptr, 0, nullptr, nullptr); // result saved in g_clPeriodMap
                    if (err != CL_SUCCESS) throw std::runtime_error("Failed to execute period map kernel");

                    err = clFinish(g_clQueue);
                    if (err != CL_SUCCESS) throw std::runtime_error("Failed to synchronize CL queue");

                    // Complete all commands in g_clQueue before displaying opengl window
                    glFinish();

                } else {
                    // Define zero value for buffer clearing
                    const float zero = 0.0f;
                    const uchar zero_uchar = 0;
                    
                    // Set g_clWarpedFreqMap to zero
                    err = clEnqueueFillBuffer(g_clQueue, g_clWarpedFreqMap, &zero, sizeof(float),
                                            0, SLM_WIDTH * SLM_HEIGHT * sizeof(float), 0, nullptr, nullptr);
                    if (err != CL_SUCCESS) throw std::runtime_error("Failed to zero warped frequency map buffer");

                    // Set g_clPeriodMap to zero
                    err = clEnqueueFillBuffer(g_clQueue, g_clPeriodMap, &zero, sizeof(float),
                                            0, SLM_WIDTH * SLM_HEIGHT * sizeof(float), 0, nullptr, nullptr);
                    if (err != CL_SUCCESS) throw std::runtime_error("Failed to zero period map buffer");

                    // Set spatial frequency map superpixels to zero
                    err = clEnqueueFillBuffer(g_clQueue, g_clSpatialFreqMapSuperpixels, &zero, sizeof(float), 
                                            0, cameraWidth * cameraHeight * sizeof(float), 0, nullptr, nullptr);
                    if (err != CL_SUCCESS) throw std::runtime_error("Failed to zero spatial frequency map superpixels buffer");

                    // Set spatial frequency map to zero
                    err = clEnqueueFillBuffer(g_clQueue, g_clSpatialFreqMap, &zero, sizeof(float),
                                            0, cameraWidth * cameraHeight * sizeof(float), 0, nullptr, nullptr);
                    if (err != CL_SUCCESS) throw std::runtime_error("Failed to zero spatial frequency map buffer");

                    // Set flow to zero
                    err = clEnqueueFillBuffer(g_clQueue, g_clFlow, &zero, sizeof(float),
                                            0, cameraWidth * cameraHeight * sizeof(float), 0, nullptr, nullptr);
                    if (err != CL_SUCCESS) throw std::runtime_error("Failed to zero flow buffer");

                    // Ensure all operations complete
                    err = clFinish(g_clQueue);
                    if (err != CL_SUCCESS) throw std::runtime_error("Failed to synchronize CL queue after zeroing buffers");
                    glFinish();
                }
                
                // Update and display SLM pattern using OpenGL
                try {
                    clFinish(g_clQueue);
                    glFinish();
                    if (showCheckerboard) {
                        UpdateSLMTexture(checkerboardImage, true);  // Use direct CPU upload for checkerboard
                    } else {
                    UpdateSLMTexture(showBlackPattern ? blackPattern : slmPattern, showBlackPattern);
                    }
                    glFinish();
                    RenderSLM();
                    glFinish();
                } catch (const std::exception& e) {
                    std::cout << "Error updating/rendering SLM: " << e.what() << "\n";
                    continue;
                }

                // Calculate FPS every second and display current estimated FPS every frame
                frameCount++;
                auto currentTime = std::chrono::high_resolution_clock::now();
                auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime).count() / 1000.0;
                if (elapsedTime >= 1.0) {
                    fps = frameCount / elapsedTime;
                    frameCount = 0;
                    startTime = currentTime;
                } else {
                    // Show current estimate based on partial second
                    fps = frameCount / elapsedTime;
                }
                std::string fpsText = "FPS: " + std::to_string(static_cast<double>(fps)).substr(0, std::to_string(static_cast<double>(fps)).find(".") + 3);
                std::cout << fpsText << std::endl;

                // Clean up converted images after we're done with them
                for (auto pImage : convertedImages) {
                    Arena::ImageFactory::Destroy(pImage);
                }
                convertedImages.clear();

                // Clean up split images
                for (auto pImage : splitImages) {
                    Arena::ImageFactory::Destroy(pImage);
                }

                // Requeue the buffer
                pDevice->RequeueBuffer(pImage);
                clFinish(g_clQueue);
                glFinish();
            }
            catch (const std::exception& e) {
                std::cout << "Error in main loop: " << e.what() << "\n";
                running = false;
            }
        }

        std::cout << "Starting cleanup...\n";
        
        // First clean up objects that use OpenCL while contexts are still valid
        if (g_warp) {
            std::cout << "Cleaning up WarpPerspective object...\n";
            delete g_warp;
            g_warp = nullptr;
            std::cout << "WarpPerspective cleanup complete\n";
        }
        
        // Then clean up OpenCL resources while OpenGL contexts are still valid
        std::cout << "Starting OpenCL cleanup...\n";
        CleanupOpenCL();
        std::cout << "OpenCL cleanup complete\n";
        
        // Then clean up OpenGL resources
        std::cout << "Starting main OpenGL cleanup...\n";
        CleanupOpenGL();
        std::cout << "Main OpenGL cleanup complete\n";
        
        std::cout << "Starting window cleanup...\n";
        // Store window handles locally to prevent double destruction
        HWND localCapturedWnd = g_capturedWnd;
        HWND localSLMWnd = g_SLMWnd;
        
        // Clear global handles first
        std::cout << "Clearing global window handles...\n";
        g_capturedWnd = nullptr;
        g_SLMWnd = nullptr;
        
        // Destroy windows if they still exist
        if (localCapturedWnd && IsWindow(localCapturedWnd)) {
            std::cout << "Destroying captured window...\n";
            DestroyWindow(localCapturedWnd);
            std::cout << "Captured window destroyed\n";
        }
        
        if (localSLMWnd && IsWindow(localSLMWnd)) {
            std::cout << "Destroying SLM window...\n";
            DestroyWindow(localSLMWnd);
            std::cout << "SLM window destroyed\n";
        }
        std::cout << "Window cleanup complete\n";

        std::cout << "Stopping stream...\n";
        // stop stream
        pDevice->StopStream();
        std::cout << "Stream stopped\n";

        std::cout << "Cleaning up device and system...\n";
        // clean up example
        pSystem->DestroyDevice(pDevice);
        Arena::CloseSystem(pSystem);

        // Close OpenCV window
        cv::destroyAllWindows();
    }
    catch (GenICam::GenericException& ge)
    {
        std::cout << "\nGenICam exception thrown: " << ge.what() << "\n";
        exceptionThrown = true;
    }
    catch (std::exception& ex)
    {
        std::cout << "\nStandard exception thrown: " << ex.what() << "\n";
        exceptionThrown = true;
    }
    catch (...)
    {
        std::cout << "\nUnexpected exception thrown\n";
        exceptionThrown = true;
    }
}

int main()
{
    std::cout << "SVAF Streaming\n\n";
    std::cout << "Controls:\n";
    std::cout << "  ESC - Stop streaming\n";
    std::cout << "  1   - Show black pattern\n";
    std::cout << "  2   - Show SLM pattern (only center square area)\n";
    std::cout << "  3   - Show SLM pattern (entire area)\n";
    std::cout << "  4   - Animated sinusoidal focus pattern\n";
    std::cout << "  5   - Animated center pattern\n";
    std::cout << "  A   - Toggle animation\n";
    std::cout << "  M   - Toggle display mode\n";
    std::cout << "  S   - Save current RGB and depth images\n";
    std::cout << "  C   - Toggle checkerboard pattern on SLM\n";
    std::cout << "  D   - Show current display layout setting\n";
    std::cout << "  7/8 - Decrease/Increase exposure\n";
    std::cout << "  9/0 - Decrease/Increase gain\n\n";
    std::cout << "Current display mode: " << (cornerDepthDisplay ? "Corner depth display" : "Side-by-side display") << "\n";
    std::cout << "To change display mode, modify 'cornerDepthDisplay' flag in svafstream.cpp and recompile.\n\n";

    // // Get user inputs
    // std::string flow_scalar_learning_rate_string;
    // std::cout << "Enter flow scalar learning rate (default 0.03): ";
    // std::cin >> flow_scalar_learning_rate_string;
    // flow_scalar_learning_rate = std::stof(flow_scalar_learning_rate_string);

    // std::string enableSuperpixels_string;
    // std::string devignet_string;
    // std::cout << "Enable superpixel averaging for optical flow? (0 for box filtering / 1 for superpixels): ";
    // std::cin >> enableSuperpixels_string;
    // std::cout << "Apply devignet map? (0/1): ";
    // std::cin >> devignet_string;
    // if (enableSuperpixels_string == "1") {
    //     enableSuperpixels = true;
    // } else {
    //     enableSuperpixels = false;
    // }
    // if (devignet_string == "1") {
    //     devignet = true;
    // } else {
    //     devignet = false;
    // }

    if (!InitOpenGL()) {
        std::cout << "Failed to initialize OpenGL\n";
        return -1;
    }

    std::cout << "Initializing OpenCL...\n";
    // Initialize OpenCL with OpenGL sharing
    if (!InitOpenCL()) {
        std::cout << "Failed to initialize OpenCL\n";
        CleanupOpenGL();
        DestroyWindow(g_capturedWnd);
        DestroyWindow(g_SLMWnd);
        return -1;
    }
    
    // Create text overlays after OpenCL is initialized
    CreateTextOverlays();
    
    // Create OpenCL buffers for all three text overlays
    if (g_clContext && g_clQueue) {
        cl_int err;
        size_t overlaySize = static_cast<int>(cameraHeight) * static_cast<int>(cameraWidth) * 4 * sizeof(uchar); // BGRA
        
        // Create buffer for "SVAF OFF" overlay
        g_clTextOverlayOff = clCreateBuffer(g_clContext, CL_MEM_READ_ONLY, overlaySize, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create SVAF OFF overlay buffer: " << err << std::endl;
            return -1;
        }
        
        // Create buffer for "SVAF ON (CENTER ONLY)" overlay
        g_clTextOverlayOnCenter = clCreateBuffer(g_clContext, CL_MEM_READ_ONLY, overlaySize, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create SVAF ON CENTER overlay buffer: " << err << std::endl;
            return -1;
        }
        
        // Create buffer for "SVAF ON" overlay
        g_clTextOverlayOn = clCreateBuffer(g_clContext, CL_MEM_READ_ONLY, overlaySize, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to create SVAF ON overlay buffer: " << err << std::endl;
            return -1;
        }
        
        // Upload all three overlays to their respective buffers
        err = clEnqueueWriteBuffer(g_clQueue, g_clTextOverlayOff, CL_TRUE, 0, overlaySize, svafOffOverlay.data, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to upload SVAF OFF overlay: " << err << std::endl;
            return -1;
        }
        
        err = clEnqueueWriteBuffer(g_clQueue, g_clTextOverlayOnCenter, CL_TRUE, 0, overlaySize, svafOnCenterOverlay.data, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to upload SVAF ON CENTER overlay: " << err << std::endl;
            return -1;
        }
        
        err = clEnqueueWriteBuffer(g_clQueue, g_clTextOverlayOn, CL_TRUE, 0, overlaySize, svafOnOverlay.data, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to upload SVAF ON overlay: " << err << std::endl;
            return -1;
        } else {
            std::cout << "All text overlays uploaded to OpenCL buffers successfully\n";
        }
    }

    std::cout << "Starting streaming...\n";
    try {
        StreamSplitImages();
    } catch (const std::exception& e) {
        std::cout << "Exception in StreamSplitImages: " << e.what() << std::endl;
    } catch (...) {
        std::cout << "Unknown exception in StreamSplitImages\n";
    }
    
    // Cleanup
    if (g_SLMWnd && IsWindow(g_SLMWnd)) {
        DestroyWindow(g_SLMWnd);
        g_SLMWnd = nullptr;
    }
    
    if (g_capturedWnd && IsWindow(g_capturedWnd)) {
        DestroyWindow(g_capturedWnd);
        g_capturedWnd = nullptr;
    }

    // Clean up shared context and DC
    if (g_sharedRC) {
        wglDeleteContext(g_sharedRC);
    }
    if (g_sharedDC) {
        ReleaseDC(NULL, g_sharedDC);
    }
    
    std::cout << "\nAutofocus streaming complete\n";
    std::cout << "Press Enter to exit...\n";
    std::cin.get();
    return 0;
}
