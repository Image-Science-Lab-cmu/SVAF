# Spatially-Varying Autofocus [ICCV 2025]

[Yingsi Qin](https://yingsiqin.github.io/), [Aswin C. Sankaranarayanan](https://users.ece.cmu.edu/~saswin/), [Matthew O'Toole](https://www.cs.cmu.edu/~motoole2/)

> **Please read:** This repository contains the code for our real-time SVAF prototype that uses a custom-built dual-pixel (DP) camera based on a polarized machine vision sensor. This is different from the original prototype with the Canon EOS R10 sensor whose bottleneck was the 0.3 FPS in reading dual-pixel (DP) images. To overcome the lack of off-the-shelf solutions for streaming DP images, we modified a machine vision sensor to enable capturing, reading, and processing DP images at 21 FPS. If you are using a Canon DP sensor instead and would like to extract dual-pixel images from Canon's Dual Pixel RAW image files, please refer to the [Extracting the Dual-Pixel Views from Canon RAW Files](#extracting-the-dual-pixel-views-from-canon-raw-files) section below.

This repository contains both (1) the code for running our autofocus algorithms, and (2) the PDAF hardware code for the real-time prototype, used for [Spatially-Varying Autofocus](https://imaging.cs.cmu.edu/svaf/). 

Visit the links below for more information:\
 [[Paper](https://imaging.cs.cmu.edu/svaf/static/pdfs/Spatially_Varying_Autofocus.pdf)] [[Supplemental PDF](https://imaging.cs.cmu.edu/svaf/static/pdfs/Spatially_Varying_Autofocus-supp.pdf)] [[Project Website](https://imaging.cs.cmu.edu/svaf/)] [[5-Minute Video](https://www.youtube.com/watch?v=WNPkUB9o2Fo)] [[Poster](https://imaging.cs.cmu.edu/svaf/static/pdfs/Spatially_Varying_Autofocus-poster.pdf)]

## Getting Started

All code contains both (1) the interface to control the sensor [Lucid Vision PHX050S1-P/Q](https://thinklucid.com/product/phoenix-5-0-mp-polarization-model-imx264mzrmyr/) and the spatial light modulator [HOLOEYE GAEA2 Phase-Only SLM](https://holoeye.com/products/spatial-light-modulators/gaea-2-phase-only/), and (2) the algorithm to produce the depth map that we use to control focus. 

## Python program

This program runs vanilla optical flow to perform spatially-varying phase-based autofocus (PDAF). 

### Dependencies
If you are using a Lucid Vision polarized camera, please download Arena Python API:
https://dce9ugryut4ao.cloudfront.net/arena_api-2.7.1-py3-none-any.zip

Install additional dependencies:
```
pip install -r requirements.txt
```

### Run the python program
How to run:
1. Please make sure that the camera is turned off in ArenaView before running the python program. The program will have GenICam errors if the camera is already turned on in ArenaView.
2. Run the following commands.
```
cd code
python svaf_vanilla.py
```

## C++ program
This program performs Spatially-Varying PDAF at 21 FPS on our real-time prototype. 

An example screen recording of running the program to autofocus a dynamic scene:
![](static/videos/dynamic_box_1minute.gif)

Please see our [website](https://imaging.cs.cmu.edu/svaf/#dynamic) for a full resolution recording and more details.

### Dependencies
Go to Lucid Vision Labs website:
https://thinklucid.com/downloads-hub

Donwload Arena SDK for Windows 10/11:
https://dce9ugryut4ao.cloudfront.net/ArenaSDK_v1.0.49.3.exe

Check if Arena SDK is installed in the following path:
```
C:\Program Files\Lucid Vision Labs\Arena SDK\include
```

Check if OpenCV is installed in the following path:
```
C:\path_to_folder\SVAF\dependencies\opencv\build\include
```

Please make sure to replace `path_to_folder` in `.vcxproj` files with your path that contains this repository.

### Compile the C++ program
```
cd cpp_program
msbuild svafstream.sln //p:Configuration=Release //p:Platform=x64
```

### Run the C++ program
How to run:
1. Please make sure that the camera is turned off in ArenaView before running the CPP program. The program will have GenICam errors if the camera is already turned on in ArenaView.
2. Make sure to turn on the lights!
3. Run the following commands.
```
cd cpp_program
./build/svafstream/x64/Release/svafstream.exe
```

## Aligning the Optical System

### Homography

We provide code to perform homography between the sensor and the spatial light modulator. The entire walkthrough is detailed in `homography/homography.ipynb`. The tutorial includes 
- generating the checkerboard
- post-processing the captured image for easy corner detection
- detecting the corners
- sorting the corners
- computing the homography matrix.

## Building the Real-Time Dual-Pixel Prototype

To achieve our custom dual pixel camera that can stream DP images, we pair a polarized machine vision sensor from [Lucid Vision](https://thinklucid.com/product/phoenix-5-0-mp-polarization-model-imx264mzrmyr/) with a custom split-polarization aperture. We can split the aperture plane into orthogonal polarization states, and use the orthogonal polarization images from our quad pixel sensor, corresponding to left and right views of a Dual-Pixel camera.

In our prototype, we split the aperture into left and right halves for 0 and 90 degrees of polarization.

We use a half waveplate (HWP) to rotate the polarization for half of the aperture.

Since a half waveplate rotates polarization by $2\theta$ when its fast axis is $\theta$ away from the polarization axis, we need to orient the HWP at **45 degrees** relative to the incoming polarization axis from a linear polarizer.

### How to align the Half Waveplate and Linear Polarizer

We need to figure out this relative orientation before we cut the HWP into half circle shape. To find the orientation, we can to test-orient it by sandwiching it between two orthogonal linear polarizers. Rotate the HWP so that it can pass through all the polarized light through the two orthonal linear polarizers. Mark this relative orientation on **both the first polarizer and the HWP**.

To make the split aperture, we can follow the steps below:
1. Put the first polarizer in front of the camera
2. Orient the polarizer so that light is polarized to 0 degrees (note that this is not orthogonal to SLM polarization state otherwise we will block all the light)
3. Now, along this marked orientation on the first polarizer, position the HWP with the relative 45 degree orientation mark.
4. Keep the orientation of the HWP and cut it into the right-half-circle shape.
5. Insert the half-circle HWP into the aperture plane slot of the camera.
6. Put a flat white light source in front of the objective lens.
7. Open ArenaView to verify that 0 and 90 degrees are seeing the opposite vignetting patterns and that the vignetting is the same between 45 and 135 degree images.
8. Verify again with real scenes that 0 and 90 degrees see disparity images. Adjust the polarizer if needed.
9. Done!

## Extracting the Dual-Pixel Views from Canon RAW Files

If you are using a Canon camera with a dual-pixel (DP) sensor instead of the Lucid Vision PHX050S1-P/Q sensor, you can extract the left and right DP view images from `.CR3` RAW files using the `unprocessed_raw` command from the [LibRaw](https://www.libraw.org/) library.

### Install LibRaw

1. Download LibRaw from https://www.libraw.org/download.
   - **From source:** build the library and the bundled sample programs (in the `samples/` directory). After building, `unprocessed_raw` will be in the `bin/` folder.
   - **From binaries** (available for macOS 11+ ARM/Intel and Windows 64-bit): `unprocessed_raw` is already in the `bin/` folder.
2. Make sure `unprocessed_raw` is on your system `PATH`.

Full documentation for LibRaw sample programs: https://www.libraw.org/docs/Samples-LibRaw.html

### Extract DP views via the command line

```bash
# Extract the A+B (combined) dual-pixel image
unprocessed_raw -T -s 0 IMG_0001.CR3

# Extract the A-only (single sub-pixel) image
unprocessed_raw -T -s 1 IMG_0001.CR3
```

This produces two TIFF files. The B-only (right) view can then be obtained by subtraction: `B = (A+B) − A`.

### Extract DP views using the Python helper

We provide `code/utils_dpraw.py` which wraps the above commands and adds Bayer demosaicing, cropping, and vanilla white balancing:

```python
from utils_dpraw import extract_dp_images_from_raw

loadfolder = "./static/example_images_canon_dpraw"
load_image_name = "IMG_0001"

imgrgb_l, imgrgb_r, imgrgb_total = extract_dp_images_from_raw(loadfolder, 
                                                              load_image_name, 
                                                              white_balancing=True)
```

See `code/utils_dpraw.py` for the full set of options including cropping and white balancing.

## Citation

If you use our code or dataset, please cite our paper:
```
@InProceedings{qin2025spatially,
      author    = {Qin, Yingsi and Sankaranarayanan, Aswin C. and O'Toole, Matthew},
      title     = {Spatially-Varying Autofocus},
      booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
      publisher = {IEEE},
      language  = {eng},
      month     = {October},
      year      = {2025},
      pages     = {24645-24654}, 
      keywords  = {All-in-Focus Imaging, Extended Depth of Field, Autofocus Algorithms, Computational Imaging}
      }
```
