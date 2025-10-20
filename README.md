# Spatially-Varying Autofocus [ICCV 2025]

[Yingsi Qin](https://yingsiqin.github.io/), [Aswin C. Sankaranarayanan](https://users.ece.cmu.edu/~saswin/), [Matthew O'Toole](https://www.cs.cmu.edu/~motoole2/)

This repository contains both (1) the code for running our autofocus algorithms, and (2) the PDAF hardware code for the real-time prototype, used for [Spatially-Varying Autofocus](https://imaging.cs.cmu.edu/svaf/). 

Visit the links below for more information:\
 [[Paper](https://imaging.cs.cmu.edu/svaf/static/pdfs/Spatially_Varying_Autofocus.pdf)] [[Supplemental PDF](https://imaging.cs.cmu.edu/svaf/static/pdfs/Spatially_Varying_Autofocus-supp.pdf)] [[Project Website](https://imaging.cs.cmu.edu/svaf/)] [[5-Minute Video](https://www.youtube.com/watch?v=WNPkUB9o2Fo)] [[Poster](https://imaging.cs.cmu.edu/svaf/static/pdfs/Spatially_Varying_Autofocus-poster.pdf)]

## Getting Started

All simulation codes are in MATLAB. The simulation is based on wave optics.

## Python program

This program runs vanilla optical flow to perform phase-based autofocus. Detailed code for our CDAF and PDAF algoriothms are coming soon.

#### Dependencies
If you are using a Lucid Vision polarized camera, please download Arena Python API:
https://dce9ugryut4ao.cloudfront.net/arena_api-2.7.1-py3-none-any.zip

Install additional dependencies:
```
pip install -r requirements.txt
```

#### Run the python program
How to run:
1. Please make sure that the camera is turned off in ArenaView before running the python program. The program will have GenICam errors if the camera is already turned on in ArenaView.
2. Run the following commands.
```
cd code
python svaf_vanilla.py
```

## C++ program
#### Dependencies
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

#### Compile the C++ program
```
cd cpp_program
msbuild svafstream.sln //p:Configuration=Release //p:Platform=x64
```

#### Run the C++ program
How to run:
1. Please make sure that the camera is turned off in ArenaView before running the CPP program. The program will have GenICam errors if the camera is already turned on in ArenaView.
2. Make sure to turn on the lights!
3. Run the following commands.
```
cd cpp_program
./build/svafstream/x64/Release/svafstream.exe
```

## Aligning the Optical System

#### Homography
The entire walkthrough is detailed in `homography/homography.ipynb`. The tutorial includes 
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

#### How to align the Half Waveplate and Linear Polarizer

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

## Citation

If you use our code or dataset, please cite our paper:
```
@inproceedings{qin2025spatially,
    author = {Qin, Yingsi and Sankaranarayanan, Aswin C. and O'Toole, Matthew},
    title = {Spatially-Varying Autofocus},
    year = {2025},
    publisher = {IEEE},
    url = {https://imaging.cs.cmu.edu/svaf/static/pdfs/Spatially_Varying_Autofocus.pdf},
    booktitle = {2025 IEEE/CVF International Conference on Computer Vision (ICCV)},
    isbn = {},
    language = {eng},
    pages = {},
    keywords = {All-in-Focus Imaging, Extended Depth of Field, Autofocus Algorithms, Computational Imaging}
    }
```