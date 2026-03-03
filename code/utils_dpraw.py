"""
Python Program to extract demosaiced and white balanced dual-pixel (DP) views from Canon RAW images.

Author: Yingsi Qin

This module provides helpers to:
- run the external tool `unprocessed_raw` from the LibRaw library 
via the command line to extract DP subimages from a .cr3 image file,
- apply simple white balancing,
- demosaic the Bayer pattern,
- and return left/right RGB views plus an averaged image.

Dependencies
-----------
This module requires the `unprocessed_raw` command-line tool from the LibRaw library, which is
one of the sample programs from the LibRaw library.

To obtain `unprocessed_raw`:
1. Download and install LibRaw from https://www.libraw.org/download
   If you downloaded the source code, build the library and the bundled sample programs (located in
   the `samples/` directory of the source distribution).  After
   building, `unprocessed_raw` will be in the `bin/` folder.
   If you downloaded the binaries (availble for Mac OS X 11+ with 64-bit ARM/Intel and windows 64 bit), 
   `unprocessed_raw` will be in the `bin/` folder.
2. Make sure `unprocessed_raw` is on your system PATH so that it can
   be invoked from the command line.

Full documentation for all LibRaw sample programs (including `unprocessed_raw`) is available at:
    https://www.libraw.org/docs/Samples-LibRaw.html

The alternative way to extract dual pixel raw .tiff images directly from a .cr3 image file via
the command line, without demosaicing and white balancing:
    unprocessed_raw -T -s 0 IMG_0001.CR3   # extracts the A+B (combined) image
    unprocessed_raw -T -s 1 IMG_0001.CR3   # extracts the A (single-view) image

"""

import subprocess
import os
import numpy as np
import skimage
import cv2
from scipy import interpolate

def get_extract_commands(load_folder, image_name, extension='.CR3'):
    """Return shell commands to extract both DP views from a RAW file.

    Args:
        load_folder: Directory containing the RAW file.
        image_name: Base filename without extension.
        extension: RAW file extension (default: .CR3).

    Returns:
        Tuple of (command_view1, command_view2) strings.
    """

    filename = image_name + extension
    extract_command1 = "unprocessed_raw -T -s 0 "+load_folder+"/"
    extract_command2 = "unprocessed_raw -T -s 1 "+load_folder+"/"

    extract_view_command1 = extract_command1 + filename
    extract_view_command2 = extract_command2 + filename

    return extract_view_command1, extract_view_command2

def get_rename_commands(load_folder, image_name, extension='.CR3'):
    """Return shell commands to rename extracted TIFFs to a consistent scheme."""
    filename = image_name + extension
    image_name_view1 = load_folder+"/"+filename+".tiff"
    image_name_view2 = load_folder+"/"+filename+"-1.tiff"
    image_name_new_view1 = load_folder+"/"+image_name+"-1.tiff"
    image_name_new_view2 = load_folder+"/"+image_name+"-2.tiff"
    change_name_command1 = "mv "+image_name_view1+" "+image_name_new_view1
    change_name_command2 = "mv "+image_name_view2+" "+image_name_new_view2

    return change_name_command1, change_name_command2

def get_clean_intermediates_commands(load_folder, image_name):
    """Return shell commands to remove intermediate TIFFs after processing."""
    image_name_new_view1 = load_folder+"/"+image_name+"-1.tiff"
    image_name_new_view2 = load_folder+"/"+image_name+"-2.tiff"
    clean_name_command1 = "rm "+image_name_new_view1
    clean_name_command2 = "rm "+image_name_new_view2

    return clean_name_command1, clean_name_command2

def dp_grey_world_white_balancing(masks_left, colorLayers_left, masks_right, colorLayers_right):
    """Apply grey-world white balancing to DP left/right Bayer layers.

    Returns:
        left_sum, right_sum: 2D arrays of combined (R+G+B) mosaics
        left_stack, right_stack: list of [R, G, B] mosaics for each view
    """
    r0_r = np.sum(colorLayers_right[0] * masks_right[0])/np.sum(masks_right[0])
    g1_r = np.sum(colorLayers_right[1] * masks_right[1])/np.sum(masks_right[1])
    g2_r = np.sum(colorLayers_right[2] * masks_right[2])/np.sum(masks_right[2])
    g0_r = (g1_r + g2_r) / 2
    b0_r = np.sum(colorLayers_right[3] * masks_right[3])/np.sum(masks_right[3])
    
    r0_l = np.sum(colorLayers_left[0] * masks_left[0])/np.sum(masks_left[0])
    g1_l = np.sum(colorLayers_left[1] * masks_left[1])/np.sum(masks_left[1])
    g2_l = np.sum(colorLayers_left[2] * masks_left[2])/np.sum(masks_left[2])
    g0_l = (g1_l + g2_l) / 2
    b0_l = np.sum(colorLayers_left[3] * masks_left[3])/np.sum(masks_left[3])
    
    r_l = colorLayers_left[0] / r0_l * g0_r
    g_l = (colorLayers_left[1]+colorLayers_left[2]) / g0_l * g0_r
    b_l = colorLayers_left[3] / b0_l * g0_r
    
    r_r = colorLayers_right[0] / r0_r * g0_r
    g_r = (colorLayers_right[1]+colorLayers_right[2]) / g0_r * g0_r
    b_r = colorLayers_right[3] / b0_r * g0_r
    
    return r_l + g_l + b_l, r_r + g_r + b_r, [r_l, g_l, b_l], [r_r, g_r, b_r]

def dp_white_world_white_balancing(masks_left, colorLayers_left, masks_right, colorLayers_right):
    """Apply white-world (max-based) balancing to DP left/right Bayer layers."""
    r0_r = np.max(colorLayers_right[0] * masks_right[0])
    g1_r = np.max(colorLayers_right[1] * masks_right[1])
    g2_r = np.max(colorLayers_right[2] * masks_right[2])
    g0_r = np.max([g1_r, g2_r])
    b0_r = np.max(colorLayers_right[3] * masks_right[3])

    r0_l = np.max(colorLayers_left[0] * masks_left[0])
    g1_l = np.max(colorLayers_left[1] * masks_left[1])
    g2_l = np.max(colorLayers_left[2] * masks_left[2])
    g0_l = np.max([g1_l, g2_l])
    b0_l = np.max(colorLayers_left[3] * masks_left[3])
    
    r_l = colorLayers_left[0] / r0_l * g0_r
    g_l = (colorLayers_left[1]+colorLayers_left[2]) / g0_l * g0_r
    b_l = colorLayers_left[3] / b0_l * g0_r
    
    r_r = colorLayers_right[0] / r0_r * g0_r
    g_r = (colorLayers_right[1]+colorLayers_right[2]) / g0_r * g0_r
    b_r = colorLayers_right[3] / b0_r * g0_r
    
    return r_l + g_l + b_l, r_r + g_r + b_r, [r_l, g_l, b_l], [r_r, g_r, b_r]

def dp_default_white_balancing(masks_left, colorLayers_left, masks_right, colorLayers_right,
                               r_scale=1.0, g_scale=1.0, b_scale=1.0):
    """Apply white balancing with the given RGB gains to DP left/right Bayer layers.

    Args:
        masks_left (np.ndarray): Bayer masks for the left view ``(4, H, W)``.
        colorLayers_left (np.ndarray): Masked colour layers for the left view.
        masks_right (np.ndarray): Bayer masks for the right view.
        colorLayers_right (np.ndarray): Masked colour layers for the right view.
        r_scale (float): Red channel gain.
        g_scale (float): Green channel gain.
        b_scale (float): Blue channel gain.

    Returns:
        tuple: ``(left_sum, right_sum, left_stack, right_stack)``.
    """
    
    r_l = colorLayers_left[0] * r_scale
    g_l = (colorLayers_left[1]+colorLayers_left[2]) * g_scale / 2
    b_l = colorLayers_left[3] * b_scale
    
    r_r = colorLayers_right[0] * r_scale
    g_r = (colorLayers_right[1]+colorLayers_right[2]) * g_scale / 2
    b_r = colorLayers_right[3] * b_scale
    
    return r_l + g_l + b_l, r_r + g_r + b_r, [r_l, g_l, b_l], [r_r, g_r, b_r]

def demosaicing(img_whitebalanced):
    """Demosaic a Bayer mosaic into an RGB image via bilinear interpolation."""
    red = img_whitebalanced[0::2, 0::2] 
    green1 = img_whitebalanced[0::2, 1::2] 
    green2 = img_whitebalanced[1::2, 0::2] 
    blue = img_whitebalanced[1::2, 1::2]
    
    ### red
    ycoor, xcoor = red.shape
    f_r = interpolate.interp2d(np.arange(xcoor), np.arange(ycoor), red, kind='linear') 
    demosaiced_r = f_r(np.arange(0,xcoor,0.5), np.arange(0,ycoor,0.5))
    ### green
    ycoor, xcoor = green1.shape
    f_g1 = interpolate.interp2d(np.arange(xcoor), np.arange(ycoor), green1, kind='linear') 
    demosaiced_g1 = f_g1(np.arange(0,xcoor,0.5), np.arange(0,ycoor,0.5))
    ycoor, xcoor = green2.shape
    f_g2 = interpolate.interp2d(np.arange(xcoor), np.arange(ycoor), green2, kind='linear') 
    demosaiced_g2 = f_g2(np.arange(0,xcoor,0.5), np.arange(0,ycoor,0.5))
    demosaiced_g = (demosaiced_g1+demosaiced_g2)/2
    ### blue
    ycoor, xcoor = blue.shape
    f_b = interpolate.interp2d(np.arange(xcoor), np.arange(ycoor), blue, kind='linear') 
    demosaiced_b = f_b(np.arange(0,xcoor,0.5), np.arange(0,ycoor,0.5))
    
    ### stack together
    demosaiced = np.dstack((demosaiced_r, demosaiced_g, demosaiced_b))

    return demosaiced

def construct_masks_and_color_layers(img):
    '''Construct boolean masks and color layers for the RGGB Bayer filter pattern.'''

    masks = []
    for i in range(4):
        masks.append(np.zeros(img.shape))
    masks = np.array(masks)
    masks[0, 0::2, 0::2] = 1 # R
    masks[1, 0::2, 1::2] = 1 # G
    masks[2, 1::2, 0::2] = 1 # G
    masks[3, 1::2, 1::2] = 1 # B
    colorLayers = []

    for i in range(4):
        colorLayers.append(img*masks[i])

    return masks, np.array(colorLayers)

def extract_dp_views(folder, imgname):
    """Read pre-extracted DP TIFF files and return the combined, left, and right views.

    Loads the two TIFF files produced by ``unprocessed_raw``:
      - ``<imgname>-1.tiff`` — the A+B (combined) dual-pixel image.
      - ``<imgname>-2.tiff`` — the A-only (single sub-pixel) image.

    Each image is converted to float64, the black level (512) is
    subtracted, and values are normalised to [0, 1] using a 13-bit
    range (2^13 − 1).  The right (B-only) view is obtained by
    subtraction: B = (A+B) − A.

    Args:
        folder  (str): Directory containing the extracted TIFF files.
        imgname (str): Base filename without extension.

    Returns:
        tuple[ndarray, ndarray, ndarray]:
            ``(img_combined, img_l, img_r)`` — the combined (A+B)
            image, the left (A) view, and the right (B = A+B − A) view,
            all as 2-D float64 arrays normalised to [0, 1].
    """

    # Load A+B combined image, subtract black level, normalise
    img_combined = skimage.io.imread(folder+'/'+imgname+'-1.tiff').astype(np.float64) - 512
    img_combined = img_combined/(2**13-1)

    # Load A-only image, subtract black level, normalise
    img_l = skimage.io.imread(folder+'/'+imgname+'-2.tiff').astype(np.float64) - 512
    img_l = img_l/(2**13-1)

    # Derive the right (B-only) view by subtraction
    img_r = img_combined - img_l

    return img_combined, img_l, img_r

def extract_dp_images(loadfolder, image_name, white_balancing=True):
    """Extract and demosaic DP left/right views from pre-extracted RAW data.

    Args:
        loadfolder: Directory containing extracted TIFFs for both views.
        image_name: Base filename without extension.
        white_balancing: Whether to apply grey-world balancing.

    Returns:
        imgrgb_l, imgrgb_r, imgrgb: left, right, and averaged RGB images.
    """

    # Read left/right DP views
    imgAandB, img1, img2 = extract_dp_views(folder=loadfolder, imgname=image_name)

    # Color balancing at the mosaic level for both views
    masks_left, colorLayers_left = construct_masks_and_color_layers(img1)
    masks_right, colorLayers_right = construct_masks_and_color_layers(img2)
    if white_balancing:
        color_seq_l, color_seq_r, color_stack_l, color_stack_r = dp_grey_world_white_balancing(masks_left, 
                                                                                            colorLayers_left, 
                                                                                            masks_right, 
                                                                                            colorLayers_right)
    else:
        color_seq_l, color_seq_r, color_stack_l, color_stack_r = dp_default_white_balancing(masks_left,
                                                                                            colorLayers_left,
                                                                                            masks_right,
                                                                                            colorLayers_right)

    # Demosaic the mosaics into RGB images
    imgrgb_r = demosaicing(color_seq_r)
    imgrgb_l = demosaicing(color_seq_l)
    imgrgb = (imgrgb_l + imgrgb_r) / 2

    return imgrgb_l, imgrgb_r, imgrgb

def extract_dp_images_from_raw(loadfolder, load_image_name, white_balancing=True):
    """Full pipeline: extract DP views from RAW and return demosaiced RGB.

    This runs the external `unprocessed_raw` tool to generate TIFFs,
    renames them, demosaics both views, and cleans intermediate files.

    Args:
        loadfolder: Directory containing the RAW file.
        load_image_name: Base filename without extension.
        white_balancing: Whether to apply grey-world balancing.

    Returns:
        imgrgb_l, imgrgb_r, imgrgb_total
    """

    # Build shell commands for extract/rename/cleanup steps
    extract_view_command1, extract_view_command2 = get_extract_commands(loadfolder, load_image_name)
    change_name_command1, change_name_command2 = get_rename_commands(loadfolder, load_image_name)
    clean_name_command1, clean_name_command2 = get_clean_intermediates_commands(loadfolder, load_image_name)

    # Extract the two views using unprocessed_raw
    subprocess.run(extract_view_command1, shell=True)
    subprocess.run(extract_view_command2, shell=True)
    subprocess.run(change_name_command1, shell=True)
    subprocess.run(change_name_command2, shell=True)

    # Preprocess and demosaic the two views
    imgrgb_l, imgrgb_r, imgrgb_total = extract_dp_images(loadfolder, load_image_name, white_balancing=white_balancing)
    # Clean up intermediate files
    subprocess.run(clean_name_command1, shell=True)
    subprocess.run(clean_name_command2, shell=True)

    return imgrgb_l, imgrgb_r, imgrgb_total

### Example Usage
if __name__ == "__main__":

    # Example: extract DP views from a RAW file called "IMG_0001.CR3" and save the images to iofolder
    # Update `iofolder` and `load_image_name` for your data

    iofolder = "./static/example_images_canon_dpraw"
    load_image_name = "IMG_0001"

    # Set the parameters
    white_balancing = True
    xbegin = 143
    ybegin = 39
    brightness_scale = 1.0
    gamma = 1/2.2

    imgrgb_l, imgrgb_r, imgrgb_total = extract_dp_images_from_raw(iofolder, load_image_name, 
                                                                  white_balancing=white_balancing)

    # Crop the images
    imgrgb_l = imgrgb_l[ybegin:, xbegin:]
    imgrgb_r = imgrgb_r[ybegin:, xbegin:]
    imgrgb_total = imgrgb_total[ybegin:, xbegin:]

    # Apply gamma correction
    imgrgb_l = np.clip(imgrgb_l * brightness_scale, 0, 1) ** gamma
    imgrgb_r = np.clip(imgrgb_r * brightness_scale, 0, 1) ** gamma
    imgrgb_total = np.clip(imgrgb_total * brightness_scale, 0, 1) ** gamma

    # Save the images
    imgrgb_l = cv2.cvtColor(np.round(imgrgb_l*65535).astype(np.uint16), cv2.COLOR_RGB2BGR)
    imgrgb_r = cv2.cvtColor(np.round(imgrgb_r*65535).astype(np.uint16), cv2.COLOR_RGB2BGR)
    imgrgb_total = cv2.cvtColor(np.round(imgrgb_total*65535).astype(np.uint16), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(iofolder, load_image_name + "_l.png"), imgrgb_l)
    cv2.imwrite(os.path.join(iofolder, load_image_name + "_r.png"), imgrgb_r)
    cv2.imwrite(os.path.join(iofolder, load_image_name + "_total.png"), imgrgb_total)
    