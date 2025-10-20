import glob
import numpy as np
import cv2
import skimage
import skimage.io
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage, interpolate
from scipy.optimize import curve_fit
import time
import os
from skimage import data, segmentation, color
from skimage.segmentation import expand_labels, watershed
from PIL import Image, ImageChops, ImageEnhance
from tqdm import tqdm
import datetime

os.environ["PYSDL2_DLL_PATH"] = r"C:\path_to_appdata\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\LocalCache\local-packages\Python39\site-packages\sdl2dll\dll"

import sys
import sdl2.ext
from sdl2.ext.image import pillow_to_surface
from sdl2 import timer
from sdl2.events import SDL_Event, SDL_QUIT

from params import Params
# from utils_isp import *
# from utils_aif import *
# from utils_dpflow import *

from arena_api.buffer import BufferFactory
from arena_api.system import system
from arena_api.__future__.save import Writer
from arena_api.enums import PixelFormat

from utils_dpsensor_isp import *
from polarizedcamera import PolarizedCamera


def ComputePhaseMaskFromSpatialPeriod(periodMap, Y_grid):
    '''
    Computes the phase mask with the full SLM tilting range given a spatial period map.
    '''

    periodMap = np.round(periodMap)
    sign = (periodMap > 0) * 2 - 1
    phaseData = np.mod(sign * Y_grid, np.abs(periodMap)) / \
        (np.abs(periodMap)-1)
    phaseData = np.round(phaseData*255).astype(np.uint8)

    return phaseData


def compute_slm_pattern_from_periodMap(periodMap, Y_grid, H_CAM2SLM, params, save_folder, save_name, save=False):

    SLMshape = (params.slmHeight, params.slmWidth)
    periodmap_slm_warpped = cv2.warpPerspective(
        periodMap, H_CAM2SLM, (SLMshape[1], SLMshape[0]), flags=cv2.INTER_NEAREST)
    periodmap_slm_warpped = np.round(periodmap_slm_warpped)
    X1, Y1 = np.meshgrid(np.arange(0, params.slmWidth)+1,
                         np.arange(0, params.slmHeight)+1)
    slm_img = ComputePhaseMaskFromSpatialPeriod(periodmap_slm_warpped, Y1)

    if save:
        skimage.io.imsave(save_folder+'/slm_img_'+save_name+'.png', slm_img)

    return slm_img


def assign_displays(infos, slm_scene_init, SLMshape):
    scenes = {}
    display_names = {}
    valid_display_ids = []
    for display_id in range(len(infos)):
        if int(infos[display_id].bounds.w) == SLMshape[1]:
            print('Detected SLM display.')
            display_names[display_id] = 'SLM'
            scenes[display_names[display_id]] = slm_scene_init
            valid_display_ids.append(display_id)
        else:
            continue
        print(display_id, display_names[display_id], infos[display_id],
              infos[display_id].bounds.w, infos[display_id].bounds.h)
        assert scenes[display_names[display_id]].size[0] == infos[display_id].bounds.w and \
            scenes[display_names[display_id]
                   ].size[1] == infos[display_id].bounds.h

    return display_names, scenes, valid_display_ids


if __name__ == "__main__":

    print('Spatially-Varying Autofocus python script on Apple M2 Max version 03/19/2025')
    print('-----------------------------------------------------------------------')
    print('This script runs for one HOLOEYE GAEA SLM display.')

    ### load parameters
    params = Params()
    
    step_lr_gd = 0.06

    SLMshape = (params.slmHeight,params.slmWidth)
    CAMshape = (1024, 1224) #(4090, 6192)

    polarized_camera = PolarizedCamera()
    polarized_camera.set_features(exposure_time=47178.6, gain=18.9)

    slm_img_off = np.zeros(shape=(SLMshape[0], SLMshape[1])).astype(np.uint8)
    H_CAM2SLM = np.load('../homography/50mm_customdpsensor/result_HomographyMatrix_Cam.npy')

    savefolder = 'C:/path_to_folder/SVAF/images/results'

    ### load slm
    print('-----------------------------------------------------------------------')
    print('Loading SLM...')

    ### load displays
    print('-----------------------------------------------------------------------')
    print('Initializing SLM...')
    slm_scene = Image.fromarray(slm_img_off)
    factory = sdl2.ext.SpriteFactory(sdl2.ext.SOFTWARE)
    sdl2.ext.init()
    infos = sdl2.ext.displays.get_displays()
    display_names, scenes, valid_display_ids = assign_displays(infos, slm_scene, SLMshape)
    print('display_names:', display_names)
    windows = {} 
    spriterenderers = {}
    for display_id in range(len(infos)):
        if display_id==0:
            continue
        if int(infos[display_id].bounds.w)==SLMshape[1]:
            windows[display_names[display_id]] = sdl2.ext.Window(
                "Displayed SLM " + display_names[display_id] + " image", 
                size=(infos[display_id].bounds.w , infos[display_id].bounds.h),
                position=(infos[display_id].bounds.x,infos[display_id].bounds.y),
                flags= sdl2.SDL_WINDOW_BORDERLESS)
            windows[display_names[display_id]].show()
            spriterenderers[display_names[display_id]] = factory.create_sprite_render_system(windows[display_names[display_id]])
        else:
            continue
        
    # initiate the SLM display
    sprite = factory.from_surface(pillow_to_surface(slm_scene), free=True)
    spriterenderers['SLM'].render(sprite)
    windows['SLM'].refresh()

    ### set up parameters
    event = SDL_Event()
    offset = 0
    close = False
    image_name = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    save_image_name = 'phase_based_autofocus-'+image_name
    accommodation_on = 1
    step = 0
    maxsteps = 21
    normalization_factor = 1.0
    img_dc_l = None
    img_dc_r = None
    step_size_map = np.ones(CAMshape)
    spatial_freq_map_prev = np.zeros(shape=CAMshape, dtype=np.float64)
    spatial_freq_map_current = np.zeros(shape=CAMshape, dtype=np.float64)
    spatial_period_map = np.inf
    X1, Y1 = np.meshgrid(np.arange(0, params.slmWidth)+1,np.arange(0, params.slmHeight)+1)

    ### begin autofocus loop
    ID = int(round(time.time()*1000))
    curr_frame_time = 0
    prev_frame_time = 0
    min_period_far = 4
    max_period_near = 4
    spatial_freq_map = np.zeros(shape=(CAMshape[0], CAMshape[1]))
    print('------------------------------------------------------------------------------')
    print('Perform optical flow autofocus...')
    while not close:

        curr_frame_time = time.time()

        events = sdl2.ext.get_events()

        # if step == maxsteps:
        #     close = True
        #     break

        ### -------------- listen for user input --------------
        for event in events:
            if event.type == SDL_QUIT or sdl2.ext.input.key_pressed(event, 'q'):
                close = True
                break     

        # ### ----------------- Capture an image -----------------
        save_capture_step_name = save_image_name + '_step_' + str(step)
        images = polarized_camera.capture_dp_image(savefolder=savefolder, saveimages=False, savegridview=False)
        imgrgb_l, imgrgb_r = images[2], images[0]

        imgrgb_total = images[3] #(imgrgb_l + imgrgb_r) / 2
        
        ### ----------------- Display the image -----------------
        npndarray = cv2.resize(images[3], None, fx=0.5, fy=0.5)
        fps = str(1/(curr_frame_time - prev_frame_time))
        
        # Display optical flow (vx) using turbo colormap
        vmin, vmax = -1/min_period_far, 1/max_period_near
        vx_normalized = (spatial_freq_map - vmin) / (vmax - vmin)
        vx_colored = cv2.applyColorMap((vx_normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        vx_colored = cv2.resize(vx_colored, None, fx=0.5, fy=0.5)
        
        image_to_show = np.concatenate((npndarray, vx_colored), axis=1)
        cv2.putText(image_to_show, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (100, 255, 0), 3, cv2.LINE_AA)
        
        cv2.imshow('Lucid', image_to_show)

        ### ----------------- Compute optical flow and set update spatial_freq_map_current -----------------
        mask = np.ones_like(imgrgb_total)[:, :, 1]
        vx_flow = compute_optical_flow_x(imgrgb_l.astype(np.float32), 
                                                imgrgb_r.astype(np.float32), 
                                                imgrgb_total.astype(np.float32),
                                                mask, width=202, lambda_reg=1e-10, 
                                                green=False, superpixels=False, 
                                                compactness=13, n_segments=500)
        vx = vx_flow * (-1)
            
        print('===============// vx min max:', np.min(vx), np.max(vx), '// ===============')
        print('===============// spatial_freq_map_current min max:', np.min(spatial_freq_map_current), np.max(spatial_freq_map_current), '// ===============')

        ### ----------------- Update spatial_freq_map and display it on the SLM -----------------
        spatial_freq_map = spatial_freq_map + spatial_freq_map_current

        ### ----------------- Discretize and show the spatial frequency map -----------------
        spatial_freq_map = np.clip(spatial_freq_map, -1/min_period_far, 1/max_period_near)
        unique_values = np.linspace(spatial_freq_map.min(), spatial_freq_map.max(), 60)
        spatial_freq_map = unique_values[np.argmin(np.abs(spatial_freq_map[:, :, np.newaxis] - unique_values), axis=2)]
            
        spatial_period_map = np.round(1 / spatial_freq_map)
        spatial_freq_map = 1 / spatial_period_map
        

        ## Display the pattern
        print('All-in-focus pattern on')
        slm_img_pattern = compute_slm_pattern_from_periodMap(spatial_period_map, Y1, H_CAM2SLM, params, 
                                                             save_folder=savefolder, save_name=save_capture_step_name, save=False)

        cv2.imwrite(os.path.join(savefolder, f'slm_pattern_python.jpg'), slm_img_pattern)
        slm_scene = Image.fromarray(slm_img_pattern)
        sprite = factory.from_surface(pillow_to_surface(slm_scene), free=True)
        spriterenderers['SLM'].render(sprite)
        windows['SLM'].refresh()

        prev_frame_time = curr_frame_time
        step += 1
        
        key = cv2.waitKey(1)
        if key == 27:
            close = True
            break

    # Proper cleanup before exiting
    del polarized_camera
    sdl2.ext.quit()
    cv2.destroyAllWindows()
