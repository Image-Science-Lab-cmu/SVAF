import time
import ctypes
import numpy as np
import skimage.io

from arena_api.buffer import BufferFactory
from arena_api.system import system
from arena_api.__future__.save import Writer
from arena_api.enums import PixelFormat

from utils_dpsensor_isp import *
from polarizedcamera import PolarizedCamera

"""
Polarization, Angles
	This example introduces the basics of working with the polarized
	angles pixel format. Specifically, this example retrieves a 4-channel
	PolarizedAngles_0d_45d_90d_135d_Mono8 or PolarizedAngles_0d_45d_90d_135d_BayerRG8,
	depending on the camera. It first splits the 4 channels into separate images. Then,
	it writes the four images onto a 2x2 grid and saves it to disk. Finally, it saves
	each individual image to disk.
"""
TAB1 = "  "
TAB2 = "    "
TAB3 = "      "


if __name__ == '__main__': 
	savegridview = int(input("Save grid view? (1/0): "))
	savefolder = 'C:/Users/matth/Desktop/SVAF/images/py_polarization_angles'

	polarized_camera = PolarizedCamera()
	images = polarized_camera.capture_dp_image(savefolder, savegridview)

	# # Create a camera device
	# devices = create_devices_with_tries()
	# device = system.select_device(devices)
	# set_features(device) 
	# nodes, pixel_format_initial_value = initialize_camera(device) 

	# images = capture_dp_image(device, savefolder, savegridview)

	# # Reset the pixel format to its initial value
	# nodes['PixelFormat'].value = pixel_format_initial_value

	# system.destroy_device()
	# print(f'{TAB1}Destroyed all created devices')
