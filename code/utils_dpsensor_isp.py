import skimage
import time
import ctypes
import numpy as np
import cv2
from skimage import data, segmentation, color
import pandas as pd

from arena_api.buffer import BufferFactory
from arena_api.system import system
from arena_api.__future__.save import Writer
from arena_api.enums import PixelFormat

from arena_api._xlayer.xarena._ximagefactory import _xImagefactory
from arena_api.buffer import _Buffer

# import numba
# from numba import jit, njit, prange

TAB1 = "  "
TAB2 = "    "
TAB3 = "      "

def create_devices_with_tries():
	'''
	This function waits for the user to connect a device before raising
		an exception
	'''

	tries = 0
	tries_max = 6
	sleep_time_secs = 10
	while tries < tries_max:  # Wait for device for 60 seconds
		devices = system.create_device()
		if not devices:
			print(
				f'{TAB1}Try {tries+1} of {tries_max}: waiting for {sleep_time_secs} '
				f'secs for a device to be connected!')
			for sec_count in range(sleep_time_secs):
				time.sleep(1)
				print(f'{TAB1}{sec_count + 1 } seconds passed ',
                                    '.' * sec_count, end='\r')
			tries += 1
		else:
			print(f'{TAB1}Created {len(devices)} device(s)')
			return devices
	raise Exception(f'{TAB1}No device found! Please connect a device and run '
                    f'the example again.')


def write_to_2x2_grid(img, dst, dst_offset, dst_step, dst_half_stride):
	"""
	Helper function to write an individual image to a location
		within a 2x2 grid
	"""
	src = img.data
	src_h = img.height
	src_w = img.width
	src_step = PixelFormat.get_bits_per_pixel(img.pixel_format) // 8

	src_index = 0
	dst_index = int(dst_offset)

	for i in range(src_h):
		for j in range(src_w):
			dst[dst_index] = src[src_index]
			src_index += int(src_step)
			dst_index += int(dst_step)
		dst_index += int(dst_half_stride)


def save_image(img, filename):
	"""
	Helper function that takes an image and saves
		it to a given path in disk
	"""
	writer_jpg = Writer.from_buffer(img)

	writer_jpg.save(img, filename)

	print(f'{TAB3}Save image to {writer_jpg.saved_images[-1]}')


def set_features(device, stream=False):
    nodemap = device.nodemap
    nodemap['AcquisitionFrameRateEnable'].value = True
    nodemap['ExposureAuto'].value = 'Off'
    if stream:
        nodemap['AcquisitionMode'].value = 'Continuous'
    nodes_values_to_set = {'AcquisitionFrameRate': 21.020768519297064,
                           'BlackLevel': 0.0,
                           'ExposureTime': 47178.6,
                           'GainAuto': 'Off',
                           'Gain': 18.0}  # 26.0}

    for node_name, value in nodes_values_to_set.items():
        node = nodemap.get_node(node_name)

        if node is None:
            print(f"Node name: {node_name}")
            raise Exception("Node not found")
        if not node.is_writable:
            print(f"Node name: {node_name}")
            raise Exception("Node is not writable")

        if hasattr(node, 'min') and hasattr(node, 'max'):
            if value < node.min or value > node.max:
                print(f"Node name: {node_name}")
                raise Exception(
                    f"Value must be between {node.min} and {node.max}")

        node.value = value
        print(node_name, node.value)

    print('Successfully set features')


def initialize_camera_for_livestream(device):
    # Stream nodemap
    tl_stream_nodemap = device.tl_stream_nodemap

    tl_stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"
    tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
    tl_stream_nodemap['StreamPacketResendEnable'].value = True

    # Get device stream nodemap
    tl_stream_nodemap = device.tl_stream_nodemap

    # Get nodes ---------------------------------------------------------------
    nodes = device.nodemap.get_node(['PixelFormat'])

    # Retrieve enumeration entries and check which polarized angles pixel format is supported
    pixel_format = ''
    for pixel_format_name in nodes['PixelFormat'].enumentry_names:
        if (pixel_format_name == 'PolarizedAngles_0d_45d_90d_135d_BayerRG8'):
            pixel_format = pixel_format_name
            num_channels = 3

    if pixel_format == '':
        print("\tError - This example requires PolarizedAngles_0d_45d_90d_135d_* pixel formats")
        return

    # Get node values that will be changed in order to return their values at
    # the end of the example
    pixel_format_initial_value = nodes['PixelFormat'].value

    # Change pixel format to either:
    #    (1) PolarizedAngles_0d_45d_90d_135d_BayerRG8 (color)
    # (2) PolarizedAngles_0d_45d_90d_135d_Mono8 (monochrome)
    #    These pixel formats have 4-channels, each containing data from
    #    each degree of polarization (0, 45, 90, and 135). These channels
    #    are allocated in memory side-by-side.
    print(f'{TAB1}Setting Pixel Format to {pixel_format}')
    nodes['PixelFormat'].value = pixel_format

    return nodes, pixel_format_initial_value, num_channels


def check_initial_settings(device):

    # Define variables before the with block so they persist outside it
    src_pixel_format = None
    src_width = None
    src_height = None
    dst_2x2_width = None
    dst_2x2_height = None
    dst_2x2_pixel_format = None
    dst_2x2_step = None
    dst_2x2_stride = None
    dst_2x2_data_size = None
    starting_positions = None
    dst_top_left = None
    dst_top_right = None
    dst_bottom_left = None
    dst_bottom_right = None

    # Get initial information
    with device.start_stream(1):
        image_buffer = device.get_buffer()
        src_pixel_format = image_buffer.pixel_format

        src_pixel_format = image_buffer.pixel_format
        if src_pixel_format != PixelFormat.PolarizedAngles_0d_45d_90d_135d_BayerRG8 and src_pixel_format != PixelFormat.PolarizedAngles_0d_45d_90d_135d_Mono8:
            print(
                "\tError - Input image pixel format [{}] is a non-polarized format".format(src_pixel_format))

        # src info
        src_width = image_buffer.width
        src_height = image_buffer.height

        # 2x2 info
        dst_2x2_width = src_width * 2
        dst_2x2_height = src_height * 2
        dst_2x2_pixel_format = PixelFormat.BayerRG8 if src_pixel_format == PixelFormat.PolarizedAngles_0d_45d_90d_135d_BayerRG8 else PixelFormat.Mono8
        dst_2x2_step = PixelFormat.get_bits_per_pixel(
            dst_2x2_pixel_format) // 8

        dst_2x2_stride = dst_2x2_width * dst_2x2_step
        dst_2x2_data_size = dst_2x2_width * dst_2x2_height * dst_2x2_step

        # Reference set up to starting position of each quadrant of
        # destination 2x2 grid to write to
        dst_top_left = 0
        dst_top_right = dst_top_left + (dst_2x2_stride / 2)
        dst_bottom_left = dst_top_left + (dst_2x2_data_size / 2)
        dst_bottom_right = dst_top_left + \
            (dst_2x2_data_size / 2) + (dst_2x2_stride / 2)
        starting_positions = [dst_top_left,
                              dst_top_right, dst_bottom_left, dst_bottom_right]

        device.requeue_buffer(image_buffer)
        device.stop_stream()

    return dst_2x2_data_size, dst_2x2_step, dst_2x2_stride, \
        dst_2x2_width, dst_2x2_height, dst_2x2_pixel_format, \
        starting_positions, src_pixel_format


def split_channels_optimized(buffer):
    """
    Takes an interleaved image and separates the channels into multiple images.
    Optimized version using NumPy for faster processing.
    """
    if not isinstance(buffer, _Buffer):
        raise TypeError(f'Buffer expected instead of {type(buffer).__name__}')

    # Get number of buffers
    num_buffers_value = _xImagefactory.xImageFactoryNumChannels(
        buffer.xbuffer.hxbuffer.value
    )

    # Deinterleave
    hxbuffer_value = _xImagefactory.xImageFactoryDeinterleaveChannels(
        buffer.xbuffer.hxbuffer.value)
    buffer_deinterleave = _Buffer(hxbuffer_value)

    # Get length
    len_value = _xImagefactory.xImageFactoryDeinterleaveChannelsLen(
        buffer.xbuffer.hxbuffer.value
    )

    # Get other info for resulting images
    pf = buffer_deinterleave.xbuffer.xImageGetPixelFormat()
    width = buffer_deinterleave.width
    height = buffer_deinterleave.height // num_buffers_value
    size = len_value // num_buffers_value

    # Convert pdata to a numpy array for faster access
    # Create a numpy array view of the deinterleaved data
    total_size = size * num_buffers_value
    np_data = np.ctypeslib.as_array(
        buffer_deinterleave.pdata, shape=(total_size,))

    # Split images and store in array as buffer
    buffer_array = []

    for i in range(num_buffers_value):
        # Create a slice of the numpy array for this channel
        # This is just a view, not a copy
        start_idx = i * size
        end_idx = start_idx + size
        channel_slice = np_data[start_idx:end_idx]

        # Create a new buffer for the data
        data_mid = (ctypes.c_uint8 * size)()

        # Copy the data all at once using numpy
        ctypes_arr = np.ctypeslib.as_array(data_mid, shape=(size,))
        np.copyto(ctypes_arr, channel_slice)

        pdata_mid = ctypes.cast(data_mid, ctypes.POINTER(ctypes.c_uint8))

        hxbuffer_value = _xImagefactory.xImageFactoryCreate(
            pdata_mid, size, width, height, pf
        )

        buffer_array.append(_Buffer(hxbuffer_value))

    _xImagefactory.xImageFactoryDestroy(
        buffer_deinterleave.xbuffer.hxbuffer.value)

    return buffer_array


# # For write_to_2x2_grid_optimized
# @njit(parallel=True)
# def _optimized_grid_copy(src_array, dst_array, src_indices, dst_indices):
#     """JIT-compiled function to copy grid data"""
#     for i in prange(len(src_indices)):
#         dst_array[dst_indices[i]] = src_array[src_indices[i]]


def write_to_2x2_grid_optimized(img, dst, src_index_list, dst_index_list):
    """
    Helper function to write an individual image to a location within a 2x2 grid
    Optimized version using NumPy for faster processing
    """
    # Get source data and dimensions
    src = bytes(img.data)

    # Create a source numpy array
    src_array = np.frombuffer(src, dtype=np.uint8)

    # Create a destination view using numpy
    dst_array = np.ctypeslib.as_array(dst)

    dst_array[dst_index_list] = src_array[src_index_list]


def compute_optical_flow_x(img_l, img_r, img_total, mask, width, lambda_reg=1e-10, green=True, superpixels=True, compactness=30, n_segments=2000, bilateral=False, d=20, sigmaColorScale=40, sigmaSpace=90):

    mask = np.pad(mask, ((0,0),(1,0)), mode='constant', constant_values=0)
    img_l = np.pad(img_l, ((0,0),(1,0),(0,0)), mode='constant', constant_values=0)
    img_r = np.pad(img_r, ((0,0),(1,0),(0,0)), mode='constant', constant_values=0)

    if green:
        xdiff = (img_l[:,1:,:] - img_l[:,:-1,:]).astype(np.float64)[:,:,1] #/ 2
        xdiff[:, 0] = 0
        xdiff[:, -1] = 0
        xdiff = xdiff * mask[:,1:]
        tdiff = (img_l - img_r)[:, 1:, 1].astype(np.float64) * mask[:,1:]
    else:
        img_l = cv2.cvtColor(img_l.astype(np.float32), cv2.COLOR_RGB2GRAY)
        img_r = cv2.cvtColor(img_r.astype(np.float32), cv2.COLOR_RGB2GRAY)
        xdiff = (img_l[:,1:] - img_l[:,:-1]).astype(np.float64) * mask[:,1:] #/ 2
        xdiff[:, 0] = 0
        xdiff[:, -1] = 0
        tdiff = (img_l - img_r)[:, 1:].astype(np.float64) * mask[:,1:]
    
    if superpixels:
        numerator = -xdiff*tdiff
        denominator = xdiff*xdiff
        numerator_segmented, labels = calculate_segmented_mean_map(numerator, img_total,
                                                                compactness=compactness, 
                                                                n_segments=n_segments)
        denominator_segmented, labels = calculate_segmented_mean_map(denominator, img_total,
                                                                    compactness=compactness, 
                                                                    n_segments=n_segments)
        lambda_reg = 1e-10
        u = numerator_segmented / (denominator_segmented + lambda_reg)
        u = u.astype(np.float64)
    else:
        numerator = cv2.boxFilter(-xdiff*tdiff, ddepth=-1, ksize=(width, width),
                                normalize=False, borderType=cv2.BORDER_DEFAULT) / (width*width)
        denominator = cv2.boxFilter(xdiff*xdiff, ddepth=-1, ksize=(
            width, width), normalize=False, borderType=cv2.BORDER_DEFAULT) / (width*width)
        u = numerator / (denominator + lambda_reg)
        u = u.astype(np.float64)
    
    if bilateral and d > 0:
        sigmaColor = (np.max(u) - np.min(u)) * sigmaColorScale
        try:
            u = cv2.ximgproc.jointBilateralFilter(
                src=u.astype(np.float32),
                joint=img_total,
                d=d,
                sigmaColor=sigmaColor,
                sigmaSpace=sigmaSpace
            )
        except AttributeError:
            print("cv2.ximgproc.jointBilateralFilter is not available. Please install opencv-contrib-python.")
            return
        
    return u


def calculate_segmented_mode_map(img, color_guide, compactness=30, n_segments=500):
    '''
    Input: a 2-dimensional grayscale image that can have uint8, float32, or float64 data type.
    Returns: the image with segmented regions having the region mode value.
    '''

    # segment the image into regions
    # uses k-means to segment the image
    labels1 = segmentation.slic(
        color_guide, compactness=compactness, n_segments=n_segments, start_label=0)

    # flatten both the array and the segments
    flat_img = img.ravel()
    flat_labels = labels1.ravel()

    # print('img.shape:', img.shape, 'labels.shape:', labels1.shape)

    # create a DataFrame for grouping
    df = pd.DataFrame({'value': flat_img, 'label': flat_labels})

    # group by segment and calculate the mode
    mode_series = df.groupby('label')['value'].agg(lambda x: x.mode().iloc[0])

    # get the mode image by using segment indexes
    mode_img = np.array(mode_series)[labels1]

    return mode_img, labels1


def calculate_segmented_mean_map(img, color_guide, compactness=30, n_segments=500):
    '''
    Input: a 2-dimensional grayscale image that can have uint8, float32, or float64 data type.
    Returns: the image with segmented regions having the region mean value.
    '''

    # segment the image into regions using k-means
    labels1 = segmentation.slic(
        color_guide, compactness=compactness, n_segments=n_segments, start_label=0)

    # flatten both the array and the segments
    flat_img = img.ravel()
    flat_labels = labels1.ravel()

    print('img.shape:', img.shape, 'labels.shape:', labels1.shape)

    # create a DataFrame for grouping
    df = pd.DataFrame({'value': flat_img, 'label': flat_labels})

    # group by segment and calculate the mean
    mean_series = df.groupby('label')['value'].mean()

    # get the mean image by using segment indexes
    mean_img = np.array(mean_series)[labels1]

    return mean_img, labels1
