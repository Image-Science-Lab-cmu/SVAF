import skimage
import time
import ctypes
import numpy as np
import cv2

from arena_api.buffer import BufferFactory
from arena_api.system import system
from arena_api.__future__.save import Writer
from arena_api.enums import PixelFormat

from arena_api._xlayer.xarena._ximagefactory import _xImagefactory
from arena_api.buffer import _Buffer

# import numba
# from numba import jit, njit, prange
import copy

TAB1 = "  "
TAB2 = "    "
TAB3 = "      "

class PolarizedCamera:
    def __init__(self, stream=False):
        devices = PolarizedCamera.create_devices_with_tries()
        self.device = system.select_device(devices)
        self.set_features(stream=stream)
        self.initialize_camera()

    @staticmethod
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
 
    def initialize_camera(self):

        # Get device stream nodemap
        tl_stream_nodemap = self.device.tl_stream_nodemap

        # Set buffer handling mode to NewestOnly
        tl_stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"

        # Enable stream auto negotiate packet size
        tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True

        # Enable stream packet resend
        tl_stream_nodemap['StreamPacketResendEnable'].value = True

        # Get nodes ---------------------------------------------------------------
        nodes = self.device.nodemap.get_node(['PixelFormat'])

        # Retrieve enumeration entries and check which polarized angles pixel format is supported
        pixel_format = ''
        for pixel_format_name in nodes['PixelFormat'].enumentry_names:
            if (pixel_format_name == 'PolarizedAngles_0d_45d_90d_135d_BayerRG8'):
                pixel_format = pixel_format_name
            # elif (pixel_format_name == 'PolarizedAngles_0d_45d_90d_135d_Mono8'):
            # 	pixel_format = pixel_format_name

        if pixel_format == '':
            print("\tError - This example requires PolarizedAngles_0d_45d_90d_135d_* pixel formats")
            return

        # Get node values that will be changed in order to return their values at
        # the end of the example
        pixel_format_initial_value = nodes['PixelFormat'].value
        print(f'{TAB1}Setting Pixel Format to {pixel_format}')
        nodes['PixelFormat'].value = pixel_format

        self.nodes = nodes
        self.pixel_format_initial_value = pixel_format_initial_value

    def set_features(self, exposure_time=47178.6, gain=18.0, stream=False):
        nodemap = self.device.nodemap
        nodemap['AcquisitionFrameRateEnable'].value = True
        nodemap['ExposureAuto'].value = 'Off'
        if stream:
            nodemap['AcquisitionMode'].value = 'Continuous'
        nodes_values_to_set = {'AcquisitionFrameRate': 21.020768519297064,
                            'BlackLevel': 0.0,
                            'ExposureTime': exposure_time,
                            'GainAuto': 'Off',
                            'Gain': gain}  # 26.0}

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
    
    @staticmethod
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
    
    def save_grid_view(self, src_width, src_height, src_pixel_format, polarized_images, ID, savefolder, num_channels):

        # 2x2 info
        dst_2x2_width = src_width * 2
        dst_2x2_height = src_height * 2
        dst_2x2_pixel_format = PixelFormat.BayerRG8 if src_pixel_format == PixelFormat.PolarizedAngles_0d_45d_90d_135d_BayerRG8 else PixelFormat.Mono8
        dst_2x2_step = PixelFormat.get_bits_per_pixel(dst_2x2_pixel_format) // 8

        dst_2x2_stride = dst_2x2_width * dst_2x2_step
        dst_2x2_data_size = dst_2x2_width * dst_2x2_height * dst_2x2_step

        # Allocate space for 2x2 grid
        dst_data = (ctypes.c_ubyte * dst_2x2_data_size)()

        # Reference set up to starting position of each quadrant of
        # destination 2x2 grid to write to
        dst_top_left = 0
        dst_top_right = dst_top_left + (dst_2x2_stride / 2)
        dst_bottom_left = dst_top_left + (dst_2x2_data_size / 2)
        dst_bottom_right = dst_top_left + \
            (dst_2x2_data_size / 2) + (dst_2x2_stride / 2)

        # Precompute deinterleave indices
        src_h = polarized_images[0].height
        src_w = polarized_images[0].width
        src_step = PixelFormat.get_bits_per_pixel(
            polarized_images[0].pixel_format) // 8
        dst_step = int(dst_2x2_step)
        dst_half_stride = int(dst_2x2_stride/2)
        src_step = int(src_step)
        src_index_list = np.arange(
            0, src_h * src_w * src_step, src_step)
        dst_index_add_list = np.arange(
            0, src_h * dst_half_stride, dst_half_stride)
        dst_index_add_list = np.repeat(dst_index_add_list, src_w)

        dst_index_lists = []
        starting_positions = [dst_top_left, dst_top_right,
                    dst_bottom_left, dst_bottom_right]
        for i in range(len(polarized_images)):
            dst_offset = int(starting_positions[i])
            dst_index_list = np.arange(int(dst_offset), int(
                dst_offset) + src_h * src_w * dst_step, dst_step)
            dst_index_list = dst_index_list + dst_index_add_list
            dst_index_lists.append(dst_index_list)

        print(f'{TAB2}Writing image buffers to a 2x2 grid')
        for i in range(len(polarized_images)):
            # Grab image from array of polarized images
            img = polarized_images[i]

            # Write image to 2x2 grid
            # write_to_2x2_grid(
            #     img, dst_data, starting_positions[i], dst_2x2_step, dst_2x2_stride / 2)
            PolarizedCamera.write_to_2x2_grid_optimized(
                img, dst_data, src_index_list, dst_index_lists[i])

        uint8_ptr = ctypes.POINTER(ctypes.c_ubyte)
        dst_data_ptr = ctypes.cast(dst_data, uint8_ptr)

        # Save the 2x2
        create_buffer = BufferFactory.create(
            dst_data_ptr, dst_2x2_data_size, dst_2x2_width, dst_2x2_height, dst_2x2_pixel_format)
        output_buffer = BufferFactory.convert(create_buffer, PixelFormat.BGR8 if src_pixel_format ==
                                            PixelFormat.PolarizedAngles_0d_45d_90d_135d_BayerRG8 else PixelFormat.Mono8)

        # writer_jpg = Writer.from_buffer(output_buffer)
        # writer_jpg.save(output_buffer, f'{savefolder}/polarized_2x2_tile_{ID}.jpg')
        
        bytes_per_pixel = int(len(output_buffer.data) /
                              (output_buffer.width * output_buffer.height))
        array = (ctypes.c_ubyte * num_channels * output_buffer.width *
                 output_buffer.height).from_address(ctypes.addressof(output_buffer.pbytes))
        image = np.ndarray(buffer=array, dtype=np.uint8, shape=(
            int(output_buffer.height), int(output_buffer.width), bytes_per_pixel))
        savepath = f'{savefolder}/polarized_2x2_tile_{ID}.jpg'
        skimage.io.imsave(savepath, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        print(f'{TAB2}Save 2x2 image to {savepath}')

        # Clean up
        BufferFactory.destroy(output_buffer)
        BufferFactory.destroy(create_buffer)
                
    @staticmethod
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

    def capture_dp_image(self, savefolder='./', saveimages=True,savegridview=False):
        """
        Demonstrates acquisition and processing of polarized angles image data:
        (1) configures camera to appropriate supported pixel format
        (2) acquires a polarized input image
        (3) splits the polarized input image into 4 separate images
        (4) saves images into a 2x2 grid image
        (5) saves the 4 separate images to disk
        """

        # Grab and save an image buffer -------------------------------------------
        print(f'{TAB1}Starting stream')
        with self.device.start_stream(1):
            print(f'{TAB2}Acquire image')
            ID = str(int(time.time()))
            image_buffer = self.device.get_buffer()

            src_pixel_format = image_buffer.pixel_format
            if src_pixel_format != PixelFormat.PolarizedAngles_0d_45d_90d_135d_BayerRG8 and src_pixel_format != PixelFormat.PolarizedAngles_0d_45d_90d_135d_Mono8:
                print(
                    "\tError - Input image pixel format [{}] is a non-polarized format".format(src_pixel_format))
                return
            
            if src_pixel_format == PixelFormat.PolarizedAngles_0d_45d_90d_135d_BayerRG8:
                num_channels = 3
            else:
                num_channels = 1

            # Splits the polarized image into 4 images:
            #    This function separates the four channels of the
            #    polarized pixel format and returns an array of
            #    pointers to the separated images.
            print(f'{TAB2}Splitting 4-channel pixel format into array of images')
            polarized_images = PolarizedCamera.split_channels_optimized(image_buffer)
            # polarized_images = BufferFactory.split_channels(image_buffer)
            print('Polarized images length: ', len(polarized_images))

            # Saves each image to disk
            #    Converts each of the images into a
            #    displayable format and saves the image as a JPEG
            # print(f'{TAB2}Saving each image to disk')

            degrees = ['0', '45', '90', '135']
            images = []
            for i in range(len(polarized_images)):
                # for i in [0, 2]:
                # Convert image to displayable format
                img = polarized_images[i]
                convert_buffer = BufferFactory.convert(img, PixelFormat.BGR8 if src_pixel_format ==
                                                    PixelFormat.PolarizedAngles_0d_45d_90d_135d_BayerRG8 else PixelFormat.Mono8)

                bytes_per_pixel = int(len(convert_buffer.data) /
                                    (convert_buffer.width * convert_buffer.height))
                array = (ctypes.c_ubyte * num_channels * convert_buffer.width *
                                convert_buffer.height).from_address(ctypes.addressof(convert_buffer.pbytes))
                image = np.ndarray(buffer=array, dtype=np.uint8, shape=(
                    int(convert_buffer.height), int(convert_buffer.width), bytes_per_pixel))

                images.append(copy.deepcopy(image))
                print(image.dtype, image.min(), image.max())

                if saveimages:
                    savepath = f'{savefolder}/polarized_{ID}_{degrees[i]}°.jpg'
                    skimage.io.imsave(savepath, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    print('Saved image to ', savepath)
                    # Save image
                    # save_image(convert_buffer, savepath)

                # Clean up
                BufferFactory.destroy(convert_buffer)

            images = np.array(images)
            
            if savegridview:
                self.save_grid_view(image_buffer.width, image_buffer.height,
                                    src_pixel_format, polarized_images, ID, savefolder, num_channels)

            self.device.requeue_buffer(image_buffer)
            self.device.stop_stream()
            print(f'{TAB1}Stream stopped')
            
        return images

    def __del__(self):
	    # Reset the pixel format to its initial value
        self.nodes['PixelFormat'].value = self.pixel_format_initial_value
        system.destroy_device()
        print(f'{TAB1}Destroyed all created devices')
