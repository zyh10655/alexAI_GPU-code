#import asyncio
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from fringe.utils.io import import_image, export_image
from fringe.utils.modifiers import ImageToArray, PreprocessHologram, ConvertToTensor
from fringe.process.gpu import AngularSpectrumSolver as AsSolver
from skimage.restoration import unwrap_phase
from PIL import Image
import os
import cv2

import ws

dtype_f = tf.float32
dtype_c = tf.complex64

Image.MAX_IMAGE_PIXELS = None
#Decoder for phase
import gc

# Custom Callback To Include in Callbacks List At Training Time
class GarbageCollectorCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
# For test
from skimage import color
class ImageToArray:
    def __init__(self, bit_depth=8, channel='gray', dtype='float32'):
        self.bit_depth = bit_depth
        self.channel = channel
        self.dtype = dtype

    def process(self, img):
        if self.channel == 'gray':
            if len(img.shape) == 3:  # Check if it's an RGB or RGBA image
                if img.shape[2] == 4:  # Check for RGBA image
                    img = img[:, :, :3]  # Drop the alpha channel
                img = color.rgb2gray(img)
        img = img.astype(self.dtype)
        img /= (2**self.bit_depth - 1)  # Normalize to [0, 1]
        return img
# For test
# TODO:Set a timeout for the decoder
def calculate_sharpness_and_best_z(hologram_folder_path, background_path, output_folder_path):
    for filename in os.listdir(hologram_folder_path):
        if filename.endswith(".png") or filename.endswith(".jpg") :
            p1 = ImageToArray(bit_depth=8, channel='gray', dtype='float32')

            print(f"Processing {filename}")
            hologram_path = os.path.join(hologram_folder_path, filename)
            # Open the image file
            hologram_check = Image.open(hologram_path)
            # Get dimensions
            h_width, h_height = hologram_check.size

            # Load background image
            bg = Image.open(background_path)

            # Resize background
            bg_resized = bg.resize((h_width, h_height))
            # Save resized background
            bg_resized_path = 'background/newbg_test.png'
            bg_resized.save(bg_resized_path)

            bg = import_image(bg_resized_path, preprocessor=p1)
            print("bg cropped")

            p2 = PreprocessHologram(background=bg)
            p3 = ConvertToTensor(dtype='complex64')
            hologram = import_image(hologram_path, preprocessor=[p1, p2, p3])
            hologram_amp = tf.math.abs(hologram)
            solver = AsSolver(shape=hologram_amp.shape, dx=1.12, dy=1.12, wavelength=405e-3)

            def calculate_sharpness(image):
                sobelx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=5)
                sobely = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=5)
                magnitude = np.sqrt(sobelx ** 2.0 + sobely ** 2.0)
                sharpness = np.var(magnitude)
                return sharpness

            #Z is Micrometres
            #904 for last example
            #3980 for raw hologram,3924/3990 for three bars
            #984 for raw hologram without upscaling
            #158000 for phone shot
            #480000,482000
            #36000，38000

            #ranges = [(34000,36000)] For 200mp test for tonight
            ranges = [(30000,40000)]
            # ranges = [(134000,136000)]
            steps = [250]
            # ranges = [(800,1600)]
            # steps = [5]
            best_sharpness_global = 0
            best_z_global = None

            for r in range(len(ranges)):
                start, end = ranges[r]
                step = steps[r]
                sharpness_values = []
                z_values = []

                for z in np.arange(start, end, step):
                    rec = solver.solve(hologram, z)
                    amp = np.abs(rec)
                    sharpness = calculate_sharpness(amp)
                    sharpness_values.append(sharpness)
                    z_values.append(z)

                local_maxima = []

                for i in range(1, len(sharpness_values) - 1):
                    if sharpness_values[i - 1] < sharpness_values[i] > sharpness_values[i + 1]:
                        local_maxima.append((z_values[i], sharpness_values[i]))

                        rec = solver.solve(hologram, z_values[i])
                        amp = np.abs(rec)
                        # plt.imshow(amp, cmap='gray')
                        # plt.title(f'Hologram at z = {z_values[i]} with sharpness {sharpness_values[i]}')

                if local_maxima:
                    best_z_local, best_sharpness_local = max(local_maxima, key=lambda item: item[1])

                    if best_sharpness_local > best_sharpness_global:
                        best_sharpness_global = best_sharpness_local
                        best_z_global = best_z_local

            # plt.plot(z_values, sharpness_values)
            # plt.xlabel('z')
            # plt.ylabel('Sharpness')
            # plt.title('Sharpness as a function of z')
            # plt.plot(best_z_global, best_sharpness_global, 'ro')

            # # After finding best_z_global, loop around it and save images
            z_range = 1000
            z_step = 1000
            for z_offset in range(-z_range, z_range + z_step, z_step):  # range from -20 to 20 with step of 5
                z_value = best_z_global + z_offset
                rec_offset = solver.solve(hologram, z_value)
                amp_offset = np.abs(rec_offset)
                phase_offset = unwrap_phase(np.angle(rec))
                amp_normalized_offset = cv2.normalize(amp_offset, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                                      dtype=cv2.CV_8U)

                phase_normalized_offset = cv2.normalize(phase_offset, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)

                output_path_amp = os.path.join(output_folder_path,
                                                  f"{os.path.splitext(filename)[0]}_decoded_z{z_value}.png")

                output_path_phase = os.path.join(output_folder_path,f"{os.path.splitext(filename)[0]}_phase_decoded_z{z_value}.png")

                cv2.imwrite(output_path_amp, amp_normalized_offset)

                cv2.imwrite(output_path_phase, phase_normalized_offset)

            print(f"Best z: {best_z_global}")
            print("Decoding process complete")
            #asyncio.run(ws.send_message_to_user("Decoding process complete"))

            best_amp_normalized = cv2.normalize(amp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                                dtype=cv2.CV_8U)
            output_path = os.path.join(output_folder_path, f"{os.path.splitext(filename)[0]}_{best_z_global}decoded.png")
            cv2.imwrite(output_path, best_amp_normalized)
            gc.collect()
            #plt.hist((hologram_amp.numpy()).flatten(), 256)
    return best_z_global, best_sharpness_global


# hologram_path = "C:/alexAI_GPU-main/experiments/HAT-L_SRx2_ImageNet-pretrain/visualization/datas"
# background_path = "C:/alexAI_GPU-main/background/bg.png"
# output_path = "C:/alexAI_GPU-main/decoded_result"
# calculate_sharpness_and_best_z(hologram_path,background_path,output_path)