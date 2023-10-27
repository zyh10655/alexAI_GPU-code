import asyncio
import gc
import os
import sys
import time

import ws
from ML.SWINIR.network_swinir import SwinIR as net
import tensorflow as tf
import warnings
from ML.decoder import calculate_sharpness_and_best_z
from ML.HAT.hat_model import *
from ML.HAT.hat_arch import *
from ML.HAT import hat_pipeline
from ML.yolov5 import yolov5

device = "gpu"
if device == "gpu":
    if len(tf.config.list_physical_devices('GPU')) > 0:
        print('GPU is up and running')
        device = "/gpu:0"
    else:
        print('No GPUs found for tensorflow. The process will run on CPU.')
        device = "/cpu:0"


def suppress_print(func):
    """
    Decorator to suppress print statements
    """

    def wrapper(*args, **kwargs):
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        result = func(*args, **kwargs)
        sys.stdout.close()
        sys.stdout = original_stdout
        return result

    return wrapper


# Apply suppress_print decorator on the functions
@suppress_print
def run_hat_pipeline(config_path):
    hat_pipeline.test_pipeline(config_path)


@suppress_print
def run_calculate_sharpness(hologram_folder_path, background_path, output_folder_path):
    calculate_sharpness_and_best_z(hologram_folder_path, background_path, output_folder_path)


async def delete_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                print(f"Skipping directory {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def process(config_path, task='color_dn', scale=1, noise=15, jpeg=40, training_patch_size=128,
            large_model=False,
            model_path='model_zoo/swinir/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth',
            folder_lq=None, folder_gt=None, tile=1024, tile_overlap=16,run_hat=False):
    asyncio.run(ws.send_message_to_user("ENHANCING DATA"))
    print("ENHANCING DATA")
    # Condition to run the hat pipeline and set the hologram folder path
    if run_hat:
        run_hat_pipeline(config_path)
        print("Decode with Upscaled image")
        hologram_folder_path = "experiments/HAT-L_SRx2_ImageNet-pretrain/visualization/datas"
    else:
        print("Decode without Upscale")
        hologram_folder_path = "datasets"
    # Change here to fit the whole pipeline
    background_path = "background/bg.png"
    output_folder_path = "decoded_result"
    asyncio.run(ws.send_message_to_user("DECODING..."))
    print("DECODING ...")
    run_calculate_sharpness(hologram_folder_path, background_path, output_folder_path)
    # This is for SWINIR
    #asyncio.run(ws.send_message_to_user("SWINIR Denoising, please wait...."))
    gc.collect()
    # print("SWINIR Denoising")
    # SWINIR command
    asyncio.run(ws.send_message_to_user("Object detecting"))
    yolov5.process_images(output_folder_path, position_threshold=1, confidence_threshold=0.2)
    print("Object Detection Complete!")
    command = 'python ML\\SWINIR\\main_test_swinir.py'
    # add arguments to command
    command += f' --task {task} --scale {scale} --noise {noise} --jpeg {jpeg}'
    command += f' --training_patch_size {training_patch_size} --model_path "{model_path}"'
    command += f' --tile_overlap {tile_overlap}'

    if large_model:
        command += ' --large_model'

    if folder_lq is not None:
        command += f' --folder_lq "{folder_lq}"'

    if folder_gt is not None:
        command += f' --folder_gt "{folder_gt}"'

    if tile is not None:
        command += f' --tile {tile}'
    asyncio.run(ws.send_message_to_user("Argument passing for Denoising model"))
    os.system(command)

    def delete_files_in_folder_Inside(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    print(f"Skipping directory {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

    delete_files_in_folder_Inside("ML/yolov5/cropping")
    asyncio.run(ws.send_message_to_user("resurt generated!"))
    print("Denoising Complete!")