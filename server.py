from ML.process import process
from ML.yolov5 import yolov5
import time
import warnings
#Original input should be defined in the yml config file.
config_path = "HAT-L_SRx2_ImageNet-pretrain.yml"

process(config_path,task='gray_dn',noise=50,model_path='model_zoo/004_grayDN_DFWB_s128w8_SwinIR-M_noise50.pth',folder_gt="decoded_result")

print("Object detecting")
SWINIR_result = "SWINIR_results/swinir_gray_dn_noise50"

yolov5.process_images(SWINIR_result)

print("Analysis complete")


