import asyncio
import json

from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse

import ws
from ws import router as ws_router
from ML.process import process
from ML.process import delete_files_in_folder
from ML.yolov5 import yolov5
import os
from pathlib import Path
import shutil

app = FastAPI()
app.include_router(ws_router)

GPU_status = "idle"
current_user_id = ""
current_uuid = ""
PHOTO_UPLOAD_DIR = "ML/yolov5/cropping"
config_path = "HAT-L_SRx2_ImageNet-pretrain.yml"


@app.get("/check_status")
async def check_status():
    return {"status": GPU_status}


@app.get("/change_status")
async def change_status():
    global GPU_status
    SWINIR_result = "SWINIR_results/swinir_gray_dn_noise50"
    HAT_path = "experiments/HAT-L_SRx2_ImageNet-pretrain/visualization/datas"
    decoded_path = "decoded_result/"
    await delete_files_in_folder(HAT_path)
    await delete_files_in_folder(decoded_path)
    await delete_files_in_folder(SWINIR_result)
    await delete_files_in_folder("datasets")
    await delete_files_in_folder("ML/yolov5/cropping")
    await delete_files_in_folder("ML/yolov5/json_files")
    GPU_status = "idle"
    return {"status": GPU_status}

#TODO:Write a API for Geting result of json file,if not then tell out 403 with message like "Still process"
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor()


def runML_blocking(config_path):
    global GPU_status

    print("exist in ML_thread: ", ws.ws_connection_to_GPU)
    print("\nStarting additional code...")
    process(config_path, task='gray_dn', noise=50, model_path='model_zoo/004_grayDN_DFWB_s128w8_SwinIR-M_noise50.pth',
            folder_gt="decoded_result")
    asyncio.run(ws.send_message_to_user("Object detecting"))
    print("Object detecting")
    SWINIR_result = "SWINIR_results/swinir_gray_dn_noise50"
    """
    Commend this out for test
    yolov5.process_images(SWINIR_result)
    """

    """
    This command for test
    """
    #Change here to modify the url for YOLO
    yolov5.process_images("decoded_result")
    #yolov5.process_images(SWINIR_result)
    print("Machine Learning process complete  : )")


@app.post("/reset")
async def clean_up_and_reset():
    global GPU_status
    await ws.send_message_to_user("Cleaning caches")
    #asyncio.run(ws.send_message_to_user("Cleaning caches"))
    SWINIR_result = "SWINIR_results/swinir_gray_dn_noise50"
    HAT_path = "experiments/HAT-L_SRx2_ImageNet-pretrain/visualization/datas"
    decoded_path = "decoded_result/"
    await delete_files_in_folder(HAT_path)
    await delete_files_in_folder(decoded_path)
    await delete_files_in_folder(SWINIR_result)
    await delete_files_in_folder("datasets")
    await delete_files_in_folder("ML/yolov5/cropping")
    await delete_files_in_folder("ML/yolov5/json_files")
    GPU_status = "idle"
    return {GPU_status}

@app.post("/upload_photo")
async def upload_photo(
        photo: UploadFile = File(...),
        user_id: str = Form(...),
        uuid: str = Form(...)
):
    global GPU_status

    if GPU_status != "idle":
        raise HTTPException(status_code=503, detail="GPU is busy, please try again later")

    GPU_status = "busy"
    global current_user_id
    global current_uuid
    current_user_id = user_id
    current_uuid = uuid
    photo_dir = 'datasets/'

    os.makedirs(photo_dir, exist_ok=True)
    photo_path = photo_dir
    while os.path.exists(photo_path):
        photo_path = os.path.join(photo_path, photo.filename)
        break
    try:
        with open(photo_path, "wb") as buffer:
            shutil.copyfileobj(photo.file, buffer)
    except Exception as e:
        SWINIR_result = "SWINIR_results/swinir_gray_dn_noise50"
        HAT_path = "experiments/HAT-L_SRx2_ImageNet-pretrain/visualization/datas"
        decoded_path = "decoded_result/"
        await delete_files_in_folder(HAT_path)
        await delete_files_in_folder(decoded_path)
        await delete_files_in_folder(SWINIR_result)
        await delete_files_in_folder("datasets")
        await delete_files_in_folder("ML/yolov5/cropping")
        await delete_files_in_folder("ML/yolov5/json_files")

        GPU_status = "idle"
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")

    print("exist in main thread:", ws.ws_connection_to_GPU)
    executor.submit(runML_blocking, config_path)

    return {"message": "Photo uploaded successfully"}


# have nothing to do with physical storage
# @app.get("/serve_photo/{holo_uuid}/{uuid}", name="serve_photo")
@app.get("/serve_photo/{uuid}", name="serve_photo")
async def serve_photo(uuid: str):
    # Step 1: Read JSON files from the "json_files" folder
    json_folder = 'ML/yolov5/json_files'
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

    for json_file in json_files:
        json_file_path = Path(json_folder) / json_file

        with open(json_file_path, 'r') as f:
            uuid_to_data = json.load(f)

        # Step 2: Check if uuid is in this JSON file
        if uuid in uuid_to_data:
            image_details = uuid_to_data[uuid]
            # crop_image_path = image_details.get("Crop_image_path", "")
            crop_image_name = image_details.get("Crop_image_name", "")


            # Step 3: Combine them into dir
            # base_dir = Path(PHOTO_UPLOAD_DIR)
            # print(base_dir)
            # file_path = base_dir / Path(crop_image_path) / crop_image_name
            # absolute_path = file_path.resolve()
            # print(absolute_path)
            #
            # if not absolute_path.exists():
            #     raise HTTPException(status_code=403, detail="Error: File does not exist")
            #
            # return FileResponse(absolute_path)

            file_path = Path("ML/yolov5/cropping") / crop_image_name
            print(file_path)  # This should print the relative path

            if not file_path.exists():
                raise HTTPException(status_code=403, detail="Error: File does not exist")



            return FileResponse(str(file_path))
    # If we've gone through all JSON files and still haven't found the uuid
    raise HTTPException(status_code=403, detail="Error: UUID not found in any JSON")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, ssl_keyfile="server.key", ssl_certfile="server.crt")
