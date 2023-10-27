import asyncio
import json
import threading
from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse

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
result_json_file_path = "ML/yolov5/json_files"
noise = 50
SWINIR_task = "gray_dn"
#SWINIR_task = "color_dn"
SWINIR_result = f'SWINIR_results/swinir_{SWINIR_task}_noise{noise}'
model_path= f'model_zoo/004_grayDN_DFWB_s128w8_SwinIR-M_noise{noise}.pth'
HAT_path = "experiments/HAT-L_SRx2_ImageNet-pretrain/visualization/datas"
operating_system = "Windows"
#Unit is second,should smaller that the main server's timeout setting (10 sec in this example)
timeout_time = 500000

@app.on_event("startup")
async def startup_event():
    await clean_up_and_reset()

@app.get("/serve_result")
async def serve_result():
    global GPU_status
    if GPU_status == "idle" or GPU_status == "busy":
        raise HTTPException(status_code=403, detail="Error: GPU is idle")

    GPU_status = "fetching"
    try:
        json_files = [f for f in os.listdir(result_json_file_path) if f.endswith('.json')]

        if not json_files:
            raise HTTPException(status_code=400, detail="No JSON files found")

        first_json_file = json_files[0]
        full_path = os.path.join(result_json_file_path, first_json_file)

        with open(full_path, 'r') as f:
            data = json.load(f)

        return JSONResponse(content=data)

    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="File not found")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Error decoding JSON")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



#import docker

@app.get("/check_status")
async def check_status():
    return {"status": GPU_status}


@app.get("/change_status")
async def change_status():
    global GPU_status
    decoded_path = "decoded_result/"
    await delete_files_in_folder(HAT_path)
    await delete_files_in_folder(decoded_path)
    await delete_files_in_folder(SWINIR_result)
    await delete_files_in_folder("datasets")
    await delete_files_in_folder("ML/yolov5/cropping")
    await delete_files_in_folder("ML/yolov5/json_files")
    GPU_status = "idle"
    return {"status": GPU_status}


import subprocess


def restart_computer():
    global operating_system
    if operating_system == "Windows":
        subprocess.run(["shutdown", "/r"])
    else:
        subprocess.run(["sudo", "shutdown", "-r", "now"])




from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor()


def runML_blocking(config_path):
    global GPU_status,timeout_time
    timer = threading.Timer(timeout_time, restart_computer)
    timer.start()
    try:
        print("exist in ML_thread: ", ws.ws_connection_to_GPU)
        print("\nStarting additional code...")
        #Change here to the conresponding location
        SWINIR_input = "ML/yolov5/cropping"
        print("Object detecting")
        process(config_path, task=SWINIR_task, noise=noise, model_path=model_path,
                folder_gt=SWINIR_input,run_hat=False)

        """
        Commend this out for test
        yolov5.process_images(SWINIR_result)
        """
        print("cleaning caches")
        GPU_status = "finished"
        timer.cancel()
        print("Machine Learning process complete  : )")

    except Exception as e:
        print(f"An error occurred: {e}")



@app.post("/reset")
async def clean_up_and_reset():
    global GPU_status
    await ws.send_message_to_user("Cleaning caches")
    #asyncio.run(ws.send_message_to_user("Cleaning caches"))
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

            file_path = Path(SWINIR_result) / crop_image_name
            print(file_path)  # This should print the relative path

            if not file_path.exists():
                raise HTTPException(status_code=403, detail="Error: File does not exist")



            return FileResponse(str(file_path))
    # If we've gone through all JSON files and still haven't found the uuid
    raise HTTPException(status_code=403, detail="Error: UUID not found in any JSON")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, ssl_keyfile="server.key", ssl_certfile="server.crt")
