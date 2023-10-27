import asyncio
import os
import json
from PIL import Image
import torch
import re
from collections import defaultdict
import uuid as uuid_lib
import time
import ws


def process_images(img_dir,confidence_threshold,position_threshold):
        uuid_pattern = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.I)
        yolo_model_path = "ML/yolov5/best_object.pt"
        yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_model_path)

        cropping_folder = 'ML/yolov5/cropping'
        json_folder = 'ML/yolov5/json_files'
        Image.MAX_IMAGE_PIXELS = None

        if not os.path.exists(img_dir):
            print(f"Directory '{img_dir}' does not exist!")
            return

        if not os.path.exists(json_folder):
            os.makedirs(json_folder)

        all_detections = []

        # 1. Collect All Detections
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            match = uuid_pattern.search(img_name)
            if not match:
                print(f"No UUID found in filename {img_name}. Skipping.")
                continue
            input_uuid = match.group()
            yolo_result = yolo_model(img_path)
            all_img = yolo_result.pandas().xyxy[0]
            for index, row in all_img.iterrows():
                detection = {
                    "image": img_name,
                    "image_path": img_path,
                    "uuid": input_uuid,
                    "position": (row['xmin'], row['ymin'], row['xmax'], row['ymax']),
                    "confidence": row['confidence'],
                    "class_name": row['name']
                }
                all_detections.append(detection)

        # 2. Group Detections by Position
        def are_positions_close(pos1, pos2, threshold=position_threshold):
            return all(abs(p1 - p2) < threshold for p1, p2 in zip(pos1, pos2))

        grouped_detections = defaultdict(list)
        for detection in all_detections:
            found_group = False
            for key_position in grouped_detections:
                if are_positions_close(detection["position"], key_position):
                    grouped_detections[key_position].append(detection)
                    found_group = True
                    break
            if not found_group:
                grouped_detections[detection["position"]].append(detection)

        # 3. Filter for Highest Confidence
        filtered_detections = []
        for _, detections in grouped_detections.items():
            best_detection = max(detections, key=lambda x: x["confidence"])
            if best_detection['confidence'] >= confidence_threshold:  # Added this line to check the confidence score
                filtered_detections.append(best_detection)

        # 4. Process & Save
        uuid_to_data = defaultdict(dict)
        data = {}  # Initialize data to an empty dictionary
        for detection in filtered_detections:
            timestamp = str(int(time.time() * 1000))  # unique identifier based on the current time in milliseconds
            OG_img = Image.open(detection["image_path"])
            left, top, right, bottom = detection["position"]
            img_res = OG_img.crop((left, top, right, bottom))
            class_name = detection["class_name"]
            img_crop_name = f"{os.path.splitext(detection['image'])[0]}_{class_name}_{timestamp}.jpg"
            crop_uuid = str(uuid_lib.uuid4())
            uuid_to_data[detection["uuid"]][crop_uuid] = {
                "Crop_uuid": crop_uuid,
                "Crop_image_path": "/measurement/" + class_name,
                "Crop_image_name": img_crop_name,
                "Type": class_name,
                "Confidence": int(detection['confidence'] * 100)
            }
            cropped_file = os.path.join(cropping_folder, img_crop_name)
            img_res.save(cropped_file, 'PNG')

        for uuid, data in uuid_to_data.items():
            output_filename = os.path.join(json_folder, f"{uuid}_data.json")
            with open(output_filename, 'w') as f:
                json.dump(data, f, indent=4)
        print(data)
        #asyncio.run(ws.finished(data))
    # Adjust this to the path of your image directory
# img_directory = "C:/alexAI_GPU-main/decoded_result"
# process_images(img_directory,position_threshold=1,confidence_threshold=0.01)
