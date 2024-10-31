import base64
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import subprocess
import os

app = FastAPI()

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    # Save the uploaded image to a file
    input_image_path = "dataset/images/train/input_image.jpg"
    with open(input_image_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Run YOLO detection using subprocess
    command = [
        'python', 'yolo/yolov7-main/detect.py',
        '--weights', 'yolo/yolov7-main/best.pt',
        '--conf-thres', '0.20',
        '--img-size', '640',
        '--source', input_image_path,
        '--project', 'out/',
        '--name', 'fixed_folder',
        '--exist-ok',
        '--save-txt',  # Save detections to a text file
    ]
    subprocess.run(command)

    # Load the result image
    output_image_path = 'out/fixed_folder/input_image.jpg'
    if not os.path.exists(output_image_path):
        return JSONResponse(content={"message": "No output image found."})

    # Read the output image
    output_image = cv2.imread(output_image_path)
    height, width, _ = output_image.shape

    # Load the detections from the text file
    detections_file = 'out/fixed_folder/labels/input_image.txt'  # Adjust path if necessary
    if not os.path.exists(detections_file):
        return JSONResponse(content={"message": "No detections file found."})

    cropped_images_base64 = []
    saved_classes = set()  # To keep track of saved class IDs

    with open(detections_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        # Read values without confidence
        class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.split())

        # Convert normalized values to pixel values
        x_center *= width
        y_center *= height
        bbox_width *= width
        bbox_height *= height

        # Calculate the bounding box coordinates
        x1 = int(x_center - bbox_width / 2)
        y1 = int(y_center - bbox_height / 2)
        x2 = int(x_center + bbox_width / 2)
        y2 = int(y_center + bbox_height / 2)

        # Ensure coordinates are within the image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        # Crop the image using the bounding box coordinates
        cropped_img = output_image[y1:y2, x1:x2]

        # Save the cropped image only if its class ID hasn't been saved yet
        if int(class_id) not in saved_classes:
            # Convert cropped image to Base64
            _, buffer = cv2.imencode('.jpg', cropped_img)
            cropped_image_base64 = base64.b64encode(buffer).decode('utf-8')
            cropped_images_base64.append(cropped_image_base64)

            # Add class ID to saved_classes
            saved_classes.add(int(class_id))

    return JSONResponse(content={
        "message": "Detection successful.",
        "cropped_images": cropped_images_base64
    })
