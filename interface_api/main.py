from fastapi import FastAPI, UploadFile, File
from typing import List
import os
import numpy as np
import onnxruntime
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

app = FastAPI()

def list_models() -> List[str]:
    models = []
    for filename in os.listdir("models"):
        if filename.endswith(".onnx"):
            models.append(filename)
    return models

@app.get("/models")
def get_available_models():
    return {"models": list_models()}

@app.post("/inference")
async def infer(model_name: str = "yolov7-custom.onnx", file: UploadFile = File(...)):
    # Load model
    ort_session = onnxruntime.InferenceSession(f"models/{model_name}")
    img = Image.open(file.file)
    img = img.convert("RGB")
    img_data = np.array(img)
    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: img_data})
    return {"predictions": outputs}

@app.post("/inference2")
async def infer2(model_name: str = "my_model.onnx", file: UploadFile = File(...)):
    ort_session = onnxruntime.InferenceSession(f"models/{model_name}")

    # Read image file
    img = Image.open(file.file)
    img = img.convert("RGB")
    img_data = np.array(img)

    # Perform inference
    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: img_data})

    # Return results
    return {"predictions": outputs}

# Endpoint for inference
@app.post("/inference_with_overlay")
async def infer_with_overlay(model_name: str = "yolov7-custom.onnx", file: UploadFile = File(...)):
    # Load model
    ort_session = onnxruntime.InferenceSession(f"models/{model_name}")

    # Read image file
    img = Image.open(file.file)
    img = img.convert("RGB")
    img_data = np.array(img)

    # Perform inference
    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: img_data})

    # Process output to draw bounding boxes on the image
    bboxes = outputs[0]  

    img_with_overlay = draw_boxes(img, bboxes)
    img_with_overlay_path = "temp_overlay.png"
    img_with_overlay.save(img_with_overlay_path)

    return FileResponse(img_with_overlay_path, media_type="image/png")

def draw_boxes(image: Image, boxes: np.ndarray) -> Image:
    draw = ImageDraw.Draw(image)
    for box in boxes:
        xmin, ymin, xmax, ymax = box[:4]
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
    return image
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
