from fastapi import FastAPI
from pydantic import BaseModel
import torch
from PIL import Image
import io

# Initialize FastAPI app
app = FastAPI()

# Load YOLOv10 model (replace this with your specific model loading code)
model = torch.hub.load('ultralytics/yolov10', 'yolov10s')  # change to yolov10 if necessary

# Define the prediction request body
class Item(BaseModel):
    image: str

# Define an endpoint for prediction
@app.post("/predict/")
async def predict(item: Item):
    # Convert image string to image
    image_data = io.BytesIO(bytes(item.image, 'utf-8'))
    image = Image.open(image_data)

    # Perform prediction (YOLOv10 inference)
    results = model(image)
    return {"predictions": results.pandas().xywh[0].to_dict(orient="records")}


