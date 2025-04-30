
import os
import io
import numpy as np
import tensorflow as tf
import subprocess

from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from PIL import Image


# Folder inside your workspace to store the model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = "ImageClassification"
MODEL_ZIP = os.path.join(MODEL_DIR)
MODEL_FILE = "card.h5"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

# Download model from Google Drive (if needed)
def download_folder_from_drive(folder_id, model_dir):
    if os.path.exists(model_dir) and os.listdir(model_dir):
        print(f"Folder '{model_dir}' already exists and is not empty. Skipping download.")
        return

    print("Downloading folder from Google Drive...")
    try:
        subprocess.run([
            "gdown",
            "--folder",
            f"https://drive.google.com/drive/folders/{folder_id}",
            "-O",
            model_dir
        ], check=True)
        print("Download completed successfully.")
    except subprocess.CalledProcessError as e:
        print("Download failed:", e)

FOLDER_ID = "1-2i19_YOapWIHZa8gm0Lk9YXtew8VwyP"
download_folder_from_drive(FOLDER_ID, MODEL_DIR)

model = load_model(MODEL_PATH)

# Initialize FastAPI app
app = FastAPI()


TEMPLATE_DIR = os.path.join(BASE_DIR, ".")
templates = Jinja2Templates(directory=TEMPLATE_DIR)


# Load the saved model


# Load class labels from folder names 
def load_card_classes(file_path=os.path.join(BASE_DIR, "class.txt")):
    with open(file_path, "r") as file:
        card_classes = [line.strip().lower() for line in file.readlines()]
    return card_classes

# Load the card classes (this will be the mapping from indices to card names)
card_classes = load_card_classes()



# ====== Preprocess Image ======
IMAGE_SIZE = (128, 128)

def preprocess_image(image: Image.Image):
    image = image.resize(IMAGE_SIZE)
    image = np.array(image) / 255.0  # normalize
    image = np.expand_dims(image, axis=0)  # add batch dimension
    return image

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        input_data = preprocess_image(image)

        # Make prediction
        prediction = model.predict(input_data)
        predicted_index = np.argmax(prediction)
        
        # Ensure predicted_index is within the range of class labels
        predicted_class = card_classes[predicted_index] if predicted_index < len(card_classes) else "Unknown"

        return {"prediction": predicted_class}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
