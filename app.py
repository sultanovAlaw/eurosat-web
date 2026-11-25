import os
import io
import zipfile
import urllib.request
import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="EuroSAT 99% Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "eurosat_99_quant.onnx")
ZIP_PATH = os.path.join(BASE_DIR, "eurosat_99_quant.zip")
MODEL_URL = "https://github.com/sultanovAlaw/eurosat-web/releases/download/v1.0/eurosat_99_quant.zip"

if not os.path.exists(MODEL_PATH):
    print("Model arxivi GitHub Release'dan yuklanmoqda...")
    urllib.request.urlretrieve(MODEL_URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(BASE_DIR)
    os.remove(ZIP_PATH)
    print("Model arxivdan chiqarildi va tayyor!")

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224), Image.Resampling.BICUBIC)
    arr = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open(os.path.join(BASE_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = preprocess_image(image)

    outputs = session.run(None, {input_name: input_tensor})[0][0]
    exp_scores = np.exp(outputs - np.max(outputs))
    probs = exp_scores / np.sum(exp_scores)

    results = [
        {"class": CLASSES[i], "confidence": round(float(probs[i]) * 100, 2)}
        for i in range(len(CLASSES))
    ]
    results.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "top_class": results[0]["class"],
        "confidence": results[0]["confidence"],
        "all_predictions": results
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
