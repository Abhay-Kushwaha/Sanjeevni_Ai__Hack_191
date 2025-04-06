import os
import torch
import numpy as np
import tensorflow as tf
from PIL import Image
import torchvision.transforms.functional as TF
from flask import Blueprint, request, jsonify, render_template
from werkzeug.utils import secure_filename
from keras.models import load_model
import h5py

# Load models
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_torch = torch.load("models/brain/brain_tumor_segmentor.pt", map_location=DEVICE)
# model_tf = load_model("models/brain/brain_model.h5")
with h5py.File("models/brain/brain_model.h5", 'r+') as f:
    layer_names = list(f['model_weights'].keys())
    for name in layer_names:
        if '/' in name:
            new_name = name.replace('/', '_')
            f['model_weights'].move(name, new_name)
model = tf.keras.models.load_model('models/brain/brain_model.h5', compile=False)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


class BrainTumorClassifier:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, img, threshold=0.5):
        self.model.eval()
        image_tensor = torch.Tensor(TF.to_tensor(img)).view((-1, 1, 512, 512)).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor).detach().cpu().numpy()
        
        output = (output > threshold).astype(np.uint8)
        return output.reshape((512, 512))

classifier = BrainTumorClassifier(model_torch, DEVICE)

def predict_brain(img):
    img = np.asarray(img, dtype=np.float32) / 255.0
    img = img.reshape((1, *img.shape))
    class2label = {0: "No Tumor", 1: "Pituitary Tumor", 2: "Meningioma Tumor", 3: "Glioma Tumor"}
    prediction = model.predict(img)
    # prediction = model_tf.predict(img)
    return class2label[np.argmax(prediction)]

# Flask Blueprint
brain_tumor_bp = Blueprint("brain_tumor", __name__)

UPLOAD_FOLDER = "uploads/"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
