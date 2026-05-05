import io
import json
import os
import sys

import tensorflow as tf
import torch
from flask import Flask, render_template, request
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
DEFAULT_FIRSTNAME = "your_firstname"
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from training.inference_utils import CLASS_NAMES, predict_tf, preprocess_pil_for_torch, set_global_seed
from training.train import TorchCNN

app = Flask(__name__)
set_global_seed(42)


def load_meta(firstname: str):
    meta_path = os.path.join(ARTIFACT_DIR, f"{firstname}_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"class_names": CLASS_NAMES, "image_size": 150, "seed": 42}


def load_pt_model(firstname: str):
    path = os.path.join(ARTIFACT_DIR, f"{firstname}_model.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)
    class_names = ckpt.get("class_names", CLASS_NAMES)
    image_size = ckpt.get("image_size", 150)
    model = TorchCNN(num_classes=len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, class_names, image_size, device


def predict_pt(pil_img: Image.Image, model, class_names, image_size: int, device):
    x = preprocess_pil_for_torch(pil_img, image_size=image_size).to(device)
    with torch.no_grad():
        logits = model(x)
        pred_idx = int(logits.argmax(dim=1).cpu().item())
    return class_names[pred_idx]


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    model_choice = "tensorflow"
    firstname = DEFAULT_FIRSTNAME

    if request.method == "POST":
        model_choice = request.form.get("model_choice", "tensorflow")
        firstname = request.form.get("firstname", DEFAULT_FIRSTNAME).strip() or DEFAULT_FIRSTNAME
        file = request.files.get("image")

        if not file:
            error = "Please upload an image."
        else:
            try:
                image = Image.open(io.BytesIO(file.read())).convert("RGB")
                meta = load_meta(firstname)
                class_names = meta.get("class_names", CLASS_NAMES)
                image_size = int(meta.get("image_size", 150))

                if model_choice == "pytorch":
                    model, class_names, image_size, device = load_pt_model(firstname)
                    prediction = predict_pt(image, model, class_names, image_size, device)
                else:
                    tf_path = os.path.join(ARTIFACT_DIR, f"{firstname}_model.keras")
                    model = tf.keras.models.load_model(tf_path)
                    prediction = predict_tf(image, model, class_names, image_size)
            except Exception as exc:
                error = f"Inference error: {exc}"

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        model_choice=model_choice,
        firstname=firstname,
    )


if __name__ == "__main__":
    app.run(debug=True)
