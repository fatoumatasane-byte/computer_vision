import json
import os

import gradio as gr
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# -----------------------------
# Config
# -----------------------------
PT_MODEL_PATH = "fatima_model.pth"
TF_MODEL_PATH = "fatima_model.keras"
META_PATH = "fatima_meta.json"

DEFAULT_CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
DEFAULT_IMAGE_SIZE = 150
DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_meta():
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        class_names = meta.get("class_names", DEFAULT_CLASS_NAMES)
        image_size = int(meta.get("image_size", DEFAULT_IMAGE_SIZE))
        mean = meta.get("imagenet_mean", DEFAULT_MEAN)
        std = meta.get("imagenet_std", DEFAULT_STD)
        return class_names, image_size, mean, std
    return DEFAULT_CLASS_NAMES, DEFAULT_IMAGE_SIZE, DEFAULT_MEAN, DEFAULT_STD


CLASS_NAMES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD = load_meta()


class TorchCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 18 * 18, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_pytorch_model():
    ckpt = torch.load(PT_MODEL_PATH, map_location=device)
    class_names = ckpt.get("class_names", CLASS_NAMES)
    image_size = int(ckpt.get("image_size", IMAGE_SIZE))
    model = TorchCNN(num_classes=len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, class_names, image_size


def preprocess_pytorch(image: Image.Image, image_size: int):
    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    x = tfm(image.convert("RGB")).unsqueeze(0)
    return x.to(device)


def load_tensorflow_model():
    return tf.keras.models.load_model(TF_MODEL_PATH)


def preprocess_tensorflow(image: Image.Image, image_size: int):
    image = image.convert("RGB").resize((image_size, image_size))
    x = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    x = (x - mean) / std
    return np.expand_dims(x, axis=0)


pt_model = None
pt_class_names = None
pt_image_size = None
tf_model = None


def predict(model_choice, image):
    global pt_model, pt_class_names, pt_image_size, tf_model

    if image is None:
        return "Please upload an image.", "", {}

    try:
        pil_img = image if isinstance(image, Image.Image) else Image.fromarray(image)

        if model_choice == "PyTorch":
            if pt_model is None:
                pt_model, pt_class_names, pt_image_size = load_pytorch_model()

            x = preprocess_pytorch(pil_img, pt_image_size)
            with torch.no_grad():
                logits = pt_model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            label = pt_class_names[pred_idx]
            confidence = float(probs[pred_idx])
            details = {pt_class_names[i]: float(probs[i]) for i in range(len(pt_class_names))}
            return f"Prediction: {label}", f"Confidence: {confidence:.2%}", details

        if tf_model is None:
            tf_model = load_tensorflow_model()
        x = preprocess_tensorflow(pil_img, IMAGE_SIZE)
        probs = tf_model.predict(x, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        label = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
        details = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
        return f"Prediction: {label}", f"Confidence: {confidence:.2%}", details

    except Exception as exc:
        return f"Inference error: {exc}", "", {}


CUSTOM_CSS = """
.gradio-container { max-width: 1050px !important; }
.main-card {
  border-radius: 20px;
  padding: 18px;
  background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 45%, #1d4ed8 100%);
  color: white;
}
.main-title { font-size: 30px; font-weight: 800; margin-bottom: 6px; }
.subtitle { color: #dbeafe; font-size: 14px; }
.badge {
  display: inline-block;
  padding: 6px 10px;
  margin-right: 8px;
  border-radius: 999px;
  background: rgba(255,255,255,0.18);
  font-size: 12px;
}
"""


with gr.Blocks(theme=gr.themes.Soft(), css=CUSTOM_CSS, title="Intel Classifier") as demo:
    gr.HTML(
        """
        <div class="main-card">
          <div class="main-title">Intel Image Classification</div>
          <div class="subtitle">Choose a model, upload an image, and get the predicted class.</div>
          <div style="margin-top:10px;">
            <span class="badge">PyTorch + TensorFlow</span>
            <span class="badge">6 Classes</span>
            <span class="badge">Image Size: 150x150</span>
          </div>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            model_choice = gr.Dropdown(
                choices=["PyTorch", "TensorFlow"],
                value="PyTorch",
                label="Model",
            )
            image_input = gr.Image(type="pil", label="Upload image")
            with gr.Row():
                predict_btn = gr.Button("Predict", variant="primary")
                clear_btn = gr.Button("Clear")

        with gr.Column(scale=1):
            pred_text = gr.Textbox(label="Predicted class")
            conf_text = gr.Textbox(label="Confidence")
            probs = gr.Label(label="Class probabilities", num_top_classes=6)

    predict_btn.click(
        fn=predict,
        inputs=[model_choice, image_input],
        outputs=[pred_text, conf_text, probs],
    )
    clear_btn.click(
        fn=lambda: ("", "", None, None),
        inputs=[],
        outputs=[pred_text, conf_text, probs, image_input],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
