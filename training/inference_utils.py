import os
import random
from typing import List

import numpy as np
from PIL import Image


CLASS_NAMES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# Identique a torchvision.transforms.Normalize (ImageNet stats)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def set_global_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except Exception:
        pass


def preprocess_pil_for_torch(pil_img: Image.Image, image_size: int = 150):
    from torchvision import transforms

    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return eval_tfms(pil_img.convert("RGB")).unsqueeze(0)


def preprocess_pil_for_tf(pil_img: Image.Image, image_size: int = 150) -> np.ndarray:
    img = pil_img.convert("RGB").resize((image_size, image_size))
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(x, axis=0)


def predict_tf(pil_img: Image.Image, model, class_names: List[str], image_size: int) -> str:
    x = preprocess_pil_for_tf(pil_img, image_size=image_size)
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return class_names[pred_idx]
