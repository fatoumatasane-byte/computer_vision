import argparse
import json
import os
from pathlib import Path

import tensorflow as tf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from inference_utils import CLASS_NAMES, set_global_seed

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
AUG_ROTATION_DEG = 10
AUG_BRIGHTNESS = 0.1
AUG_CONTRAST = 0.1
AUG_ZOOM = 0.1


class TorchCNN(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 18 * 18, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_torch_dataloaders(train_dir: str, test_dir: str, image_size: int, batch_size: int, seed: int):
    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(AUG_ROTATION_DEG),
            transforms.ColorJitter(
                brightness=AUG_BRIGHTNESS,
                contrast=AUG_CONTRAST,
                saturation=AUG_BRIGHTNESS,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    full_train = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_size = int(0.2 * len(full_train))
    train_size = len(full_train) - val_size
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_train, [train_size, val_size], generator=gen)
    val_ds.dataset = datasets.ImageFolder(train_dir, transform=eval_tfms)
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader


def train_pytorch(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = os.path.join(args.data_root, "seg_train", "seg_train")
    test_dir = os.path.join(args.data_root, "seg_test", "seg_test")
    train_loader, val_loader, test_loader = build_torch_dataloaders(
        train_dir, test_dir, args.image_size, args.batch_size, args.seed
    )

    model = TorchCNN(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    best_acc = -1.0
    out_path = Path(args.out_dir) / f"{args.firstname}_model.pth"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            total += y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
        tr_loss, tr_acc = running_loss / total, correct / total

        model.eval()
        v_total, v_correct, v_loss = 0, 0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                v_loss += loss.item() * y.size(0)
                v_total += y.size(0)
                v_correct += (logits.argmax(1) == y).sum().item()
        va_loss, va_acc = v_loss / v_total, v_correct / v_total
        scheduler.step(va_acc)
        print(f"[PT] Epoch {epoch}/{args.epochs} | train {tr_acc:.4f} | val {va_acc:.4f}")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "image_size": args.image_size,
                    "seed": args.seed,
                },
                out_path,
            )

    print(f"Saved best PyTorch model: {out_path}")


def build_tf_datasets(train_dir: str, test_dir: str, image_size: int, batch_size: int, seed: int):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="int",
        validation_split=0.2,
        subset="training",
        seed=seed,
        shuffle=True,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="int",
        validation_split=0.2,
        subset="validation",
        seed=seed,
        shuffle=True,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="int",
        shuffle=False,
    )

    imagenet_mean = tf.constant(IMAGENET_MEAN, dtype=tf.float32)
    imagenet_std = tf.constant(IMAGENET_STD, dtype=tf.float32)
    vr = (0.0, 1.0)
    rot_factor = float(AUG_ROTATION_DEG) / 360.0
    aug = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(rot_factor, fill_mode="reflect", seed=seed),
            tf.keras.layers.RandomBrightness(AUG_BRIGHTNESS, value_range=vr, seed=seed),
            tf.keras.layers.RandomContrast(AUG_CONTRAST, value_range=vr, seed=seed),
            tf.keras.layers.RandomZoom(AUG_ZOOM, AUG_ZOOM, fill_mode="reflect", seed=seed),
        ]
    )

    def prep_train(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        x = aug(x, training=True)
        x = (x - imagenet_mean) / imagenet_std
        return x, y

    def prep_eval(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        x = (x - imagenet_mean) / imagenet_std
        return x, y

    auto = tf.data.AUTOTUNE
    return (
        train_ds.map(prep_train, num_parallel_calls=1).prefetch(auto),
        val_ds.map(prep_eval, num_parallel_calls=1).prefetch(auto),
        test_ds.map(prep_eval, num_parallel_calls=1).prefetch(auto),
    )


def build_tf_model(num_classes: int, image_size: int, lr: float):
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = tf.keras.layers.Conv2D(32, 3, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_tensorflow(args):
    train_dir = os.path.join(args.data_root, "seg_train", "seg_train")
    test_dir = os.path.join(args.data_root, "seg_test", "seg_test")
    train_ds, val_ds, test_ds = build_tf_datasets(train_dir, test_dir, args.image_size, args.batch_size, args.seed)
    model = build_tf_model(num_classes=len(CLASS_NAMES), image_size=args.image_size, lr=args.lr)

    out_path = Path(args.out_dir) / f"{args.firstname}_model.keras"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(out_path), monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6, min_delta=1e-4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=1)
    print("Test:", model.evaluate(test_ds, verbose=0))
    print(f"Saved best TensorFlow model: {out_path}")

    meta_path = Path(args.out_dir) / f"{args.firstname}_meta.json"
    meta_path.write_text(
        json.dumps({"class_names": CLASS_NAMES, "image_size": args.image_size, "seed": args.seed}, indent=2),
        encoding="utf-8",
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--framework", choices=["pytorch", "tensorflow"], required=True)
    p.add_argument("--data_root", required=True, help="Kaggle dataset root that contains seg_train and seg_test.")
    p.add_argument("--firstname", required=True)
    p.add_argument("--out_dir", default="../artifacts")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--image_size", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)
    if args.framework == "pytorch":
        train_pytorch(args)
    else:
        train_tensorflow(args)


if __name__ == "__main__":
    main()
