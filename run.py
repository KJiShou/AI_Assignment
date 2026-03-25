import os
import sys
import cv2
import joblib
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from skimage.feature import hog

# TensorFlow / Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D

# PyTorch
import torch
import torch.nn as nn
from torchvision import transforms, models


# =========================================================
# Utility
# =========================================================
def resource_path(relative_path):
    """
    Works for dev and for PyInstaller exe.
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def extract_model_from_dict(obj, candidate_keys):
    """
    If loaded object is a dict checkpoint, try to extract the real model
    from one of the expected keys.
    """
    if not isinstance(obj, dict):
        return obj

    for key in candidate_keys:
        if key in obj:
            return obj[key]

    return obj


# =========================================================
# Face Detector
# =========================================================
class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )
        return faces


# =========================================================
# CNN Age Model Wrapper
# =========================================================
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None)
        super().__init__(**kwargs)


class CNNAdultDetector:
    """
    CNN binary classifier:
    output probability of Adult
    """
    def __init__(self, model_path, threshold=0.5):
        self.model = load_model(
            model_path,
            custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D},
        )
        self.input_size = (128, 128)
        self.threshold = threshold

    def preprocess(self, face_bgr):
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, self.input_size)
        face_normalized = face_resized.astype("float32") / 255.0
        return np.expand_dims(face_normalized, axis=0)

    def predict(self, face_bgr):
        x = self.preprocess(face_bgr)
        pred_prob = float(self.model.predict(x, verbose=0)[0][0])

        if pred_prob > self.threshold:
            label = "Adult"
            confidence = pred_prob
        else:
            label = "Not Adult"
            confidence = 1 - pred_prob

        return label, confidence, {"prob": round(pred_prob, 4)}


# =========================================================
# HOG + SVM Wrapper
# =========================================================
class HOGSVMAdultDetector:
    """
    HOG + SVM binary classifier
    """
    def __init__(self, model_path):
        loaded = joblib.load(model_path)

        # If the joblib file is a dict/checkpoint, extract the real sklearn model
        self.model = extract_model_from_dict(
            loaded,
            candidate_keys=["model", "classifier", "svm_model", "best_model"]
        )

        if isinstance(self.model, dict):
            raise ValueError(
                f"HOG model file loaded as dict, but no usable classifier key was found. "
                f"Available keys: {list(loaded.keys())}"
            )

        self.input_size = (64, 64)

    def preprocess(self, face_bgr):
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, self.input_size)

        features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )
        return features.reshape(1, -1)

    def predict(self, face_bgr):
        x = self.preprocess(face_bgr)

        pred = self.model.predict(x)[0]
        label = "Adult" if int(pred) == 1 else "Not Adult"

        confidence = 0.0
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x)[0]
            confidence = float(np.max(proba))
        elif hasattr(self.model, "decision_function"):
            score = self.model.decision_function(x)
            if np.ndim(score) > 0:
                score = float(np.ravel(score)[0])
            confidence = float(abs(score))

        return label, confidence, {}


# =========================================================
# ViT Wrapper
# =========================================================
class ViTBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.vit_b_16(weights=None)
        in_features = self.backbone.heads.head.in_features

        self.backbone.heads = nn.Identity()
        self.head = nn.Linear(in_features, 1)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits


class ViTAdultDetector:
    def __init__(self, model_path, device=None, threshold=0.5):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

        self.model = ViTBinaryClassifier()

        checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # remove possible 'module.' prefix from DataParallel
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("module.", "")
            cleaned_state_dict[new_k] = v

        self.model.load_state_dict(cleaned_state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def preprocess(self, face_bgr):
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        x = self.transform(rgb).unsqueeze(0).to(self.device)
        return x

    @torch.no_grad()
    def predict(self, face_bgr):
        x = self.preprocess(face_bgr)
        logit = self.model(x)
        prob = torch.sigmoid(logit).item()

        label = "Adult" if prob >= self.threshold else "Not Adult"
        confidence = prob if prob >= self.threshold else (1 - prob)

        return label, confidence, {"prob": round(prob, 4)}

# =========================================================
# Main App
# =========================================================
class AdultDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Adult Face Detection")
        self.root.geometry("1100x750")
        self.root.configure(bg="#1e1e1e")

        self.face_detector = FaceDetector()
        self.model = None
        self.cap = None
        self.running = False

        # Model paths
        self.model_paths = {
            "CNN": resource_path("Model/CNN/runs/cnn_age_detector/best_age_model_finetuned.h5"),
            "HOG_SVM": resource_path("Model/HOG_SVM/runs/hog_svm_adult_binary/best_hog_svm.joblib"),
            "ViT": resource_path("Model/ViT/runs/vit_adult_binary/best_vit_finetuned.pt"),
        }

        self.create_widgets()

    def create_widgets(self):
        top_frame = tk.Frame(self.root, bg="#1e1e1e")
        top_frame.pack(fill="x", padx=20, pady=15)

        title = tk.Label(
            top_frame,
            text="Adult Face Detection System",
            font=("Segoe UI", 20, "bold"),
            fg="white",
            bg="#1e1e1e"
        )
        title.pack(anchor="w")

        control_frame = tk.Frame(self.root, bg="#2a2a2a")
        control_frame.pack(fill="x", padx=20, pady=10)

        tk.Label(
            control_frame,
            text="Select Model:",
            font=("Segoe UI", 12),
            fg="white",
            bg="#2a2a2a"
        ).pack(side="left", padx=10, pady=10)

        self.model_var = tk.StringVar(value="CNN")
        self.model_combo = ttk.Combobox(
            control_frame,
            textvariable=self.model_var,
            values=["CNN", "HOG_SVM", "ViT"],
            state="readonly",
            width=20
        )
        self.model_combo.pack(side="left", padx=10)

        self.load_btn = tk.Button(
            control_frame,
            text="Load Model",
            command=self.load_selected_model,
            font=("Segoe UI", 11),
            bg="#3c7cff",
            fg="white",
            relief="flat",
            padx=15,
            pady=8
        )
        self.load_btn.pack(side="left", padx=10)

        self.start_btn = tk.Button(
            control_frame,
            text="Start Webcam",
            command=self.start_webcam,
            font=("Segoe UI", 11),
            bg="#22aa66",
            fg="white",
            relief="flat",
            padx=15,
            pady=8
        )
        self.start_btn.pack(side="left", padx=10)

        self.stop_btn = tk.Button(
            control_frame,
            text="Stop Webcam",
            command=self.stop_webcam,
            font=("Segoe UI", 11),
            bg="#cc4444",
            fg="white",
            relief="flat",
            padx=15,
            pady=8
        )
        self.stop_btn.pack(side="left", padx=10)

        self.status_label = tk.Label(
            self.root,
            text="Status: Select a model and load it.",
            font=("Segoe UI", 11),
            fg="#cccccc",
            bg="#1e1e1e"
        )
        self.status_label.pack(anchor="w", padx=20, pady=5)

        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.pack(padx=20, pady=10, fill="both", expand=True)

    def load_selected_model(self):
        model_name = self.model_var.get()
        path = self.model_paths[model_name]

        if not os.path.exists(path):
            messagebox.showerror("Error", f"Model file not found:\n{path}")
            return

        try:
            self.status_label.config(text=f"Status: Loading {model_name}...")

            if model_name == "CNN":
                self.model = CNNAdultDetector(path)
            elif model_name == "HOG_SVM":
                self.model = HOGSVMAdultDetector(path)
            elif model_name == "ViT":
                self.model = ViTAdultDetector(path)

            self.status_label.config(text=f"Status: {model_name} loaded successfully.")
            messagebox.showinfo("Success", f"{model_name} loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{e}")
            self.status_label.config(text="Status: Model loading failed.")

    def start_webcam(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please load a model first.")
            return

        if self.running:
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam.")
            return

        self.running = True
        self.status_label.config(text="Status: Webcam started.")
        self.update_frame()

    def stop_webcam(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        self.video_label.config(image="")
        self.status_label.config(text="Status: Webcam stopped.")

    def update_frame(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_webcam()
            return

        frame = cv2.flip(frame, 1)
        faces = self.face_detector.detect(frame)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]

            try:
                label, conf, extra = self.model.predict(face)

                if label == "Adult":
                    color = (0, 200, 0)
                else:
                    color = (0, 0, 255)

                text = label
                if "prob" in extra:
                    text += f" | Prob: {extra['prob']:.2f}"
                elif conf is not None:
                    text += f" | Conf: {conf:.2f}"

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.rectangle(frame, (x, y-30), (x+w, y), color, -1)
                cv2.putText(
                    frame,
                    text,
                    (x + 5, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

            except Exception as e:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 2)
                cv2.putText(
                    frame,
                    "Prediction Error",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2
                )
                print("Prediction error:", e)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((960, 600))
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def on_closing(self):
        self.stop_webcam()
        self.root.destroy()


# =========================================================
# Run
# =========================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = AdultDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()