import os
import sys
import time
import cv2
import joblib
import numpy as np
import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
from skimage.feature import hog

# TensorFlow / Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D

# PyTorch
import torch
import torch.nn as nn
from torchvision import transforms, models


def resource_path(relative_path):
    """Works for dev and for PyInstaller exe."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def extract_model_from_dict(obj, candidate_keys):
    """Extract the real model from a checkpoint dict when present."""
    if not isinstance(obj, dict):
        return obj

    for key in candidate_keys:
        if key in obj:
            return obj[key]

    return obj


def largest_face(faces):
    if faces is None or len(faces) == 0:
        return None
    return max(faces, key=lambda box: int(box[2]) * int(box[3]))


def crop_face(frame, box):
    x, y, w, h = [int(v) for v in box]
    return frame[y:y + h, x:x + w]


def draw_detection_box(frame, box, label=None, color=(0, 122, 255)):
    annotated = frame.copy()
    x, y, w, h = [int(v) for v in box]
    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
    if label:
        cv2.rectangle(annotated, (x, max(0, y - 34)), (x + w, y), color, -1)
        cv2.putText(
            annotated,
            label,
            (x + 8, max(22, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def cv_to_pil(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )


class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(**kwargs)


class CNNAdultDetector:
    def __init__(self, model_path, threshold=0.5):
        self.model = load_model(
            model_path,
            custom_objects={"DepthwiseConv2D": CustomDepthwiseConv2D},
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
        label = "Adult" if pred_prob >= self.threshold else "Not Adult"
        confidence = pred_prob if label == "Adult" else 1 - pred_prob
        return label, confidence, {"prob": pred_prob, "threshold": self.threshold}


class HOGSVMAdultDetector:
    def __init__(self, model_path):
        loaded = joblib.load(model_path)
        self.model = extract_model_from_dict(
            loaded,
            candidate_keys=["model", "classifier", "svm_model", "best_model"],
        )

        if isinstance(self.model, dict):
            raise ValueError(
                "HOG model file loaded as dict, but no usable classifier key was found. "
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
            block_norm="L2-Hys",
        )
        return features.reshape(1, -1)

    def predict(self, face_bgr):
        x = self.preprocess(face_bgr)
        pred = int(self.model.predict(x)[0])
        label = "Adult" if pred == 1 else "Not Adult"
        margin = 0.0
        if hasattr(self.model, "decision_function"):
            score = self.model.decision_function(x)
            if np.ndim(score) > 0:
                score = float(np.ravel(score)[0])
            margin = float(score)
        return label, abs(margin), {"margin": margin}


class ViTBinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        vit = models.vit_b_16(weights=None)
        in_features = vit.heads.head.in_features
        vit.heads = nn.Identity()
        self.backbone = vit
        self.head = nn.Linear(in_features, 1)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features).squeeze(1)


class ViTAdultDetector:
    def __init__(self, model_path, device=None, threshold=0.457286):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.model = ViTBinaryClassifier()

        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        cleaned_state_dict = {}
        for key, value in state_dict.items():
            cleaned_state_dict[key.replace("module.", "")] = value

        self.model.load_state_dict(cleaned_state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(int(224 * 1.14)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def preprocess(self, face_bgr):
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        return self.transform(rgb).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, face_bgr):
        x = self.preprocess(face_bgr)
        logit = self.model(x)
        prob = torch.sigmoid(logit).item()
        label = "Adult" if prob >= self.threshold else "Not Adult"
        confidence = prob if label == "Adult" else 1 - prob
        return label, confidence, {"prob": prob, "threshold": self.threshold}


class PredictionAggregator:
    def __init__(self, model_name):
        self.model_name = model_name
        self.samples = []

    def reset(self):
        self.samples.clear()

    def add(self, prediction):
        self.samples.append(prediction)

    def count(self):
        return len(self.samples)

    def summarize(self):
        if not self.samples:
            raise ValueError("No capture samples available.")

        if self.model_name in {"CNN", "ViT"}:
            probabilities = []
            thresholds = []
            for sample in self.samples:
                prob = sample.get("adult_probability")
                threshold = sample.get("threshold")
                if prob is None:
                    prob = sample.get("confidence")
                if threshold is not None:
                    thresholds.append(float(threshold))
                if prob is not None:
                    probabilities.append(float(prob))

            if not probabilities:
                raise ValueError("No probability-like values available for aggregation.")

            avg_prob = float(np.mean(probabilities))
            threshold = thresholds[0] if thresholds else 0.5
            label = "Adult" if avg_prob >= threshold else "Not Adult"
            confidence = avg_prob if label == "Adult" else 1 - avg_prob
            return {
                "label": label,
                "score_text": f"Average adult probability: {avg_prob:.2%}",
                "confidence_text": f"Decision confidence: {confidence:.2%}",
                "detail_text": f"Based on {len(self.samples)} captured face samples.",
            }

        adult_votes = sum(1 for sample in self.samples if sample["label"] == "Adult")
        not_adult_votes = len(self.samples) - adult_votes
        margins = []
        for sample in self.samples:
            margin = sample.get("margin")
            if margin is None:
                margin = sample.get("confidence", 0.0)
            margins.append(abs(float(margin)))

        avg_margin = float(np.mean(margins)) if margins else 0.0
        label = "Adult" if adult_votes >= not_adult_votes else "Not Adult"
        return {
            "label": label,
            "score_text": f"Vote split: {adult_votes} adult / {not_adult_votes} not adult",
            "confidence_text": f"Average decision margin: {avg_margin:.3f}",
            "detail_text": f"Majority vote across {len(self.samples)} captured face samples.",
        }


class AdultDetectionApp:
    PREVIEW_SIZE = (760, 500)
    SAMPLE_TARGET = 5
    SAMPLE_INTERVAL_SEC = 0.4

    def __init__(self, root):
        self.root = root
        self.root.title("Adult Face Detection Studio")
        self.root.geometry("1480x860")
        self.root.minsize(1320, 760)

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.face_detector = FaceDetector()
        self.model = None
        self.cap = None
        self.running = False
        self.capture_active = False
        self.last_capture_at = 0.0
        self.current_preview_pil = None
        self.preview_image = None
        self.aggregator = None

        self.model_paths = {
            "CNN": resource_path("Model/CNN/runs/cnn_age_detector/best_age_model_finetuned.h5"),
            "HOG_SVM": resource_path("Model/HOG_SVM/runs/hog_svm_adult_binary/best_hog_svm.joblib"),
            "ViT": resource_path("Model/ViT/runs/vit_adult_binary/best_vit_finetuned.pt"),
        }

        self.palette = {
            "bg": "#EEF3F8",
            "panel": "#F9FBFE",
            "panel_alt": "#F2F6FB",
            "border": "#D6E0EB",
            "text": "#10243E",
            "muted": "#5E728D",
            "accent": "#0B7CFF",
            "accent_soft": "#DCEBFF",
            "success": "#12805C",
            "success_soft": "#DDF6EC",
            "danger": "#BF3653",
            "danger_soft": "#FFE6EC",
            "warning": "#8A5A12",
            "warning_soft": "#FFF4DF",
        }

        self.model_var = tk.StringVar(value="CNN")
        self.status_var = tk.StringVar(value="Choose a model and load it to begin.")
        self.mode_var = tk.StringVar(value="Idle")
        self.progress_var = tk.StringVar(value="Ready")
        self.result_label_var = tk.StringVar(value="Awaiting Analysis")
        self.result_score_var = tk.StringVar(value="No prediction available yet.")
        self.result_confidence_var = tk.StringVar(value="Confidence details will appear here.")
        self.result_detail_var = tk.StringVar(value="Start the webcam or upload an image to analyze a face.")
        self.face_var = tk.StringVar(value="Largest face detector is ready.")
        self.source_var = tk.StringVar(value="Source: None")

        self._build_ui()
        self._set_result_state(
            "Awaiting Analysis",
            "No prediction available yet.",
            "Confidence details will appear here.",
            "Use a loaded model to capture or upload one image.",
            "neutral",
        )
        self._refresh_controls()

    def _build_ui(self):
        self.root.configure(bg=self.palette["bg"])
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(
            self.root,
            width=320,
            fg_color=self.palette["panel"],
            corner_radius=24,
            border_width=1,
            border_color=self.palette["border"],
        )
        self.sidebar.grid(row=0, column=0, sticky="nsw", padx=(22, 12), pady=22)
        self.sidebar.grid_propagate(False)

        self.content = ctk.CTkFrame(self.root, fg_color="transparent")
        self.content.grid(row=0, column=1, sticky="nsew", padx=(0, 22), pady=22)
        self.content.grid_columnconfigure(0, weight=3)
        self.content.grid_columnconfigure(1, weight=2)
        self.content.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_preview_panel()
        self._build_result_panel()

    def _create_sidebar_button(self, parent, text, command, secondary=False):
        if secondary:
            return ctk.CTkButton(
                parent,
                text=text,
                command=command,
                height=42,
                corner_radius=14,
                fg_color="#FFFFFF",
                hover_color=self.palette["panel_alt"],
                border_width=1,
                border_color=self.palette["border"],
                text_color=self.palette["text"],
            )
        return ctk.CTkButton(
            parent,
            text=text,
            command=command,
            height=42,
            corner_radius=14,
            fg_color=self.palette["accent"],
            hover_color="#0A69D8",
            text_color="#FFFFFF",
        )

    def _build_sidebar(self):
        ctk.CTkLabel(
            self.sidebar,
            text="Adult Face\nDetection Studio",
            justify="left",
            font=ctk.CTkFont(size=30, weight="bold"),
            text_color=self.palette["text"],
        ).pack(anchor="w", padx=24, pady=(24, 8))

        ctk.CTkLabel(
            self.sidebar,
            text="Capture from webcam or upload one image to check the current model result.",
            justify="left",
            wraplength=260,
            font=ctk.CTkFont(size=14),
            text_color=self.palette["muted"],
        ).pack(anchor="w", padx=24, pady=(0, 18))

        section = ctk.CTkFrame(
            self.sidebar,
            fg_color=self.palette["panel_alt"],
            corner_radius=20,
            border_width=1,
            border_color=self.palette["border"],
        )
        section.pack(fill="x", padx=18, pady=(0, 14))

        ctk.CTkLabel(
            section,
            text="Model",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=self.palette["text"],
        ).pack(anchor="w", padx=16, pady=(16, 8))

        self.model_combo = ctk.CTkComboBox(
            section,
            variable=self.model_var,
            values=["CNN", "HOG_SVM", "ViT"],
            state="readonly",
            height=40,
            button_color=self.palette["accent"],
            button_hover_color="#0A69D8",
            border_color=self.palette["border"],
            fg_color="#FFFFFF",
            text_color=self.palette["text"],
            dropdown_fg_color="#FFFFFF",
            dropdown_hover_color=self.palette["accent_soft"],
        )
        self.model_combo.pack(fill="x", padx=16, pady=(0, 12))

        self.load_btn = self._create_sidebar_button(section, "Load Selected Model", self.load_selected_model)
        self.start_btn = self._create_sidebar_button(section, "Start Webcam", self.start_webcam)
        self.capture_btn = self._create_sidebar_button(section, "Take Picture", self.capture_series)
        self.stop_btn = self._create_sidebar_button(section, "Stop Webcam", self.stop_webcam, secondary=True)
        self.upload_btn = self._create_sidebar_button(section, "Upload Image", self.upload_image, secondary=True)

        self.load_btn.pack(fill="x", padx=16, pady=(0, 10))
        self.start_btn.pack(fill="x", padx=16, pady=(0, 10))
        self.capture_btn.pack(fill="x", padx=16, pady=(0, 10))
        self.stop_btn.pack(fill="x", padx=16, pady=(0, 10))
        self.upload_btn.pack(fill="x", padx=16, pady=(0, 16))

        status_card = ctk.CTkFrame(
            self.sidebar,
            fg_color="#FFFFFF",
            corner_radius=20,
            border_width=1,
            border_color=self.palette["border"],
        )
        status_card.pack(fill="x", padx=18, pady=(0, 14))

        for label_text, variable in (
            ("Status", self.status_var),
            ("Mode", self.mode_var),
            ("Progress", self.progress_var),
            ("Subject", self.face_var),
            ("Source", self.source_var),
        ):
            ctk.CTkLabel(
                status_card,
                text=label_text,
                anchor="w",
                font=ctk.CTkFont(size=12, weight="bold"),
                text_color=self.palette["muted"],
            ).pack(fill="x", padx=16, pady=(14 if label_text == "Status" else 8, 2))
            ctk.CTkLabel(
                status_card,
                textvariable=variable,
                justify="left",
                anchor="w",
                wraplength=252,
                font=ctk.CTkFont(size=14),
                text_color=self.palette["text"],
            ).pack(fill="x", padx=16, pady=(0, 4 if label_text != "Source" else 16))

    def _build_preview_panel(self):
        preview_container = ctk.CTkFrame(
            self.content,
            fg_color=self.palette["panel"],
            corner_radius=24,
            border_width=1,
            border_color=self.palette["border"],
        )
        preview_container.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        preview_container.grid_rowconfigure(1, weight=1)
        preview_container.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(preview_container, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=22, pady=(20, 12))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="Preview",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=self.palette["text"],
        ).grid(row=0, column=0, sticky="w")

        self.preview_badge = ctk.CTkLabel(
            header,
            text="Camera Idle",
            corner_radius=999,
            fg_color=self.palette["warning_soft"],
            text_color=self.palette["warning"],
            padx=14,
            pady=6,
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.preview_badge.grid(row=0, column=1, sticky="e")

        self.preview_label = ctk.CTkLabel(
            preview_container,
            text="Start the webcam or upload an image.",
            text_color=self.palette["muted"],
            fg_color="#E8EEF5",
            corner_radius=22,
            width=self.PREVIEW_SIZE[0],
            height=self.PREVIEW_SIZE[1],
            font=ctk.CTkFont(size=18),
        )
        self.preview_label.grid(row=1, column=0, sticky="nsew", padx=22, pady=(0, 20))

    def _build_result_panel(self):
        panel = ctk.CTkFrame(
            self.content,
            fg_color=self.palette["panel"],
            corner_radius=24,
            border_width=1,
            border_color=self.palette["border"],
        )
        panel.grid(row=0, column=1, sticky="nsew")

        ctk.CTkLabel(
            panel,
            text="Final Result",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=self.palette["text"],
        ).pack(anchor="w", padx=22, pady=(20, 12))

        self.result_card = ctk.CTkFrame(
            panel,
            fg_color="#FFFFFF",
            corner_radius=22,
            border_width=1,
            border_color=self.palette["border"],
        )
        self.result_card.pack(fill="x", padx=22, pady=(0, 14))

        self.result_pill = ctk.CTkLabel(
            self.result_card,
            text="Awaiting Analysis",
            corner_radius=999,
            fg_color=self.palette["accent_soft"],
            text_color=self.palette["accent"],
            padx=16,
            pady=7,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.result_pill.pack(anchor="w", padx=18, pady=(18, 12))

        ctk.CTkLabel(
            self.result_card,
            textvariable=self.result_score_var,
            justify="left",
            wraplength=300,
            anchor="w",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=self.palette["text"],
        ).pack(fill="x", padx=18, pady=(0, 10))

        ctk.CTkLabel(
            self.result_card,
            textvariable=self.result_confidence_var,
            justify="left",
            wraplength=300,
            anchor="w",
            font=ctk.CTkFont(size=14),
            text_color=self.palette["muted"],
        ).pack(fill="x", padx=18, pady=(0, 8))

        ctk.CTkLabel(
            self.result_card,
            textvariable=self.result_detail_var,
            justify="left",
            wraplength=300,
            anchor="w",
            font=ctk.CTkFont(size=14),
            text_color=self.palette["muted"],
        ).pack(fill="x", padx=18, pady=(0, 18))

        ctk.CTkFrame(
            panel,
            fg_color="transparent",
        ).pack(fill="both", expand=True, padx=22, pady=(0, 22))

    def _refresh_controls(self):
        model_ready = self.model is not None
        running = self.running
        capture_active = self.capture_active
        self.load_btn.configure(state="disabled" if capture_active else "normal")
        self.model_combo.configure(state="disabled" if capture_active else "readonly")
        self.start_btn.configure(state="normal" if model_ready and not running else "disabled")
        self.capture_btn.configure(state="normal" if model_ready and running and not capture_active else "disabled")
        self.stop_btn.configure(state="normal" if running else "disabled")
        self.upload_btn.configure(state="disabled" if capture_active else "normal")

    def _set_status(self, message, mode=None):
        self.status_var.set(message)
        if mode is not None:
            self.mode_var.set(mode)

    def _set_progress(self, count):
        if count <= 0:
            self.progress_var.set("Ready")
        elif count >= self.SAMPLE_TARGET:
            self.progress_var.set("Complete")
        else:
            self.progress_var.set("Capturing...")

    def _set_result_state(self, label, score_text, confidence_text, detail_text, tone):
        self.result_label_var.set(label)
        self.result_score_var.set(score_text)
        self.result_confidence_var.set(confidence_text)
        self.result_detail_var.set(detail_text)

        tone_map = {
            "success": (self.palette["success_soft"], self.palette["success"]),
            "danger": (self.palette["danger_soft"], self.palette["danger"]),
            "warning": (self.palette["warning_soft"], self.palette["warning"]),
            "neutral": (self.palette["accent_soft"], self.palette["accent"]),
        }
        bg, fg = tone_map.get(tone, tone_map["neutral"])
        self.result_pill.configure(text=label, fg_color=bg, text_color=fg)

    def _set_preview_badge(self, text, tone):
        tone_map = {
            "live": (self.palette["accent_soft"], self.palette["accent"]),
            "success": (self.palette["success_soft"], self.palette["success"]),
            "warning": (self.palette["warning_soft"], self.palette["warning"]),
            "danger": (self.palette["danger_soft"], self.palette["danger"]),
        }
        bg, fg = tone_map.get(tone, tone_map["warning"])
        self.preview_badge.configure(text=text, fg_color=bg, text_color=fg)

    def _render_preview(self, frame_bgr):
        preview = cv_to_pil(frame_bgr)
        self.current_preview_pil = preview.copy()
        preview.thumbnail(self.PREVIEW_SIZE, Image.Resampling.LANCZOS)
        self.preview_image = ctk.CTkImage(
            light_image=preview,
            dark_image=preview,
            size=preview.size,
        )
        self.preview_label.configure(text="", image=self.preview_image)

    def _create_prediction_payload(self, face_bgr):
        label, confidence, extra = self.model.predict(face_bgr)
        payload = {
            "label": label,
            "confidence": float(confidence),
            "adult_probability": None,
            "threshold": None,
            "margin": None,
        }
        if "prob" in extra:
            payload["adult_probability"] = float(extra["prob"])
            payload["threshold"] = float(extra.get("threshold", 0.5))
        elif "margin" in extra:
            payload["margin"] = float(extra["margin"])
        elif self.model_var.get() == "HOG_SVM":
            payload["margin"] = float(confidence)
        return payload

    def _single_result_display(self, prediction):
        if prediction["adult_probability"] is not None:
            return f"{prediction['label']} | Prob {prediction['adult_probability']:.2f}"
        return f"{prediction['label']} | Margin {abs(prediction['margin']):.3f}"

    def _color_for_label(self, label):
        return (18, 128, 92) if label == "Adult" else (191, 54, 83)

    def _apply_single_result(self, prediction, detail_text):
        if prediction["adult_probability"] is not None:
            prob = prediction["adult_probability"]
            confidence = prob if prediction["label"] == "Adult" else 1 - prob
            score_text = f"Adult probability: {prob:.2%}"
            confidence_text = f"Decision confidence: {confidence:.2%}"
        else:
            margin = abs(float(prediction["margin"] if prediction["margin"] is not None else prediction["confidence"]))
            score_text = f"HOG_SVM vote: {prediction['label']}"
            confidence_text = f"Decision margin: {margin:.3f}"

        tone = "success" if prediction["label"] == "Adult" else "danger"
        self._set_result_state(prediction["label"], score_text, confidence_text, detail_text, tone)

    def load_selected_model(self):
        model_name = self.model_var.get()
        path = self.model_paths[model_name]
        if not os.path.exists(path):
            self._set_status(f"Model file not found for {model_name}.", "Error")
            self._set_result_state(
                "Model Missing",
                f"Cannot find model file: {path}",
                "Choose another model or restore the missing artifact.",
                "The selected model could not be loaded.",
                "danger",
            )
            return

        self._set_status(f"Loading {model_name}...", "Loading")
        self._refresh_controls()

        try:
            if model_name == "CNN":
                self.model = CNNAdultDetector(path)
            elif model_name == "HOG_SVM":
                self.model = HOGSVMAdultDetector(path)
            else:
                self.model = ViTAdultDetector(path)

            self.aggregator = PredictionAggregator(model_name)
            self._set_status(f"{model_name} loaded successfully.", "Ready")
            self._set_result_state(
                "Model Ready",
                f"{model_name} is loaded and ready for webcam or upload analysis.",
                "Use Take Picture or Upload Image to start.",
                "Largest detected face is used in both webcam and upload modes.",
                "neutral",
            )
        except Exception as exc:
            self.model = None
            self.aggregator = None
            self._set_status("Model loading failed.", "Error")
            self._set_result_state(
                "Load Failed",
                f"Failed to load {model_name}.",
                str(exc),
                "Check the model file and dependency versions before retrying.",
                "danger",
            )
        finally:
            self._refresh_controls()

    def start_webcam(self):
        if self.model is None:
            self._set_status("Load a model before starting the webcam.", "Idle")
            return
        if self.running:
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = None
            self._set_status("Cannot open webcam.", "Error")
            self._set_result_state(
                "Webcam Error",
                "Unable to start the camera.",
                "Check that no other app is using the webcam.",
                "The app stays idle until the webcam can be opened.",
                "danger",
            )
            return

        self.running = True
        self.capture_active = False
        self.last_capture_at = 0.0
        self.source_var.set("Source: Webcam")
        self.face_var.set("Largest visible face will be sampled.")
        self._set_progress(0)
        self._set_status("Webcam started. Position one face in the preview.", "Webcam Live")
        self._set_preview_badge("Webcam Live", "live")
        self._refresh_controls()
        self.update_frame()

    def stop_webcam(self):
        self.capture_active = False
        self.running = False
        self.last_capture_at = 0.0

        if self.cap:
            self.cap.release()
            self.cap = None

        if self.aggregator:
            self.aggregator.reset()

        self._set_progress(0)
        self._set_preview_badge("Camera Idle", "warning")
        self._set_status("Webcam stopped.", "Idle")
        self.face_var.set("Largest face detector is ready.")
        self.source_var.set("Source: None")
        self.preview_label.configure(text="Webcam stopped. Start the webcam or upload an image.", image=None)
        self.preview_image = None
        self._refresh_controls()

    def capture_series(self):
        if self.model is None or not self.running:
            self._set_status("Start the webcam with a loaded model before capturing.", "Idle")
            return
        if self.capture_active:
            return

        self.aggregator = PredictionAggregator(self.model_var.get())
        self.capture_active = True
        self.last_capture_at = 0.0
        self._set_progress(0)
        self._set_status("Capturing image...", "Capturing")
        self._set_preview_badge("Capturing", "live")
        self._set_result_state(
            "Capturing",
            "Capturing from the largest detected face.",
            "Frames without a detected face are skipped automatically.",
            "Hold still briefly for more stable input.",
            "warning",
        )
        self._refresh_controls()

    def upload_image(self):
        if self.capture_active:
            return

        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All files", "*.*")],
        )
        if not path:
            return

        if self.running:
            self.stop_webcam()
        if self.model is None:
            self._set_status("Load a model before uploading an image.", "Idle")
            return

        frame = cv2.imread(path)
        if frame is None:
            self._set_status("Failed to read the selected image file.", "Error")
            self._set_result_state(
                "Upload Failed",
                "The selected file could not be opened as an image.",
                "Choose a supported image format and try again.",
                "The app did not run detection because the file could not be decoded.",
                "danger",
            )
            return

        faces = self.face_detector.detect(frame)
        box = largest_face(faces)
        self.source_var.set(f"Source: {os.path.basename(path)}")

        if box is None:
            self._render_preview(frame)
            self.face_var.set("No face detected in the uploaded image.")
            self._set_status("No face detected in uploaded image.", "Upload")
            self._set_preview_badge("No Face Found", "danger")
            self._set_result_state(
                "No Face Detected",
                "The upload did not contain a detectable face.",
                "Try a clearer frontal face photo with better lighting.",
                "Only the largest detected face is analyzed, and this image had none.",
                "danger",
            )
            self._refresh_controls()
            return

        face = crop_face(frame, box)
        prediction = self._create_prediction_payload(face)
        annotated = draw_detection_box(
            frame,
            box,
            self._single_result_display(prediction),
            self._color_for_label(prediction["label"]),
        )
        self._render_preview(annotated)
        self.face_var.set("Largest detected face selected from upload.")
        self._set_status("Uploaded image analyzed successfully.", "Upload")
        self._set_preview_badge("Upload Analyzed", "success")
        self._apply_single_result(prediction, "Uploaded image analyzed from the largest detected face.")
        self._refresh_controls()

    def update_frame(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_webcam()
            self._set_status("Webcam feed ended unexpectedly.", "Error")
            return

        frame = cv2.flip(frame, 1)
        faces = self.face_detector.detect(frame)
        box = largest_face(faces)
        display_frame = frame.copy()
        now = time.time()

        if box is not None:
            display_frame = draw_detection_box(display_frame, box, "Primary face target", (11, 124, 255))
            self.face_var.set("Largest face locked as the capture target.")
        else:
            self.face_var.set("No face detected. Sample capture is paused.")

        if self.capture_active:
            if box is None:
                self._set_preview_badge("Waiting For Face", "warning")
            elif now - self.last_capture_at >= self.SAMPLE_INTERVAL_SEC:
                face = crop_face(frame, box)
                prediction = self._create_prediction_payload(face)
                self.aggregator.add(prediction)
                self.last_capture_at = now
                count = self.aggregator.count()
                self._set_progress(count)
                self._set_status("Capturing image...", "Capturing")
                display_frame = draw_detection_box(
                    frame,
                    box,
                    self._single_result_display(prediction),
                    self._color_for_label(prediction["label"]),
                )

                if count >= self.SAMPLE_TARGET:
                    self.capture_active = False
                    summary = self.aggregator.summarize()
                    tone = "success" if summary["label"] == "Adult" else "danger"
                    self._set_result_state(
                        summary["label"],
                        summary["score_text"],
                        summary["confidence_text"],
                        summary["detail_text"],
                        tone,
                    )
                    self._set_status("Capture complete. Final result is ready.", "Capture Complete")
                    self._set_preview_badge("Capture Complete", "success")
                    self._refresh_controls()
            else:
                self._set_preview_badge("Capturing", "live")
        elif box is not None:
            self._set_preview_badge("Webcam Live", "live")

        self._render_preview(display_frame)
        self.root.after(30, self.update_frame)

    def on_closing(self):
        self.stop_webcam()
        self.root.destroy()


if __name__ == "__main__":
    root = ctk.CTk()
    app = AdultDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
