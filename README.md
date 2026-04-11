# Age Detection Studio

A desktop application and training pipeline for binary adult/minor face classification using three approaches: CNN (TensorFlow/Keras), HOG+SVM (scikit-image + scikit-learn), and Vision Transformer (PyTorch).

---

## Project Structure

```
AI_Assignment/
├── run.py                      # Main GUI application (CustomTkinter)
├── run.spec                    # PyInstaller spec for building the .exe
├── hook_cv2.py                 # PyInstaller hook: fixes cv2 data path in bundled exe
├── requirements.txt            # Python dependencies
├── data/Face_Age_Dataset/      # Training images (asian_indian/, non_asian/)
│   └── test/                   # Test images
├── Model/
│   ├── CNN/
│   │   ├── Training.ipynb      # CNN training with TensorFlow/Keras
│   │   ├── Test.ipynb          # CNN inference notebook
│   │   └── runs/cnn_age_detector/
│   │       ├── best_age_model_finetuned.h5   # Trained CNN model
│   │       └── ...
│   ├── HOG_SVM/
│   │   ├── Training.ipynb      # HOG feature extraction + SVM training
│   │   ├── Test.ipynb          # HOG+SVM inference notebook
│   │   └── runs/hog_svm_adult_binary/
│   │       ├── best_hog_svm.joblib           # Trained SVM model
│   │       └── ...
│   └── ViT/
│       ├── Training.ipynb      # Vision Transformer training with PyTorch
│       ├── Test.ipynb          # ViT inference notebook
│       └── runs/vit_adult_binary/
│           ├── best_vit_finetuned.pt         # Trained ViT model
│           └── ...
└── dist/AdultFaceDetectionStudio/
    └── AdultFaceDetectionStudio.exe          # Packaged application
```

---

## Install Requirements

**Python 3.10** is required (TensorFlow 2.10 + DirectML plugin only supports Python < 3.11 on Windows).

```powershell
# 1. Create and activate a virtual environment
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Upgrade pip
python -m pip install --upgrade pip

# 3. Install all dependencies
pip install -r requirements.txt
```

> **Note:** If you already have a `.venv` created with Python 3.11 or 3.13, delete it and recreate it with Python 3.10 before installing.

### What is installed

| Package | Purpose |
|---------|---------|
| `tensorflow-cpu` + `tensorflow-directml-plugin` | CNN training/inference with GPU on Windows |
| `torch` + `torchvision` | ViT training and transforms |
| `opencv-python` | Webcam access, face detection (Haar cascade) |
| `scikit-image` | HOG feature extraction |
| `scikit-learn` | SVM model, dataset splits, metrics |
| `customtkinter` | GUI framework |
| `Pillow` | Image loading |
| `jupyter` / `notebook` | Running training and test notebooks |

---

## Train Models

Open Jupyter Notebook from the project root:

```powershell
jupyter notebook
```

Then open and run the training notebook for the model you want to train.

### CNN (TensorFlow/Keras)
- **Notebook:** `Model/CNN/Training.ipynb`
- **Model output:** `Model/CNN/runs/cnn_age_detector/best_age_model_finetuned.h5`
- **Input:** UTKFace-style images under `data/Face_Age_Dataset/`
- **Process:** Initial training → fine-tuning → saves `.h5` model + training history CSVs

### HOG + SVM (scikit-learn)
- **Notebook:** `Model/HOG_SVM/Training.ipynb`
- **Model output:** `Model/HOG_SVM/runs/hog_svm_adult_binary/best_hog_svm.joblib`
- **Process:** HOG feature extraction → SVM training → saves `.joblib` model + metrics

### Vision Transformer (PyTorch)
- **Notebook:** `Model/ViT/Training.ipynb`
- **Model output:** `Model/ViT/runs/vit_adult_binary/best_vit_finetuned.pt`
- **Process:** ViT-B/16 fine-tuning → saves `.pt` checkpoint + training history

---

## Test Models

Each model has a corresponding test notebook.

```powershell
jupyter notebook
```

### CNN
- **Notebook:** `Model/CNN/Test.ipynb`

### HOG + SVM
- **Notebook:** `Model/HOG_SVM/Test.ipynb`

### Vision Transformer
- **Notebook:** `Model/ViT/Test.ipynb`

Test images are located in `test/test/` with ground truth labels in `test/list/` (`test_age.txt`, `test_name.txt`, `test_dis.txt`).

---

## What run.py Uses

### Models (auto-loaded at runtime)
`run.py` loads pre-trained models from these paths:

| Model | Path |
|-------|------|
| CNN | `Model/CNN/runs/cnn_age_detector/best_age_model_finetuned.h5` |
| HOG+SVM | `Model/HOG_SVM/runs/hog_svm_adult_binary/best_hog_svm.joblib` |
| ViT | `Model/ViT/runs/vit_adult_binary/best_vit_finetuned.pt` |

### Libraries
| Library | Role |
|---------|------|
| `opencv-python` | Face detection via Haar Cascade (`haarcascade_frontalface_default.xml`) |
| `tensorflow.keras` | CNN model loading and inference |
| `torch` + `torchvision` | ViT model loading and inference |
| `scikit-learn` + `scikit-image` | HOG feature extraction + SVM prediction |
| `customtkinter` | GUI (light-themed desktop UI) |
| `numpy`, `joblib` | Array operations, SVM model loading |

### External Resources
- **Haar Cascade XML:** `cv2/data/haarcascade_frontalface_default.xml` (bundled with `opencv-python`)
- **Dataset:** `data/Face_Age_Dataset/` with `asian_indian/` and `non_asian/` subdirectories

### Application Workflow
1. User loads a model (CNN / HOG+SVM / ViT)
2. User starts webcam **or** uploads a single image
3. OpenCV Haar Cascade detects the largest face
4. The selected model predicts **Adult** or **Not Adult**
5. For webcam mode: captures 5 face samples at 0.4s intervals, aggregates predictions
6. Result displayed with label, score, and confidence

---

## Build the Executable

### Requirements
- Install PyInstaller:
  ```powershell
  pip install pyinstaller
  ```

### Build command
From the project root (where `run.spec` is located):

```powershell
pyinstaller run.spec --clean
```

The packaged application will be output to:

```
dist/AdultFaceDetectionStudio/AdultFaceDetectionStudio.exe
```

### What the build does
- Bundles `run.py` as a Windows GUI executable
- Includes all three trained models from `Model/`
- Includes `cv2/data/*.xml` (Haar cascade files)
- Sets `sys._MEIPASS` so `hook_cv2.py` can redirect `cv2.data.haarcascades` to the bundled path
- Output is a standalone folder (`dist/AdultFaceDetectionStudio/`) — no installer required

To run the built application, open:
```
dist\AdultFaceDetectionStudio\AdultFaceDetectionStudio.exe
```
