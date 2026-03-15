# Age Detection Notebook Project

This project contains notebook-based workflows for age detection experiments with multiple approaches:

- `Model/CNN/Training.ipynb`: CNN training and fine-tuning with TensorFlow/Keras
- `Model/CNN/Test.ipynb`: CNN webcam and single-image inference
- `Model/ViT/Training.ipynb`: Vision Transformer training with PyTorch
- `Model/ViT/Test.ipynb`: Vision Transformer testing notebook
- `Model/HOG_SVM/Training.ipynb`: HOG + SVM training notebook
- `Model/HOG_SVM/Test.ipynb`: HOG + SVM testing notebook

## What We Use

The current notebooks use these main libraries:

- `tensorflow-cpu==2.10.x` with `tensorflow-directml-plugin` on native Windows: CNN training and inference with GPU support
- `torch` and `torchvision`: ViT training and image transforms
- `opencv-python`: webcam access and face detection
- `pandas`: dataset tables and saved training history
- `matplotlib` and `seaborn`: charts and evaluation plots
- `scikit-learn`: dataset split and evaluation metrics
- `Pillow`: image loading for PyTorch datasets
- `jupyter` and `notebook`: running the notebooks locally

## Install Dependencies

Use Python `3.10`.

- This project is configured to use GPU on native Windows.
- The TensorFlow setup uses `tensorflow-cpu==2.10.x` with `tensorflow-directml-plugin`.
- Python `3.10` is required for this TensorFlow GPU configuration.

1. Create and activate a virtual environment.

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Upgrade `pip`.

```powershell
python -m pip install --upgrade pip
```

3. Install all required packages.

```powershell
pip install -r requirements.txt
```

## Start Jupyter Notebook

Run:

```powershell
jupyter notebook
```

Then open the notebook you want to use.

## Notes

- The CNN notebooks expect the UTKFace dataset structure under the configured base path.
- The ViT notebook is arranged for step-by-step Jupyter execution and saves epoch history during training.
- GPU use is expected for the CNN TensorFlow workflow on native Windows.
- If you already created `.venv` with Python `3.11` or `3.13`, delete it and recreate it with Python `3.10` before installing requirements.
