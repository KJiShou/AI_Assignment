# -*- mode: python ; coding: utf-8 -*-

import os
import glob

block_cipher = None

ROOT = os.path.dirname(os.path.abspath(SPEC))
MODEL_DIR = os.path.join(ROOT, "Model")
CV2_DATA_SRC = os.path.join(ROOT, ".venv", "lib", "site-packages", "cv2", "data")

# Collect cv2 xml files (haarcascades)
cv2_xml_files = glob.glob(os.path.join(CV2_DATA_SRC, "*.xml"))

a = Analysis(
    ["run.py"],
    pathex=[ROOT],
    binaries=[],
    datas=[
        (MODEL_DIR, "Model"),
    ] + [(f, "cv2/data") for f in cv2_xml_files],
    hiddenimports=[
        "cv2",
        "cv2.cv2",
        "skimage",
        "skimage.feature",
        "skimage.feature.hog",
        "tensorflow",
        "tensorflow.keras",
        "tensorflow.keras.models",
        "tensorflow.keras.layers",
        "torch",
        "torch.nn",
        "torchvision",
        "torchvision.models",
        "torchvision.transforms",
    ],
    hookspath=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AdultFaceDetectionStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AdultFaceDetectionStudio",
)
