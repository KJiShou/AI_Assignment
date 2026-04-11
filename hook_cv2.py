import os
import sys

# Fix cv2.data.haarcascades path when running as bundled exe
if getattr(sys, 'frozen', False):
    import cv2
    cv2.data.haarcascades = os.path.join(sys._MEIPASS, "cv2", "data") + os.sep
