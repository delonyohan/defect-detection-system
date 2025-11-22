
import cv2
import numpy as np

def preprocess_image(img, size=(512,512)):
    """OpenCV-based preprocessing: resize, CLAHE on V channel, normalize to [0,1]."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[:,:,2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v = clahe.apply(v)
    hsv[:,:,2] = v
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    img = img.astype(np.float32) / 255.0
    # HWC -> CHW
    img = np.transpose(img, (2,0,1))
    return img
