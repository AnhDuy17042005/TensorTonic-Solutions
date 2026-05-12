import cv2
import numpy as np

def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    img = np.asarray(image, dtype=np.float32)
    
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).tolist()