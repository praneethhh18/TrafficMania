import cv2
import numpy as np

def anonymize_frame(frame: np.ndarray, method: str = 'blur', ksize: int = 31) -> np.ndarray:
    if method == 'blur':
        k = (ksize | 1)  # ensure odd
        return cv2.GaussianBlur(frame, (k, k), 0)
    elif method == 'pixelate':
        h, w = frame.shape[:2]
        temp = cv2.resize(frame, (w//20, h//20), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    return frame
