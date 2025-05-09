import cv2
import numpy as np

def resize_with_padding(image: np.ndarray, target_size: int = 640, pad_color: int = 114) -> np.ndarray:
    h, w = image.shape[:2]

    if h == 0 or w == 0:
        raise ValueError("Input image has zero height or width.")

    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded_image = np.full((target_size, target_size, 3), pad_color, dtype=np.uint8)

    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    padded_image[top:top+new_h, left:left+new_w] = resized_image

    return padded_image
