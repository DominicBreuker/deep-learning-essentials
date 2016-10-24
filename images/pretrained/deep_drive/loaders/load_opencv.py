import numpy as np
from cv2 import imread, resize
from .utils import preprocess_image


def load_image_deep_drive(image_path, target_size=(227, 227)):
    image = resize(imread(image_path), target_size).astype(np.float32)
    return preprocess_image(image)
