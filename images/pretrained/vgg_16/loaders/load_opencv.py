import numpy as np
from cv2 import imread, resize
from .utils import preprocess_image


def load_image_vgg_16(image_path, target_size=(256, 256)):
    image = resize(imread(image_path), target_size).astype(np.float32)
    return preprocess_image(image)
