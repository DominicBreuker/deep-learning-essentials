import numpy as np
from PIL import Image
from .utils import preprocess_image, preprocess_bgr_color_mode


def load_image_deep_drive(image_path, target_size=(227, 227)):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = image.resize(target_size)
    image = np.asarray(image, dtype='float32')
    image = preprocess_bgr_color_mode(image)
    return preprocess_image(image)


def image_loader_deep_drive(target_size):
    def load_image_deep_drive(image_path):
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = image.resize(target_size)
        image = np.asarray(image, dtype='float32')
        image = preprocess_bgr_color_mode(image)
        return preprocess_image(image)
    return load_image_deep_drive
