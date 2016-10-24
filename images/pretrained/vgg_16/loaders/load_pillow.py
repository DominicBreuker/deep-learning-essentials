import numpy as np
from PIL import Image
from .utils import preprocess_image, preprocess_bgr_color_mode


def load_image_vgg_16(image_path, target_size=(256, 256)):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = image.resize(target_size)
    image = np.asarray(image, dtype='float32')
    image = preprocess_bgr_color_mode(image)
    return preprocess_image(image)


def image_loader_vgg_16(target_size):
    def load_image_vgg_16(image_path):
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = image.resize(target_size)
        image = np.asarray(image, dtype='float32')
        image = preprocess_bgr_color_mode(image)
        return preprocess_image(image)
    return load_image_vgg_16
