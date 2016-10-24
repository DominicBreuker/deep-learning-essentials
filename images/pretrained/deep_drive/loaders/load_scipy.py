import numpy as np
from scipy.misc import imread, imresize, imsave
from .utils import preprocess_image, preprocess_bgr_color_mode


def load_image_deep_drive(image_path, target_size=(227, 227)):
    image = imread(image_path, mode='RGB')
    image = imresize(image, target_size)
    image = image.astype('float32')
    image = preprocess_bgr_color_mode(image)
    return preprocess_image(image)
