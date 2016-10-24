import numpy as np


def preprocess_image(image):
    image[:, :, 0] -= 104.006
    image[:, :, 1] -= 116.669
    image[:, :, 2] -= 122.679
    image = image.transpose((2, 0, 1))
    return image


def preprocess_bgr_color_mode(image):
    image[:, :, [0, 1, 2]] = image[:, :, [2, 1, 0]]
    return image


def deep_drive_feature_batch(feature_extractor, preprocessed_image_batch):
    return feature_extractor.predict(preprocessed_image_batch)
