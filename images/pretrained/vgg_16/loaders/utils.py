import numpy as np

# orignal VGG_16 used (256, 256) images cropped to (224, 224)
# we apply the same cropping during preprocessing
CROP_FACTOR = 0.875


def preprocess_image(image):
    image[:, :, 0] -= 123.68
    image[:, :, 1] -= 116.779
    image[:, :, 2] -= 103.939
    image = image.transpose((2, 0, 1))
    image_dims = cropped_image_size((image.shape[1], image.shape[2]))
    image = crop_image(image, image_dims)
    return image


def cropped_image_size(image_size):
    return (int(CROP_FACTOR * image_size[0]), int(CROP_FACTOR * image_size[1]))


def preprocess_bgr_color_mode(image):
    image[:, :, [0, 1, 2]] = image[:, :, [2, 1, 0]]
    return image


def crop_image(img, crop_size):
    img_size = img.shape[1:]
    img = img[:, (img_size[0]-crop_size[0])//2:(img_size[0]+crop_size[0])//2,
                 (img_size[1]-crop_size[1])//2:(img_size[1]+crop_size[1])//2]
    return img


def vgg_16_feature_batch(feature_extractor, preprocessed_image_batch):
    return feature_extractor.predict(preprocessed_image_batch)
