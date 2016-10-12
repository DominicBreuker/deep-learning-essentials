from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf


def vgg_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model


def load_image_vgg_16(image_path):
    return load_image_vgg_16_scipy(image_path)


def load_image_vgg_16_cv(image_path):
    from cv2 import imread, resize

    image = resize(imread(image_path), (256, 256)).astype(np.float32)
    return preprocess_image(image)


def load_image_vgg_16_pil(image_path):
    from PIL import Image

    image = Image.open(image_path)
    image = image.convert('RGB')
    image = image.resize((256, 256))
    image = np.asarray(image, dtype='float32')
    image = preprocess_bgr_color_mode(image)
    return preprocess_image(image)


def load_image_vgg_16_scipy(image_path):
    from scipy.misc import imread, imresize, imsave

    image = imread(image_path, mode='RGB')
    image = imresize(image, (256, 256))
    image = image.astype('float32')
    image = preprocess_bgr_color_mode(image)
    return preprocess_image(image)


def preprocess_image(image):
    image[:, :, 0] -= 123.68
    image[:, :, 1] -= 116.779
    image[:, :, 2] -= 103.939
    image = image.transpose((2, 0, 1))
    image = crop_image(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    return image


def preprocess_bgr_color_mode(image):
    image[:, :, [0, 1, 2]] = image[:, :, [2, 1, 0]]
    return image


def crop_image(img, crop_size):
    img_size = img.shape[1:]
    img = img[:, (img_size[0]-crop_size[0])//2:(img_size[0]+crop_size[0])//2,
                 (img_size[1]-crop_size[1])//2:(img_size[1]+crop_size[1])//2]
    return img


if __name__ == "__main__":
    import os
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cat1.jpg")
    cvim = load_image_vgg_16(image_path)
    pilim = load_image_vgg_16_pil(image_path)
    scipyim = load_image_vgg_16_scipy(image_path)
    print(cvim.shape)
    print(pilim.shape)
    print(scipyim.shape)
    print(cvim - scipyim)
    print(cvim - pilim)
