import numpy as np
import os
from keras.optimizers import SGD
from model import vgg_16, load_image_vgg_16
from functools import lru_cache


def test_pretrained_weights(weights_path):
    model = load_model(weights_path)
    test_image_paths = load_test_image_paths()
    for image_path in test_image_paths:
        image = load_image_vgg_16(image_path)
        prediction = model.predict(image)
        top5 = np.argsort(prediction)[0][::-1][0:5]
        print("Predictions for {}:".format(os.path.split(image_path)[1]))
        for class_label_index in top5:
            class_label = get_label_by_index(class_label_index)
            print("{} - {}".format(class_label_index, class_label))
    return None


def load_model(weights_path):
    model = vgg_16(weights_path)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    return model


def load_test_image_paths():
    image_paths = []
    for filename in os.listdir(current_directory()):
        if filename.endswith(".jpg"):
            image_paths.append(os.path.abspath(filename))
    return image_paths


@lru_cache(maxsize=1)
def load_class_labels():
    labels_path = os.path.join(current_directory(), "synset_words.txt")
    class_labels = []
    with open(labels_path, "r") as lines:
        for line in lines:
            class_labels.append(line.rstrip("\n").split(" ", 1)[1])
    return class_labels


def get_label_by_index(index):
    return load_class_labels()[index]


def current_directory():
    return os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    test_pretrained_weights('vgg16_weights_tensorflow.h5')
