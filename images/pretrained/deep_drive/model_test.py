import numpy as np
import os
from keras import backend as K
from images.pretrained.deep_drive.layers.lrn2d import LRN2D
from images.pretrained.deep_drive.loaders.load_pillow import load_image_deep_drive
from keras.models import model_from_json
from images.data_utils.generators import get_image_files
import csv


def load_model():
    model = model_from_json(open(model_structure_file()).read(),
                            custom_objects={"LRN2D": LRN2D})
    model.load_weights(weights_file())
    return model


def current_directory():
    return os.path.dirname(os.path.abspath(__file__))


def weights_file():
    assert K._BACKEND in ['theano', 'tensorflow']
    if K._BACKEND == 'tensorflow':
        return os.path.join(current_directory(),
                            'deep_drive_weights_tensorflow.h5')
    else:
        return os.path.join(current_directory(),
                            'deep_drive_weights_theano.h5')


def model_structure_file():
    return os.path.join(current_directory(), 'deep_drive_model_structure.json')


def example_targets():
    targets = {}
    targets_file = os.path.join(current_directory(), "images", "targets.csv")
    with open(targets_file, 'r') as f:
        r = csv.reader(f, delimiter=';', quotechar='"')
        for i, row in enumerate(r):
            if i == 0:
                continue
            image = row[0]
            values = np.array(row[1:]).astype(np.float)
            targets[image] = values
    return targets


def example_images():
    images = {}
    examples_folder = os.path.join(current_directory(), "images")
    image_files = get_image_files(examples_folder, 'jpg')
    for image_file in image_files:
        filename = os.path.split(image_file)[1]
        images[filename] = load_image_deep_drive(image_file)
    return images


def test_pretrained_weights():
    model = load_model()
    images = example_images()
    targets = example_targets()
    for image_name in sorted(images.keys()):
        prediction = model.predict(np.expand_dims(images[image_name],
                                   axis=0))[0]
        target = targets[image_name]
        loss = ((target - prediction) ** 2).mean()
        print("Target:     {}".format(
            ["{0:0.4f}".format(float(j)) for j in target]))
        print("Prediction: {}".format(
            ["{0:0.4f}".format(float(j)) for j in prediction]))
        print("Loss", loss)
        print("-------------------------------------------")
    return None


if __name__ == "__main__":
    print(current_directory())
    test_pretrained_weights()
