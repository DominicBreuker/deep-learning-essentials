from images.pretrained.deep_drive.generators import deep_drive_features_dataset
from images.pretrained.deep_drive.loaders.load_pillow import image_loader_deep_drive
import h5py
import os
from datetime import datetime


def current_directory():
    return os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    features_filename = "deep_drive_227x227_features.h5"
    features_directory = os.path.join(
        current_directory(), os.pardir, "Features")
    f = h5py.File(os.path.join(features_directory, features_filename), "w")
    images_done = 0
    for x_batch, y_batch, names in deep_drive_features_dataset(
            './Data/extracted', 'jpg', image_loader_deep_drive, 32, (227, 227)):
        for i in range(len(x_batch)):
            group = f.create_group("img_{}".format(images_done + i))
            group.create_dataset("features", data=x_batch[i])
            group.create_dataset("label", data=y_batch[i])
            metadata = names[i].split('/')[-3:]
            utf_metadata = [field.encode('utf8') for field in metadata]
            group.create_dataset("metadata", data=utf_metadata)
        images_done += len(x_batch)
        if images_done % (32 * 1) == 0:
            print("{} - images processed: {}".format(
                datetime.now(), images_done))
