import numpy as np
import pandas as pd
import os
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from images.pretrained.vgg_16.loaders.load_pillow import load_image_vgg_16
from images.data_utils.buffers import buffered_generator_thread


def sequential_image_generator(image_files, image_loader, passes=1):
    i = 0
    while i < passes:
        for image_file in image_files:
            yield image_loader(image_file), image_file
        i += 1


def random_image_generator(image_files, image_loader):
    while True:
        image_file = np.random.choice(image_files)
        yield image_loader(image_file), image_file


def image_filename_labeler(images_with_image_files, labels):
    for image, image_file in images_with_image_files:
        try:
            label = labels[int(os.path.split(image_file)[1][:-4])]
            yield image, label, image_file
        except:
            pass


def batcher(images_with_labels_and_filenames, batch_size):
    while True:
        image_batch, label_batch, image_file_batch = \
            zip(*[next(images_with_labels_and_filenames)
                for i in range(batch_size)])
        batch = (np.stack(image_batch), np.stack(label_batch),
                 np.stack(image_file_batch))
        yield batch


def load_dataset(directory, batch_size):
    files = get_image_files(directory, "jpg")
    labels = get_labels(directory)
    images_with_image_files = \
        random_image_generator(files, image_loader=load_image_vgg_16)
    images_with_labels_and_filenames = image_filename_labeler(
        images_with_image_files, labels)
    dataset = batcher(images_with_labels_and_filenames, batch_size)
    return dataset


def get_image_files(directory, extension):
    files = []
    print("Looking for images in: {}".format(os.path.abspath(directory)))
    for root, directories, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".{}".format(extension)):
                files.append(os.path.join(root, filename))
    print("Number of images: {}".format(len(files)))
    return files


def get_labels(directory):
    files = []
    print("Looking for steering angle files in: {}"
          .format(os.path.abspath(directory)))
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("steering_angles"):
                files.append(os.path.join(root, filename))
    print("Number of steering angle files found: {}".format(len(files)))
    dictionaries = []
    for steering_angle_file in files:
        csv = pd.read_csv(steering_angle_file, index_col="timestamp")
        dictionaries.append(csv["angle"].to_dict())
    return merge_dicts(*dictionaries)


def merge_dicts(*dict_args):
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


def save_image(image_array, image_file):
    image = array_to_img(image_array)
    image.save(image_file)


def save_batch_to_files(x_batch, y_batch, directory, extension):
    for i in range(len(x_batch)):
        save_image(x_batch[i],
                   os.path.join(directory, "img{}.{}".format(y_batch[i],
                                                             extension)))
