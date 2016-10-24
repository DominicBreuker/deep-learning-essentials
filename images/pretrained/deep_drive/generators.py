from images.data_utils.generators import get_image_files, \
                                  get_labels, \
                                  sequential_image_generator, \
                                  image_filename_labeler, \
                                  batcher
from images.pretrained.deep_drive.model import DeepDrive


def deep_drive_raw_dataset(directory, extension, image_loader, batch_size):
    files = get_image_files(directory, extension)
    labels = get_labels(directory)
    images_with_image_files = \
        sequential_image_generator(files, image_loader=image_loader)
    images_with_labels_and_filenames = image_filename_labeler(
        images_with_image_files, labels)
    dataset_generator = batcher(images_with_labels_and_filenames, batch_size)
    return dataset_generator


def deep_drive_features_dataset(directory, extension, image_loader, batch_size,
                                image_size=(227, 227)):
    deep_drive_model = DeepDrive(image_size)
    feature_extractor = deep_drive_model.load_feature_extractor()

    def feature_generator():
        for x_batch, y_batch, name_batch in deep_drive_raw_dataset(directory,
                extension, image_loader(image_size), batch_size):
            yield feature_extractor.predict(x_batch), y_batch, name_batch
    return feature_generator()
