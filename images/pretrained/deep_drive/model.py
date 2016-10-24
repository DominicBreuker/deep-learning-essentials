from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from Models.deep_drive.layers.lrn2d import LRN2D
from keras.models import model_from_json
from keras import backend as K
import os
import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf


class DeepDrive(object):

    def __init__(self, weights_path=None, target_size=(227, 227)):
        self.weights_path = weights_path
        self.model_structure_file = self.model_structure_file()
        self.weights_file = self.weights_file()

    def load_feature_extractor(self):
        model = model_from_json(open(self.model_structure_file).read(),
                                custom_objects={"LRN2D": LRN2D})
        model.load_weights(self.weights_file)
        return model

    def current_directory(self):
        return os.path.dirname(os.path.abspath(__file__))

    def weights_file(self):
        assert K._BACKEND in ['theano', 'tensorflow']
        if K._BACKEND == 'tensorflow':
            return os.path.join(self.current_directory(),
                                'deep_drive_weights_tensorflow.h5')
        else:
            return os.path.join(self.current_directory(),
                                'deep_drive_weights_theano.h5')

    def model_structure_file(self):
        return os.path.join(self.current_directory(),
                            'deep_drive_model_structure.json')

    # def _create_feature_layers(self, target_size):
    #     feature_layer_1 = [
    #         Convolution2D(96, 11, 11, subsample=(4, 4), dim_ordering='th', activation='relu', input_shape=(3, target_size[0], target_size[1]), name='conv1'),
    #         MaxPooling2D(pool_size=(3, 3), strides=(2, 2), dim_ordering='th', name='pool1'),
    #         LRN2D(alpha=0.0001, k=1, beta=0.75, n=5, name='norm1')
    #     ]
    #
    #     feature_layer_2 = [
    #         ZeroPadding2D(padding=(2, 2), name='conv2_zeropadding'),
    #         Convolution2D(256, 5, 5, subsample=(1, 1), dim_ordering='th', activation='relu', name='conv2'),
    #         MaxPooling2D(pool_size=(3, 3), strides=(2, 2), dim_ordering='th', name='pool2'),
    #         LRN2D(alpha=0.0001, k=1, beta=0.75, n=5, name='norm2')
    #     ]
    #
    #     feature_layer_3 = [
    #         ZeroPadding2D(padding=(1, 1), name='conv3_zeropadding'),
    #         Convolution2D(384, 3, 3, subsample=(1, 1), dim_ordering='th', activation='relu', name='conv3')
    #     ]
    #
    #     feature_layer_4 = [
    #         ZeroPadding2D(padding=(1, 1), name='conv4_zeropadding'),
    #         Convolution2D(384, 3, 3, subsample=(1, 1), dim_ordering='th', activation='relu', name='conv4')
    #     ]
    #
    #     feature_layer_5 = [
    #         ZeroPadding2D(padding=(1, 1), name='conv5_zeropadding'),
    #         Convolution2D(256, 3, 3, subsample=(1, 1), dim_ordering='th', activation='relu', name='conv5'),
    #         MaxPooling2D(pool_size=(3, 3), strides=(2, 2), dim_ordering='th', name='pool5')
    #     ]
    #
    #     return feature_layer_1 + \
    #            feature_layer_2 + \
    #            feature_layer_3 + \
    #            feature_layer_4 + \
    #            feature_layer_5
    #
    # def _create_classification_layers(self):
    #     classification_layers = [
    #         Flatten(),
    #         Dense(4096, activation='relu', name='fc6_gtanet'),
    #         Dropout(0.5),
    #         Dense(4096, activation='relu', name='fc7_gtanet'),
    #         Dropout(0.5),
    #         Dense(6, activation='relu', name='gtanet_fctop')
    #     ]
    #     return classification_layers
