from keras import backend as K
from keras.utils.np_utils import convert_kernel
import tensorflow as tf
tf.python.control_flow_ops = tf # hotfix for Keras issue - remove in future


def convert_weights_theano2tensorflow(model_builder,
                                      theano_weights_file,
                                      tensorflow_weights_file):
    """
    Theano and Tensorflow implement convolutional layers differently.
    This functions transforms pretrained weights for a Theano-based CNN
    to Tensorflow format.
    check out https://github.com/fchollet/keras/wiki/Converting-convolution-kernels-from-Theano-to-TensorFlow-and-vice-versa
    """
    assert K._BACKEND == 'tensorflow'
    model = model_builder(theano_weights_file)
    ops = []
    for layer in model.layers:
        if layer.__class__.__name__ in ['Convolution1D',
                                        'Convolution2D',
                                        'Convolution3D',
                                        'AtrousConvolution2D']:
            original_w = K.get_value(layer.W)
            converted_w = convert_kernel(original_w)
            ops.append(tf.assign(layer.W, converted_w).op)

    K.get_session().run(ops)
    model.save_weights(tensorflow_weights_file)
