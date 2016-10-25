from keras.engine.topology import Layer
from keras import backend as K

import theano.tensor as T
import tensorflow as tf
from keras.engine import Model, Input

tf.python.control_flow_ops = tf # hotfix for Keras issue - remove in future


class LRN2D(Layer):
    def __init__(self, alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
        if n % 2 == 0:
            raise NotImplementedError("Only works with odd n")
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
        super(LRN2D, self).__init__(**kwargs)

    def call(self, x, mask=None):
        X = x
        half_n = self.n // 2
        input_sqr = K.square(X)
        if K._BACKEND == 'theano':
            b, ch, r, c = X.shape
            extra_channels = T.alloc(0., b, ch + 2*half_n, r, c)
            input_sqr = T.set_subtensor(
                extra_channels[:, half_n:half_n+ch, :, :], input_sqr)
        elif K._BACKEND == 'tensorflow':
            b, ch, r, c = K.int_shape(X)
            up_dims = tf.pack([tf.shape(X)[0], half_n, r, c])
            up = tf.fill(up_dims, 0.0)
            middle = input_sqr
            down_dims = tf.pack([tf.shape(X)[0], half_n, r, c])
            down = tf.fill(down_dims, 0.0)
            input_sqr = K.concatenate([up, middle, down], axis=1)
        scale = self.k
        norm_alpha = self.alpha / self.n
        for i in range(self.n):
            scale += norm_alpha * input_sqr[:, i:i+ch, :, :]
        scale = scale ** self.beta
        result = X / scale
        return result

    def get_config(self):
        return {
            "alpha": self.alpha,
            "k": self.k,
            "beta": self.beta,
            "n": self.n}


# make sure Theano and Tensorflow backends return the same values
# run this with KERAS_BACKEND=theano and KERAS_BACKEND=tensorflow
if __name__ == "__main__":
    import numpy as np
    np.random.seed(42)

    layer = LRN2D()
    input_data = np.random.rand(2, 3, 4, 4)
    input_dtype = np.dtype('int64')
    input_shape = input_data.shape
    x = Input(batch_shape=input_shape, dtype=input_dtype)
    y = layer(x)
    model = Model(input=x, output=y)
    model.compile('rmsprop', 'mse')

    actual_output = model.predict(input_data)
    actual_output_shape = actual_output.shape

    expected_output = np.array(
    [[[[ 0.22270249,  0.56528926,  0.43524078,  0.35596153],
       [ 0.0927689,   0.09275486,  0.03453658,  0.51502728],
       [ 0.35742357,  0.42101815,  0.01223961,  0.57670307],
       [ 0.49496925,  0.12625712,  0.10811336,  0.1090527 ]],

      [[ 0.1809032,   0.31201714,  0.25683287,  0.17316446],
       [ 0.36380854,  0.08294351,  0.17370954,  0.21783832],
       [ 0.27117968,  0.46686363,  0.11872671,  0.30576098],
       [ 0.35224888,  0.02761948,  0.36124697,  0.101394  ]],

      [[ 0.03867984,  0.56420189,  0.57416111,  0.48067197],
       [ 0.18112376,  0.05807616,  0.40684569,  0.26171413],
       [ 0.07256406,  0.29443091,  0.02044753,  0.54067695],
       [ 0.15387021,  0.39393663,  0.18534382,  0.30923355]]],


     [[[ 0.32507312,  0.10991452,  0.57651383,  0.46089247],
       [ 0.55862534,  0.53206003,  0.35550949,  0.5481444 ],
       [ 0.05261764,  0.11653129,  0.0268922,   0.19344088],
       [ 0.23110659,  0.16134465,  0.49276501,  0.2121262 ]],

      [[ 0.16704325,  0.32268724,  0.08379338,  0.47698474],
       [ 0.04432775,  0.58679819,  0.45917436,  0.11815591],
       [ 0.00328345,  0.48487291,  0.42029828,  0.43346652],
       [ 0.45859548,  0.04402709,  0.21314272,  0.06889596]],

      [[ 0.5132001,   0.37061325,  0.19675156,  0.03779167],
       [ 0.18490984,  0.19335243,  0.43382159,  0.37909028],
       [ 0.52753669,  0.28077874,  0.07111089,  0.42409423],
       [ 0.45236096,  0.33373648,  0.45841494,  0.29361179]]]])


    assert np.allclose(actual_output, expected_output)
    print(actual_output - expected_output)
    print(actual_output_shape)
