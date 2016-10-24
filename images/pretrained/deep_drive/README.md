# VGG 16

Deepdrive is a pre-trained CNN based on Alexnet and created by [Craig Quiter](https://github.com/crizCraig) from [deepdrive.io](http://deepdrive.io/).

## 1. Download weights

The original model is in Caffe and does not work with Keras.
I have transformed the weights to both Theano and Tensorflow formats usable by Keras.
You can download files with pre-trained weights below.
You can find the original weights [here](https://gist.github.com/aiworld/66e69c10c9ec82b299279bc7609544d2).
Weights are backend-specific.
You can either use Theano or Tensorflow.
Be sure to download the correct file:
- Theano: [deep_drive_weights_theano](https://coming-soon.com)
- Tensorflow: [deep_drive_weights_tensorflow](https://coming-soon.com)

## 2. Load your image files

You must make sure to apply the same preprocessing that was used during training.
Examples for image loader are in the folders of the networks.
You can use either `opencv`, `pillow` or `scipy` to load images.

## 3. Test model with pre-trained weights_path

To see if you are using the weights correctly, check out `model_test.py`.
It will predict 'spin', 'direction', 'speed', 'acceleration', 'steering', 'throttle' for a bunch of example `jpg` files in the images directory.
They should not be too bad, although they differ depending on the lib you load images with.

If you see something like this, it should work:

```bash
target:     ['-0.2302', '-1.0000', '0.5727', '0.1617', '-0.0526', '0.4000']
Prediction: ['-0.2075', '-0.9194', '0.4819', '0.0151', '-0.0889', '0.2869']
Loss 0.00847404640701
-------------------------------------------
Target:     ['-0.2128', '-1.0000', '0.6045', '0.0164', '-0.0789', '0.4053']
Prediction: ['-0.2074', '-0.9962', '0.5105', '0.0180', '-0.0869', '0.3319']
Loss 0.00238786971414
-------------------------------------------
Target:     ['-0.2330', '-1.0000', '0.6174', '0.0130', '-0.0789', '0.4090']
Prediction: ['-0.1847', '-1.0264', '0.5443', '0.0057', '-0.0598', '0.1569']
Loss 0.0120549358461
-------------------------------------------
Target:     ['-0.2296', '-1.0000', '0.6356', '0.0181', '-0.0526', '0.4319']
Prediction: ['-0.1968', '-0.9999', '0.5232', '0.0179', '-0.0791', '0.3404']
Loss 0.0037943337835
-------------------------------------------
Target:     ['-0.2080', '-1.0000', '0.6468', '0.0112', '-0.0526', '0.4400']
Prediction: ['-0.1828', '-1.0087', '0.5318', '0.0155', '-0.0670', '0.2959']
Loss 0.00582174072584
-------------------------------------------
Target:     ['-0.1643', '-1.0000', '0.6647', '0.0179', '-0.0526', '0.4399']
Prediction: ['-0.1557', '-0.9968', '0.5216', '0.0102', '-0.0531', '0.2054']
Loss 0.0125973180843
```
