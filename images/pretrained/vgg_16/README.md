# VGG 16

VGG is a pre-trained CNN created using the ImageNet dataset. Read [this paper](https://arxiv.org/pdf/1409.1556.pdf) for details.

## 1. Download weights

You can download files with pre-trained weights below.
Orginally, these weights stem this [Gist](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3) and they are a direct transformation of the original authors' Caffe model.
Weights are backend-specific.
You can either use Theano or Tensorflow.
Be sure to download the correct file:
- Theano: [vgg16_weights_theano]()
- Tensorflow: [vgg16_weights_tensorflow]()

## 2. Load your image files

You must make sure to apply the same preprocessing that was used during training.
Examples for image loader are in the folders of the networks.
You can use either `opencv`, `pillow` or `scipy` to load images.

## 3. Test model with pre-trained weights_path

To see if you are using the weights correctly, check out `model_test.py`.
It will predict the top5 class labels for each `jpg` file in the directory.
What exactly it predicts will depend on the image library you load.
However, you can see that it works if reasonable predictions show up in the top 5 list.
For each image, look out for:
- cat1.jpg: "287 - lynx, catamount"
- cat2.jpg: "285 - Egyptian cat"
- dog1.jpg: "211 - vizsla, Hungarian pointer"
- dog2.jpg: "235 - German shepherd, German shepherd dog, German police dog, alsatian"
- ipod.jpg: "605 - iPod"

### Sources of images:
- cat1.jpg: [Dwight Sipler](http://www.flickr.com/people/62528187@N00) from Stow, MA, USA, [Gillie hunting (2292639848)](https://commons.wikimedia.org/wiki/File:Gillie_hunting_(2292639848).jpg), [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/legalcode)
- cat2.jpg: The original uploader was [DrL](https://en.wikipedia.org/wiki/User:DrL) at [English Wikipedia](https://en.wikipedia.org/wiki/) [Blackcat-Lilith](https://commons.wikimedia.org/wiki/File:Blackcat-Lilith.jpg), [CC BY-SA 2.5
](https://creativecommons.org/licenses/by-sa/2.5/legalcode)
- dog1.jpg: HiSa Hiller, Schweiz, [Thai-Ridgeback](https://commons.wikimedia.org/wiki/File:Thai-Ridgeback.jpg), [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/legalcode)
- dog2.jpg: [Military dog barking](https://commons.wikimedia.org/wiki/File:Military_dog_barking.JPG), in the [public domain](https://en.wikipedia.org/wiki/public_domain)
- ipod.jpg: [Marcus Quigmire](http://www.flickr.com/people/41896843@N00) from Florida, USA, [Baby Bloo taking a dip (3402460462)](https://commons.wikimedia.org/wiki/File:Baby_Bloo_taking_a_dip_(3402460462).jpg), [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/legalcode)
