# Semi-Supervised-Learning-Image-Classification
This library contains Semi-Supervised Learning Algorithms for Computer Vision tasks implemented with TensorFlow 2.x and Python 3.x.

## Preface

<p>With this library I pursue two goals. The first is an easy to use high-level API to run Semi-Supervised Learning Algorithms on private or public datasets. The code should of course be easy to read and applicable to as many custom applications as possible. Further it provides and easy to use API, making implementing Semi-Supervised Learning 10-liner. The second is of course the personal learning goal of understanding and implementing state-of-the-art and older semi-supervised learning algorithms. Therefore I will focus on the well-known scientific publications of the last few years that lead to successful results in terms of rankings and the standard performance metrics on the current Benchmarks such as <a href="https://paperswithcode.com/sota/semi-supervised-image-classification-on-cifar-6">Semi-Supervised Image Classification on CIFAR-10, 250 Labels</a> or <a href="https://paperswithcode.com/sota/semi-supervised-image-classification-on-2">Semi-Supervised Image Classification on ImageNet - 10% labeled data</a>.</p>
<p>Since I not aim to publish all my codes at ones, this repository will be constantly changing one that takes the latest research in the SSL field into account.</p>

## Main Library Features

 - High Level API
 - 4 Semi Supervised Learning Algorithms for Image Classification (more to come)
 - Many classic and state-of-the-art CNN Models for training available (including pretraining on ImageNet)

# Table of Contents

 - [Examples](#examples)
 - [Installation and Setup](#installation-and-setup)
 - [Run an example](#run-a-pipeline)
 - [Algorithms and Papers](#algorithms-and-papers)
 - [Models and Optimizers](#models-and-optimizers)
 - [Citing](#citing)
 - [License](#license)
 - [References](#references)
 
## Examples

(To be developed)
 
## Installation and Setup

<p>To get the repository running just check the following requirements.</p>

**Requirements**
1) Python >= 3.6
2) tensorflow >= 2.0.0
3) tensorflow_probability >= 0.10.1
4) numpy >= 1.16.0
5) pyyaml >= 5.0.0
6) tqdm >= 4.0.0

<p>Furthermore just execute the following command to download and install the git repository.</p>

**Clone Repository**

    $ git clone https://github.com/JanMarcelKezmann/Semi-Supervised-Learning-Image-Classification.git
    
## Run An Example

(To be developed)

## Algorithms and Papers

(To be developed)

## Models and Optimizers

### CNN Models

|Type         | Names |
|-------------|-------|
|**VGG**          | ``'vgg16' 'vgg19'``|
|**ResNet**       | ``'resnet50' 'resnet50v2' 'resnet101' 'resnet101v2' 'resnet152' 'resnet152v2'``|
|**Inception**    | ``'inceptionv3' 'inceptionsresnetv2'`` |
|**Xception**     | ``'xception'``|
|**DenseNet**     | ``'densenet121' 'densenet169' 'densenet201'``|
|**MobileNet**    | ``'mobilenet' 'mobilenetv2'`` |
|**NasNet**    | ``'nasnetlarge' 'nasnetmobile'`` |
|**EfficientNet** | ``'efficientnetb0' 'efficientnetb1' 'efficientnetb2' 'efficientnetb3' 'efficientnetb4' 'efficientnetb5' 'efficientnetb6' efficientnetb7'``|

### Optimizers

(To be developed)

## Citing

    @misc{Kezmann:2021,
      Author = {Jan-Marcel Kezmann},
      Title = {Semi-Supervised Learning Image Classification},
      Year = {2021},
      Publisher = {GitHub},
      Journal = {GitHub repository},
      Howpublished = {\url{https://github.com/JanMarcelKezmann/Semi-Supervised-Learning-Image-Classification}}
    }

## License

Project is distributed under <a href="https://github.com/JanMarcelKezmann/Semi-Supervised-Learning-Image-Classification/blob/main/LICENSE">MIT License</a>.

## References

 - Google Research, FixMatch, github.com, <a href="https://github.com/google-research/fixmatch">FixMatch</a>
 - Nathan Tozer, mixmatch-tensorflow2.0, github.com <a href="https://github.com/ntozer/mixmatch-tensorflow2.0">mixmatch-tensorflow2.0</a>
