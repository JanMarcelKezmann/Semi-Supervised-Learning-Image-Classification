# Semi-Supervised-Learning-Image-Classification
This library contains Semi-Supervised Learning Algorithms for Computer Vision tasks implemented with TensorFlow 2.x and Python 3.x.

## Preface

<p>With this library I pursue two goals. The first is an easy to use high-level API to run Semi-Supervised Learning Algorithms on private or public datasets. The code should of course be easy to read and applicable to as many custom applications as possible. Further it provides and easy to use API, making implementing Semi-Supervised Learning 10-liner. The second is of course the personal learning goal of understanding and implementing state-of-the-art and older semi-supervised learning algorithms. Therefore I will focus on the well-known scientific publications of the last few years that lead to successful results in terms of rankings and the standard performance metrics on the current Benchmarks such as <a href="https://paperswithcode.com/sota/semi-supervised-image-classification-on-cifar-6">Semi-Supervised Image Classification on CIFAR-10, 250 Labels</a> or <a href="https://paperswithcode.com/sota/semi-supervised-image-classification-on-2">Semi-Supervised Image Classification on ImageNet - 10% labeled data</a>.</p>
<p>Since I not aim to publish all my codes at ones, this repository will be constantly changing one that takes the latest research in the SSL field into account.</p>

## Main Library Features

 - High Level API
 - 9 Semi Supervised Learning Algorithms for Image Classification (more to come)
 - Many classic and state-of-the-art CNN Models for training available (including pretraining on ImageNet)
 - Code is executable in the console as well as in all kind of Jupyter Notebooks

# Table of Contents

 - [Examples](#examples)
 - [Installation and Setup](#installation-and-setup)
 - [Run an example](#run-an-example)
 - [Algorithms its Papers and Implementations](#algorithms-its-papers-and-implementations)
 - [Models and Optimizers](#models-and-optimizers)
 - [Add Public and Custom Datasets](#add-public-and-custom-datasets)
 - [Citing](#citing)
 - [License](#license)
 - [References](#references)
 
## Examples

(To be developed)
 
## Installation and Setup

<p>To get the repository running just check the following requirements.</p>

**Requirements**
1) Python >= 3.6
2) tensorflow >= 2.4.1
3) tensorflow_probability >= 0.10.1
4) numpy >= 1.19.2
5) pyyaml >= 5.0.0
6) tqdm >= 4.0.0

<p>Future updates will allow older tensorflow 2.x versions to be compatible with this repository</p>

<p>Furthermore just execute the following command to download and install the git repository.</p>

**Clone Repository and Install Requirements**

    $ git clone https://github.com/JanMarcelKezmann/Semi-Supervised-Learning-Image-Classification.git
    
    
cd into the directory where requirements.txt is located.

    $ pip install -r requirements.txt
    
or directly install it:<br>
**Pip Install Repository**

    $ pip install git+https://github.com/JanMarcelKezmann/Semi-Supervised-Learning-Image-Classification.git
    
## Run An Example

First, please check that the requirements are all fullfilled.

Now create a file called *test_sslic.py*

To import the library just use the standard python import statement:

```python
import ssl_image_classifcation as sslic
```

Now you can easly check if the installation worked properly by running:

```python
print(sslic.__version__)
```

Then in order to set up and use the argument parser properly add the following lines:

```python
# Get argument parser from sslic
parser = sslic.get_arg_parser(console_args=True)

# Parse arguments
parser_args = parser.parse_args()

# Save parser_args as python dictionary
parser_args = vars(parser_args)
```

The main() function in ssl_image_classification can now be run easily, which will automatically trigger a training process depeding on the arguments specified in the argument parser.

```python
sslic.main(args=parser_args)
```

Now, in order to run the code in the console just run the following (in the console):

    $ python test_sslic.py
    
You can (and probably should) change some arguments in order to use different ssl algorithms, cnn models, optimizers, datasets, epochs, learning rates and so on. For details about the arguments the parser takes take a look <a href="">here</a>.

    $ python test_sslic.py --algorithm "fixmatch" --models "efficientnetb3" --dataset "cifar10" --epochs 100 --batch-size 128 --config-path "dataset configurations"
    
If the installation was executed properly a trainging process should start.
In order to see what arguments the argument parser finally parsed, the dictionary will automatically be printed including all modified and default arguments.

For further examples, please take a look at the examples section or the examples folder.

## Algorithms its Papers and Implementations

 - **<a href="https://arxiv.org/pdf/1710.09412.pdf">Mixup</a>** &nbsp; <a href="https://github.com/JanMarcelKezmann/Semi-Supervised-Learning-Image-Classification/blob/main/ssl_image_classification/algorithms/mixup.py"><img align="center" width="20px" src="https://cdn.iconscout.com/icon/free/png-512/code-280-460136.png" /></a>
 - **<a href="https://arxiv.org/pdf/1905.02249v2.pdf">Mixmatch</a>** &nbsp; <a href="https://github.com/JanMarcelKezmann/Semi-Supervised-Learning-Image-Classification/blob/main/ssl_image_classification/algorithms/mixmatch.py"><img align="center" width="20px" src="https://cdn.iconscout.com/icon/free/png-512/code-280-460136.png" /></a>
 - **<a href="https://arxiv.org/pdf/1911.09785v2.pdf">ReMixMatch</a>** &nbsp; <a href="https://github.com/JanMarcelKezmann/Semi-Supervised-Learning-Image-Classification/blob/main/ssl_image_classification/algorithms/remixmatch.py"><img align="center" width="20px" src="https://cdn.iconscout.com/icon/free/png-512/code-280-460136.png" /></a>
 - **<a href="https://arxiv.org/pdf/2001.07685.pdf">FixMatch</a>** &nbsp; <a href="https://github.com/JanMarcelKezmann/Semi-Supervised-Learning-Image-Classification/blob/main/ssl_image_classification/algorithms/fixmatch.py"><img align="center" width="20px" src="https://cdn.iconscout.com/icon/free/png-512/code-280-460136.png" /></a>
 - **<a href="https://arxiv.org/pdf/1707.03976v2.pdf">VAT</a>** &nbsp; <a href="https://github.com/JanMarcelKezmann/Semi-Supervised-Learning-Image-Classification/blob/main/ssl_image_classification/algorithms/vat.py"><img align="center" width="20px" src="https://cdn.iconscout.com/icon/free/png-512/code-280-460136.png" /></a>
 - **<a href="https://arxiv.org/pdf/1703.01780v6.pdf">Mean Teacher</a>** &nbsp; <a href="https://github.com/JanMarcelKezmann/Semi-Supervised-Learning-Image-Classification/blob/main/ssl_image_classification/algorithms/meanteacher.py"><img align="center" width="20px" src="https://cdn.iconscout.com/icon/free/png-512/code-280-460136.png" /></a>
 - **<a href="https://arxiv.org/pdf/1606.04586v1.pdf">Pi Model</a>** &nbsp; <a href="https://github.com/JanMarcelKezmann/Semi-Supervised-Learning-Image-Classification/blob/main/ssl_image_classification/algorithms/pimodel.py"><img align="center" width="20px" src="https://cdn.iconscout.com/icon/free/png-512/code-280-460136.png" /></a>
 - **<a href="https://www.researchgate.net/publication/280581078_Pseudo-Label_The_Simple_and_Efficient_Semi-Supervised_Learning_Method_for_Deep_Neural_Networks">Pseudo Label</a>** &nbsp; <a href="https://github.com/JanMarcelKezmann/Semi-Supervised-Learning-Image-Classification/blob/main/ssl_image_classification/algorithms/pseudolabel.py"><img align="center" width="20px" src="https://cdn.iconscout.com/icon/free/png-512/code-280-460136.png" /></a>
 - **<a href="https://arxiv.org/abs/1903.03825.pdf">Interpolation Consistency Training (ICT)</a>** &nbsp; <a href="https://github.com/JanMarcelKezmann/Semi-Supervised-Learning-Image-Classification/blob/main/ssl_image_classification/algorithms/ict.py"><img align="center" width="20px" src="https://cdn.iconscout.com/icon/free/png-512/code-280-460136.png" /></a>

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

| Name        | Adjustable Parameters |
|-------------|-----------------------|
|**Adadelta** | ``'learning_rate', 'rho'`` |
|**Adagrad**  | ``'learning_rate'`` |
|**Adam**     | ``'learning_rate', 'beta_1', 'beta_2'`` |
|**Adamax**   | ``'learning_rate', 'beta_1', 'beta_2'`` |
|**Nadam**    | ``'learning_rate', 'beta_1', 'beta_2'`` |
|**RMSprop**  | ``'learning_rate', 'rho', 'momentum'`` |
|**SGD**      | ``'learning_rate', 'momentum'`` |

The according parser arguments to adjust the parameters of the optimizers are the following (ordered alphabetically):

| Parameter Name   | Parser argument |
|------------------|-----------------|
|**beta_1**        | ``--beta1`` |
|**beta_2**        | ``--beta2`` |
|**learning_rate** | ``--lr`` |
|**momentum**      | ``--momentum`` |
|**rho**           | ``--rho`` |

## Add Public and Custom Datasets

<p>When cloning the current repository only the general and dataset related configurations for the public *Cifar10* are provided. In order to use any public dataset, e.g. from TensorFlow Dataset or any custom dataset with custom data, a few simple steps needs to be taken. To incorporate new datasets and its configurations follow the next couple of steps either for public TFDS dataset or any other custom dataset.</p>

### Public TensorFlow Datasets

<p>Embedding a new TensorFlow Dataset into the semi-supervised learning pipeline requires only a few steps. First, you need to pick a dataset from TensorFlow datasets and some informations of it. Second, you need to create a <TFDS Dataset Name>.yaml which will be later used by the semi-supervised learning pipeline for dataset downloading and information extraction.</p>
<p>Now, follow the following steps, in order to embed a new TensorFlow Dataset into your training pipeline:</p>

 1.) Visit <a href="https://www.tensorflow.org/datasets/catalog/overview">TensorFlow Datasets Overview</a><br>
 2.) Under the Category *Image Classification* pick a dataset by clicking on it<br>
 3.) Now, create a YAML file by naming it <TFDS Dataset Name><at><num_lab_samples>.yaml.<br>
   - For example, you picked the *svhn_cropped* dataset and the total number of labeled data samples you want to have is 10.000, then the file should be named: *svhn_cropped@10000.yaml*<br>
   
 4.) Add information about the dataset in the YAML file:

    dataset: 'svhn_cropped'
    num_lab_samples: <Number of labeled samples>
    val_samples: <Number of validation samples>
    total_train_samples: <Total number of training samples>
    pre_val_iter: <Round((total_train_samples - num_lab_samples - val_samples) / BATCH_SIZE)>
    height: <Height of the images>
    width: <Width of the images>
    channels: <Number of channels>
    
 5.) Store the file, e.g., under *dataset configurations*, this subdirectory which is in the same directory as the main.py file from where your ssl code is run should be put as argument *--config-path* in the ArgumentParser<br>
 6.) Finished!<br>

Under 4.), *total_train_samples* can be taken from the information page of the dataset in TensorFlow, i.e. point 2.).<br>
**Important**, *num_lab_samples* and *val_samples* can be in theory chosen arbitrarily, but I recommend to choose it, such that the number of unlabeled samples, i.e. (total_train_samples - num_lab_samples - val_samples) is a multiple of num_lab_samples itself.

### Custom Datasets

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
