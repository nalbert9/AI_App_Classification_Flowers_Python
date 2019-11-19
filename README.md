## AI App in Python for Flowers Classification using PyTorch Framework

This repository contains project files for the Artificial Intelligence with Python Nanodegree via [Udacity](https://eu.udacity.com/course/ai-programming-python-nanodegree--nd089). In this project, we first develop code for an image classifier built with PyTorch, then convert it into a command line application. GPU is necessary for training the Deep Learning model.

<p align="center">
	<img src="assets/Flowers.png" align="middle" alt="drawing" width="500px">
</p>

### Accomplishment:
- Train Image Classifier Model, which uses Deep Learning Pre-trained Neural Networks (VGG/DenseNet)
to train a neural network to recognize different species of flowers (dataset of 102 flower categories);

- Python App which allows user to input some arguments in order to train and make prediction.

### Local Environment Instructions

1. Clone the repository, and navigate to the downloaded folder. This may take a minute or two to clone due to the included image data
```sh
$ git clone https://github.com/gfoxx29/AI_App_Classification_Flowers_Python.git
```

2. Create (and activate) a new Anaconda environment, named ai-app with Python 3.7

- Linux or Mac:
```sh
	$ conda create -n ai-app python=3.7
	$ source activate ai-app`
```

- Windows:
```sh
	$ conda create --name ai-app python=3.7
	$ activate ai-app`
```
3. Install PyTorch and torchvision; this should install the latest version of PyTorch
```sh
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

### Examples
**Command Line** 
 
Train a new network on a data set with ```train.py```

- Basic usage: ```python train.py data_directory```
- Prints out training loss, validation loss, and validation accuracy as the network trains
- Options:
	- Set directory to save checkpoints: ```python train.py data_dir --save_dir save_directory```
	- Choose architecture: ```python train.py data_dir --arch "vgg16"```
	- Set hyperparameters: ```python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20```
	- Use GPU for training: ```python train.py data_dir --gpu```
- Predict flower name from an image with ```predict.py``` along with the probability of that name. That is, you'll pass in a single image ```/path/to/image``` and return the flower name and class probability.

- Basic usage: ```python predict.py /path/to/image checkpoint```
- Options:
	- Return top **K** most likely classes: python predict.py --input checkpoint ```--top_k 3```
	- Use a mapping of categories to real names: ```python predict.py --input checkpoint --category_names cat_to_name.json```
	- Use GPU for inference: ```python predict.py --input checkpoint --gpu```


<b>You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at.</b>

<p align="center">
	<img src="assets/inference_example.png" align="middle" alt="drawing" width="250px"> 
</p>

## License

The contents of this repository are covered under the [MIT License](LICENSE).
