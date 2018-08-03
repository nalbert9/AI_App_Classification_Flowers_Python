# AI application for indentifying flower taxanomy

## Transfer Learning for Deep Learning - PyTorch

This project was created as part of a submission for the Artificial Intelligence with Python Nanodegree via Udacity.com
<p align="center">
	<img src="assets/Flowers.png" align="middle" alt="drawing" width="500px">
</p>
<b>Accomplishment:</b> 

- Train Image classifier model, which uses Deep Learning pre-trained neural networks (VGG/DenseNet)
to train a neural network to recognize different species of flowers (dataset of 102 flower categories) ;

- Python app which allows user to input some arguments in order to train and make prediction.

<b>Command line application</b> 
 
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
	- Return top **K** most likely classes: python predict.py input checkpoint ```--top_k 3```
	- Use a mapping of categories to real names: ```python predict.py input checkpoint --category_names cat_to_name.json```
	- Use GPU for inference: ```python predict.py input checkpoint --gpu```


<b>You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at.</b>

<p align="center">
	<img src="assets/inference_example.png" align="middle" alt="drawing" width="250px">
</p>
