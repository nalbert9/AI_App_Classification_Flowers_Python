#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# DATE CREATED: 02/06/2018
# REVISED DATE: 25/10/2019
# PURPOSE: Developping AI application for Image Classifier
# using deep learning model
#
# Example call: python predict.py --data_dir flower --top_k 3
# Default call: python predict.py 
##

# Imports modules
import numpy as np

import torch
import torchvision.transforms as transforms
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models

import argparse
import json

from train import train_model
from PIL import Image


def main():
    in_arg = get_predict_arg()

    # Loading model from checkpoint
    print('Checkpoint loaded')
    model = load_checkpoint(in_arg.checkpoint, in_arg.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device =", device)

    if torch.cuda.is_available():
        model.to(device)

    # Invert class_to_idx dictionary to get mapping from index to class well
    model.class_to_idx = dict((value, key) for key, value in model.class_to_idx.items())

    probs, classes = predict(in_arg.input, model, in_arg.top_k, in_arg.gpu)

    print('probs:', probs, '\n')
    print('labels:', classes, '\n')

    # Match with cat_to_json file labels
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    classes = [cat_to_name[i] for i in classes]

    print('classes:', classes, '\n')


def get_predict_arg():
    """Get command line arguments"""

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument('--input', type=str, default='flowers/test/35/image_06986.jpg',
                        help='Image to predict')

    parser.add_argument('--checkpoint', type=str, default='checkpoint.pth',
                        help='Model checkpoint for prediction')

    parser.add_argument('--top_k', type=int, default=5,
                        help='top k most probable classes')

    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='JSON labels files')

    parser.add_argument('--gpu', action='store_true',
                        help='Enable GPU')

    return parser.parse_args()


def load_checkpoint(filepath, gpu):
    """ Load checkpoint """

    if gpu == False:
        # Cpu device
        checkpoint = torch.load(
            filepath, map_location=lambda storage, loc: storage)
    else:
        # GPU device
        checkpoint = torch.load(filepath)

    model = checkpoint['model'] 
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model


def process_image(image):
    """ Inference for classification """

    # Make tranformation resize, crop, center
    resize_image = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])

    # Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    pil_image = resize_image(pil_image).float()
    np_image = np.array(pil_image)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Swap color axis because
    # Numpy image: H x W x C
    # Torch image: C X H X W
    np_image = (np_image.transpose((1, 2, 0)) - mean) / std
    np_image = np_image.transpose((2, 0, 1))

    return np_image


def predict(image_path, model, top_k, gpu):
    """ Predict the class of an image using a trained deep learning model """

    model.eval()
    # Image preprocessing
    image = process_image(image_path)
    image = Variable(torch.FloatTensor(image), requires_grad=True)
    image = image.unsqueeze(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = image.to(device)

    # Get the Top K largest probabilities and the indices of those probabilities
    output = model(image).topk(top_k)
    probs = []
    classes = []

    if torch.cuda.is_available():
        probs = F.softmax(output[0].data, dim=1).cpu().numpy()[0]
        classes = output[1].data.cpu().numpy()[0]
    else:
        probs = F.softmax(output[0].data, dim=1).numpy()[0]
        classes = output[1].data.numpy()[0]

    # Match with cat_to_json file labels
    classes = [model.class_to_idx[i] for i in classes.data]
    return probs, classes


# Call to main function to run the program
if __name__ == "__main__":
    main()
