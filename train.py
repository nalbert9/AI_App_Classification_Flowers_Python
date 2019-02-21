#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# DATE CREATED: 02/06/2018
# REVISED DATE: 18/06/2018
# PURPOSE: Developping AI application for Image Classifier
# using deep learning model
#
# Some parts of this project are based on the official pytorch tutorial
# https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
#
# Example call: python train.py  --learning_rate 0.01 --hidden_units 512
##

# Imports modules
import numpy as np

import argparse
import time
import os
import copy

import torch
import torchvision
import torch.nn.functional as F

from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms, models


def main():
    in_arg = get_inputs_arg()

    model, epochs, hidden_units, optimizer, train_dataset, arch = train_model(
        in_arg.data_dir, in_arg.hidden_units, in_arg.arch, in_arg.learning_rate, in_arg.epochs, in_arg.save_dir, in_arg.gpu)


    # Save checkpoint
    model.class_to_idx = train_dataset.class_to_idx
    check = save_checkpoint(in_arg.save_dir, model, epochs, hidden_units, optimizer, arch)

    print('Checkpoint saved \n')


def get_inputs_arg():
    """Creates Command line arguments"""

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument('--data_dir', type=str, default='flowers',
                        help='Path images folder')

    parser.add_argument('--save_dir', type=str, default='checkpoint.pth',
                        help='Path save checkpoints')

    parser.add_argument('--arch', type=str, default='vgg16',
                        help='Model architecture: vgg16|densenet121')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')

    parser.add_argument('--hidden_units', type=int,
                        help='Number of hidden units')

    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')

    parser.add_argument('--gpu', action='store_true', default=True,
                        help='Enable GPU')
    return parser.parse_args()


def train_model(data_dir, hidden_units, arch, learning_rate, epochs, save_dir, gpu):
    """ Building and training the classifier """

    data_dir = data_dir

    # Define transforms for the training, validation, and testing sets
    data_transforms = {
        'train': transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(30),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])
                                     ]),
        'test': transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])
                                    ]),

        'valid': transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])
                                     ]),
    }

    # Load datasets with ImageFolder
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), 
        data_transforms[x]) for x in list(data_transforms.keys())}

    # Using image datasets and the trainforms, define the dataloaders
    data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                       shuffle=True, num_workers=4)
        for x in list(data_transforms.keys())
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in list(image_datasets.keys())}

    # Using vgg16 and densenet121 pre-trained networks
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        print('Architecture: ', arch, '\n')
        input_units  = 25088
        hidden_units = 4096
        output_units = 102
        drop_p       = 0.5

        # Define new untrained feed-forward network as a classifier
        for param in model.parameters():
            param.requires_grad = False

        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(input_units, hidden_units)),
                                  ('relu', nn.ReLU()),
                                  ('fc2', nn.Linear(hidden_units, 1000)),
                                  ('relu', nn.ReLU()),
                                  ('dropout', nn.Dropout(drop_p)),
                                  ('fc3', nn.Linear(1000, output_units)),
                                  ('output', nn.LogSoftmax(dim=1))
        ]))

        model.classifier = classifier
        print('Classifier: ', classifier, '\n')

    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        print('Architecture: ', arch, '\n')

        input_size = 1024
        hidden_units = 512
        output_size = 102
        drop_p = 0.5

        for param in model.parameters():
            param.requires_grad = False

        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(input_size, hidden_units)),
                                  ('relu', nn.ReLU()),
                                  ('dropout', nn.Dropout(drop_p)),
                                  ('fc2', nn.Linear(hidden_units, output_size)),
                                  ('output', nn.LogSoftmax(dim=1))
        ]))
        model.classifier = classifier
        print('Classifier: ', classifier, '\n')
    else:
        raise ValueError('Wrong architecture', arch)

    # Training on GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)

    if gpu == True and torch.cuda.is_available():
        print('On GPU \n')
    else:
        print('On CPU \n')

    model.to(device)

    # Create network, define the criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Track loss and accuracy on the validation set to determine the best hyperparameters
    print('Training..')
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs  = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss     = criterion(outputs, labels).item()

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss     += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # Deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc), '\n')

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Testing network
    model.eval()

    accuracy = 0

    for inputs, labels in data_loaders['test']:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs  = model(inputs)
        equality = (labels.data == outputs.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    train_dataset = image_datasets['train']
    print("Test accuracy: {:.3f}".format(accuracy/len(data_loaders['test'])))
    print("Inference complete")

    return model, epochs, hidden_units, optimizer, train_dataset, arch

# Save checkpoint
def save_checkpoint(save_dir, model, epochs, hidden_units, optimizer, arch):

    checkpoint = {
        'arch': arch,
        'epochs': epochs+1,
        'hidden_units': hidden_units,
        'optimizer': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict(),
        'model': model
    }
    torch.save(checkpoint, save_dir)


# Call to main function to run the program
if __name__ == "__main__":
    main()
