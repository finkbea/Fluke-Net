'''Goal: Develop a random transformation  method that takes in a tensor of image data and alters it in some random combination of the following: scaling, cropping, rotating, horizontal flips, B&W?

If the dataset pipeline is based on what we've done for the labs then those transformations will be applied to numpy arrays instead of tensors'''

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage.transform import resize
from skimage.transform import rotate


#very useful resource https://towardsdatascience.com/image-augmentation-using-python-numpy-opencv-and-skimage-ef027e9898da 
def applyRandomTransformations(numPyArray):
    rList = [1, 1, 1, 2, 2, 3] #number of transformations to be made
    rChoice = random.choice(rList)
    tList = [1, 2, 3, 4, 5, 6] #possible transformation possibilities
	
    for x in range(rChoice):
        tChoice = random.choice(tList) #picks a random transformation
        if (tChoice == 1) #converts to greyscale
            numPyArray = rgb2gray(numPyArray)
        elif (tChoice == 2) #horizantal flip
            numPyArray = np.fliplr(numPyArray)
        elif (tChoice == 3) #vertical flip
            numPyArray = np.flipud(numPyArray)
        elif (tChoice == 4) #random rotation
            numPyArray = rotate(numPyArray, random.randint(-359, 359)
        elif (tChoice == 5) #shifts and wraps
            transform = AffineTransform(translation=(random.randint(-200,200), random.randint(-200,200)))
            numPyArray = warp(numPyArray, transform, mode ="wrap") #mode can be constant, edge, symmetric, reflect, or wrap
        else #resizes the array
            numPyArray = resize(numPyArray, (random.randint(100,400), random.randint(100,400)))
            
            
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
