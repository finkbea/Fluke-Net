import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

#very useful resource https://towardsdatascience.com/image-augmentation-using-python-numpy-opencv-and-skimage-ef027e9898da 
def applyRandomTransformations(iTensor):
    rList = [1, 1, 1, 2, 2, 3] #number of transformations to be made
    rChoice = random.choice(rList)
    tList = [1, 2, 3, 4, 5, 6] #possible transformation possibilities
    
    transformList = []
    
    for x in range(rChoice):
        tChoice = random.choice(tList) #picks a random transformation
        if (tChoice == 1): #converts to greyscale
            transformList.append(RandomGrayscale(p=random.uniform(0.2, 1)))
        elif (tChoice == 2): #horizantal flip
            transformList.append(RandomHorizontalFlip(p=1))
        elif (tChoice == 3): #vertical flip
            transformList.append(RandomVerticalFlip(p=1))
        elif (tChoice == 4): #random rotation
            transformList.append(RandomRotation((-359, 359), resample=False,expand=True, center=None))
        elif (tChoice == 5): #shifts and wraps
            transformList.append(RandomCrop((randint(iTensor.size/4, iTensor.size), randint(iTensor.size/4, iTensor.size)), scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2))
        else: #resizes the array
            transformList.append(Resize(randint(iTensor.size/2, iTensor.size*2)))
            
        return transforms.Compose(transformList) 

