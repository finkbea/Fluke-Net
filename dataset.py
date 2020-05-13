 
"""
@authors: Logan Pashby
Source = https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
Custom dataset class to feed to pytorch dataloader for train/dev picture files.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import random
import os
from PIL import Image
from skimage import io
import csv
import randomTransformation

class FlukeDataset(Dataset):
    def __init__(self, filepath):
        'Initialization'
        self.inputs = []
        self.targets = []
        self.filepath = filepath
        with open("train_labels.csv", newline='') as labels:
            labels = csv.DictReader(labels)
            for row in labels:
                self.inputs.append(row['Image'])
                self.targets.append(row['Id'])

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs)

    def __getitem__(self, index):
        'Generates one sample of data'
        
        file = os.path.join(self.filepath, self.inputs[index])
        x = applyRandomTransformations(torch.from_numpy(io.read(file)))
        y = self.targets[index]

        return x, y

    def applyRandomTransformations(self, iTensor):
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