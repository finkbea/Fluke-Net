 
"""
@authors: Logan Pashby
Source = https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
Custom dataset class to feed to pytorch dataloader for train/dev picture files.
"""

import torch
from torch.utils import data
import numpy as np
import os
from skimage import io
import csv

class FlukeDataset(data.Dataset):
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

        x = io.read(file)
        y = self.targets[index]

        return x, y