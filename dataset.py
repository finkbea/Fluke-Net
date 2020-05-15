
"""
@authors: Logan Pashby, Dylan Thompson, Adicus Finkbeiner
Source = https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
Custom dataset class to feed to pytorch dataloader for train/dev picture files.
"""

import torch
import numpy as np
import os
import csv
import random
from PIL import Image
from torchvision import transforms

class FlukeDataset(torch.utils.data.Dataset):
    def __init__(self, input_filepath, input_target_pairs_filepath, class_dict):
        'Initialization'
        self.inputs = []
        self.targets = []

        self.input_filepath = input_filepath
        with open(input_target_pairs_filepath, newline='') as labels:
            labels = csv.DictReader(labels)
            for row in labels:
                # for now, just ignore the catchall "new_whale" class
                if (row['Id'] == 'new_whale'):
                    continue

                if (not class_dict.hasClass(row['Id'])):
                    class_dict.addClass(row['Id'])

                self.inputs.append(row['Image'])

                # set our target to an int instead of the human-readable filename string
                # because torch.nn.CrossEntropyLoss needs an int to identify classes;
                # this is functionally equivalent to a one-hot vector
                self.targets.append(class_dict.getClassId(row['Id']))
        
        self.inputs = self.inputs
        self.targets = self.targets


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs)

    def __getitem__(self, index):
        'Generates one sample of data'
        file = os.path.join(self.input_filepath, self.inputs[index])

        # load the image and ensure that it has 3 channels (vs only 1 for grayscale)
        image = Image.open(file).convert('RGB')
        image = self.applyRandomTransformation(image)

        x = transforms.functional.to_tensor(image)
        if (torch.cuda.is_available()):
            x = x.cuda()

        y = self.targets[index]

        return x, y

    def getUniqueTargets(self):
        'Gets list of target classes in this dataset'
        return np.unique(self.targets)

    def applyRandomTransformation(self, image):
        """
        Applies a random combination of image transformations to a PIL image,
        resizing to 100x100 at the end
        """
        transform_list = []

        # no need for a loop here - we can get an equivalent range of outputs
        # with only one call to each transformation
        transform_list.append(transforms.RandomGrayscale(p=0.5))
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        transform_list.append(transforms.RandomRotation((-30, 30), resample=False, expand=True, center=None))

        # disabling cropping for now because I keep getting randrange errors from inside transforms.RandomCrop;
        # I must be using the wrong dimensions or something
        # transform_list.append(transforms.RandomCrop((random.randint(image.size[1] // 4, image.size[1]), random.randint(image.size[0] // 4, image.size[0]))))

        transform_list.append(transforms.Resize((100, 100)))
        final_transform = transforms.Compose(transform_list)

        return final_transform(image)

    def numClasses(self):
        return len(self.targets)

"""
Dictionary class to store mappings between human-readable class names
and identifiers that we can feed into a NN. An instance of this class
can be shared across multiple DataSets.
"""
class ClassDictionary():
    def __init__(self):
        'Initialization'
        self.id_counter = 0
        self.name_to_id = {}
        self.id_to_name = {}

    def addClass(self, name):
        'Adds a new class to dictionary and assigns it a unique id'
        self.name_to_id[name] = self.id_counter
        self.id_to_name[self.id_counter] = name
        self.id_counter += 1

    def hasClass(self, name):
        'Returns true if this class has already been assigned an id'
        return name in self.name_to_id

    def getClassName(self, class_id):
        'Maps a unique class id to a human-readable class name'
        return self.id_to_name[class_id]

    def getClassId(self, name):
        'Maps a human-readable class name to a unique class id'
        return self.name_to_id[name]