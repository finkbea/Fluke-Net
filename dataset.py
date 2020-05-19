
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
from random import shuffle
from PIL import Image
from torchvision import transforms

class PrototypicalDataset(torch.utils.data.Dataset):

    def __init__(self, image_dir_path: str, labels_file_path: str, n_classes: int, apply_enhancements=True, 
                 psuedo_second_image=False, n_support=1, n_query=1, image_shape=(100,100)):
        """
        Non-Obvious Parameters

        apply_enhancements
            Whether or not basic image enhancements are applied.

        n_classes, n_support (default: 1), n_query (default: 1?)
            Number of classes per episode, support examples per class, and query examples per class.

        image_shape (default: (100,100))
            Fixed-size output size of images.

        pseudo_second_image: (default False)
            Whether or not aggressive image transformations are applied to give the appearance that
                a class has 2 samples.
        """
        self.apply_enhancements = apply_enhancements
        self.psuedo_second_image = psuedo_second_image

        self.image_shape = image_shape
        self.image_dir_path = image_dir_path
        self.class_dict = ClassDictionary(labels_file_path)

        self.n_classes = n_classes
        self.n_support = n_support
        self.n_query = n_query

    # This is janky and shouldn't work like this
    def initEpoch(self):
        self.epoch_order = list(self.class_dict.getClasses())
        shuffle(self.epoch_order)
        self.episode_index = 0

    # But, the alternative is very painful
    def epochFinished(self):
        return self.episode_index >= len(self.epoch_order)

    # See the example script for how this should work
    def nextEpisode(self):
        self.episode = self.epoch_order[self.episode_index:self.episode_index+self.n_classes]
        self.episode_index += self.n_classes

    def __len__(self):
        'The amount of classes in the given episode'
        return len(self.episode)

    def __getitem__(self, index):
        'Generates the support and query set for one class in the episode'
        id = self.episode[index]
        img_paths = self.class_dict.getImages(id)
        shuffle(img_paths)

        support = []
        query   = []

        # Add all support examples, erroring out if n_support was too
        #   high for the given dataset
        for _ in range(self.n_support):
            support.append(self.getImageTensor(img_paths.pop()))

        # If query set exceeded, that's expected
        for _ in range(self.n_query):
            try:
                query.append(self.getImageTensor(img_paths.pop()))
            except:
                # If no samples were had, generate a second pseudo image if that option
                #   was set.
                if self.psuedo_second_image and len(query) == 0:
                    img_paths = shuffle(self.class_dict.getImages(id))
                    query.append(self.getImageTensor(img_paths.pop(), aggressive=True))
                break

        support = torch.stack(support)
        if len(query) > 0:
            query = torch.stack(query)
        else:
            query = None

        return (support,len(self.class_dict.getImages(id))), query

    def getImageTensor(self, img_path, aggressive=False):
        file = os.path.join(self.image_dir_path, img_path)

        # load the image and ensure that it has 3 channels (vs only 1 for grayscale)
        image = Image.open(file).convert('RGB')

        # If applying aggressive transformations, don't stack those on top of 
        #   normal ones as well.
        if self.apply_enhancements and not aggressive:
            image = self.applyRandomTransformation(image)
        elif aggressive:
            image = self.applyAggressiveTransformation(image)

        out = transforms.functional.to_tensor(image)
        if (torch.cuda.is_available()):
            out = out.cuda()
        return out

    def applyRandomTransformation(self, image):
        """
        Applies a random combination of image transformations to a PIL image,
        resizing to image_size at the end
        """
        transform_list = []

        # no need for a loop here - we can get an equivalent range of outputs
        # with only one call to each transformation

        # Approximately the proportion of images grayscale in the dataset
        transform_list.append(transforms.RandomGrayscale(p=0.2))
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        transform_list.append(transforms.RandomRotation((-30, 30), resample=False, expand=True, center=None))

        if (random.random() > 0.5):
            scale = random.uniform(0.5, 1.0)
            crop = (int(image.size[1] * scale), int(image.size[0] * scale))
            transform_list.append(transforms.RandomCrop(crop))

        transform_list.append(transforms.Resize(self.image_shape))
        final_transform = transforms.Compose(transform_list)

        return final_transform(image)

    def applyAggressiveTransformation(self, image):
        """
        Applies aggressive random transformations to hopefully approxmiate
        a different datapoint entirely.
        """
        transform_list = []

        # no need for a loop here - we can get an equivalent range of outputs
        # with only one call to each transformation
        transform_list.append(transforms.RandomGrayscale(p=0.5))
        # always flip
        transform_list.append(transforms.RandomHorizontalFlip(p=1.0))
        # possibly rotating too much is a bad idea?
        transform_list.append(transforms.RandomRotation((-45, 45), resample=False, expand=True, center=None))

        # Randomly crop where aspect ratio is (most likely) not maintained
        scale_h = random.uniform(0.5, 0.8)
        scale_w = random.uniform(0.5, 0.8)
        crop = (int(image.size[1] * scale_h), int(image.size[0] * scale_w))
        transform_list.append(transforms.RandomCrop(crop))

        transform_list.append(transforms.Resize(self.image_shape))
        final_transform = transforms.Compose(transform_list)

        return final_transform(image)

"""
Creates a dictionary holding all the image paths in a given class
"""
class ClassDictionary():
    def __init__(self, labels_file_path):
        self.class_dict = {}

        with open(labels_file_path, newline='') as labels:
            labels = csv.DictReader(labels)
            for row in labels:
                id = row['Id']
                image = row['Image']

                if id in self.class_dict:
                    self.class_dict[id].append(image)
                else:
                    self.class_dict[id] = [image]

    def getClasses(self):
        return self.class_dict.keys()
    
    def getImages(self, id):
        return self.class_dict[id].copy()
