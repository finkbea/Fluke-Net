
"""
@authors: Logan Pashby, Dylan Thompson, Adicus Finkbeiner, Connor Barlow
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

def protoCollate(batch):
    queries = []
    support_ids = []
    support_map = {}
    id_counter = 0

    for b in batch:
        queries += b[0]
        support_ids += [ id_counter ] * len(b[0])
        support_map[id_counter] = b[2]
        id_counter += 1

    return (torch.stack(queries), torch.tensor(support_ids), support_map)

class PrototypicalDataset(torch.utils.data.Dataset):

    def __init__(self, image_dir_path: str, labels_file_path: str, apply_enhancements=True, 
                 n_support=1, n_query=1, image_shape=(100,100)):
        """
        Non-Obvious Parameters

        apply_enhancements
            Whether or not basic image enhancements are applied.

        n_support (default: 1), n_query (default: 1)
            Number of support examples per class and query examples per class.

        image_shape (default: (100,100))
            Fixed-size output size of images.

        """
        self.apply_enhancements = apply_enhancements

        self.image_shape = image_shape
        self.image_dir_path = image_dir_path
        self.class_dict = ClassDictionary(labels_file_path)

        self.n_support = n_support
        self.n_query = n_query

        self.classes = list(self.class_dict.getClasses())

    def __len__(self):
        'The amount of classes in the given episode'
        return len(self.classes)

    def __getitem__(self, index):
        'Generates the support and query set for one class in the episode'
        id = self.classes[index]
        img_paths = self.class_dict.getImages(id)
        shuffle(img_paths)

        support = []
        query = []

        # Add all support examples, erroring out if n_support was too
        #   high for the given dataset
        for _ in range(self.n_support):
            support.append(self.getImageTensor(img_paths.pop()))

        # Add all query examples, erroring out if n_query was too
        #   high for the given dataset
        for _ in range(self.n_query):
            query.append(self.getImageTensor(img_paths.pop()))

        # query/id are turned into input/targed pairs in collate_fn
        # support is added to a dictionary in collate_fn
        # pretty hacky
        return query, id, torch.stack(support)

    def getImageTensor(self, img_path, aggressive=False):
        file = os.path.join(self.image_dir_path, img_path)

        # load the image and ensure that it has 3 channels (vs only 1 for grayscale)
        image = Image.open(file).convert('RGB')

        transform_list = []
        if self.apply_enhancements:
            transform_list = self.getRandomTransformations(image)

        # default transforms
        transform_list.append(transforms.Resize(self.image_shape))
        transform_list.append(transforms.ToTensor())

        transform = transforms.Compose(transform_list)
        out = transform(image)

        if (torch.cuda.is_available()):
            out = out.cuda()

        return out

    def getRandomTransformations(self, image):
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
            h = min(int(image.size[1] * scale), image.size[1]-1)
            w = min(int(image.size[0] * scale), image.size[0]-1)
            crop = (h,w)
            # This madness must end. Even with the above code, it failed after 29 epochs, or 
            #   after ~50,000 images were pushed through this function. It must be the
            #   rotation actually slightly shrinking the image in specific cases, and the fact
            #   that its only ever off by a tiny power of 2  makes me think bit-level fuckery
            #   is at play. Therefore, lets let it add the 4 pixels it needs to not throw 
            #   an error with pad_if_needed=True
            transform_list.append(transforms.RandomCrop(crop,pad_if_needed=True))

        return transform_list


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
