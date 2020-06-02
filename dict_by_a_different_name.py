# Caelan Booker
# Dictionary based kernel initialization for fluke net

import os
import sys
import numpy as np
import scipy as sc
import sklearn.decomposition as skde
import sklearn.feature_extraction as skfe
import torch
import argparse
from utils import parse_filter_specs
from protoDataset import PrototypicalDataset
from random import shuffle

def prepare_dictionaries (Samples, Filter_specs, Dict_alpha=2, Dict_minibatch_size=5, Dict_epochs=1, Dict_jobs=1, Debug_flag=False) :
    """
    Prepare dictionary filters for the convolution layers of fluke_net.
    
    Parameters:
    Samples ........... A tensor of all the samples used to make the dictionaries. This
                        tensor should be of format:
                        [Num_samples, Channels, Height, Width]
                        The type of these tensors should be 64 bit floats.
    Filter_specs ...... A list stating how many filters must be made and their specifications.
                        see the relevant argument for details.
    Return values:
    Filters_output .... A list of tensors. Each tensor is a set of all the kernels for a
                        layer. The tensors are of the following format:
                        [Num_of_kernels, Channels, Kernel_height, Kernel_width]
    """
    
    Filters_output = []

    for Layer,(C,K,M,_) in enumerate(Filter_specs):
        if Debug_flag :
            print('Layer ' + str(Layer) + ' Samples size: ' + str(Samples.shape))
        
        # Extract patches from all the samples.
        # First unfold returns view of all slices of size 'Kernel_height', unfolding
        # along the height dimension. Second call handles unfolding along the width
        # dimension with slices of size 'Kernel_width'. The end result is a tensor
        # view of the samples cut into the patches needed for training. Both use a
        # stride of 1.
        # This results in a tensor of the following format:
        # [Num_samples, Channels, Num_height_slices, Num_width_slices, K, K]
        Patches = Samples.unfold(2, K, 1).unfold(3, K, 1).cpu()
        if Debug_flag :
            print('Layer ' + str(Layer) + ' Patches view size: ' + str(Patches.shape))

        # Move channels dimension to the front and reshape tensor to following format:
        # [Channel, Num_patches, Patch_data]
        Patches = Patches.permute(1, 0, 2, 3, 4, 5)
        Patches = Patches.reshape(Patches.shape[0], -1, K**2)
        if Debug_flag :
            print('Layer ' + str(Layer) + ' Patches reshaped size: ' + str(Patches.shape))        

        # Fit the dictionary and append the atoms to the list of finished kernels
        # We must loop through each channel of the Samples to compute the parts of
        # the kernels that will act on that channel.
        Kernels_list = []
        for Channel in range(Patches.shape[0]) :
            # NOTE:
            # The sklearn functions take 'array-like' as parameters for fitting.
            # I am just passing in the tensors and it seems to be working fine,
            # I don't think I need to convert these back to numpy ndarrays before use.
            
            # Initialize a dictionary for the given channel of the samples.
            Dict = skde.MiniBatchDictionaryLearning(n_components=C,  # num of dict elements to extract
                                                        alpha=Dict_alpha,  # sparsity controlling param
                                                        n_iter=Dict_epochs,  # num of epochs per partial_fit()
                                                        batch_size=Dict_minibatch_size, 
                                                        transform_algorithm='omp',
                                                        n_jobs=Dict_jobs)  # number of parallel jobs to run
            
            # Fit the dictionary to the current channels patches.
            # Fit takes an array parameter of the following format:
            # [Num_samples, Num_features]
            Dict.fit(Patches[Channel,:,:])

            # Reshape the atoms (dictionary components) into kernels and append
            # them to our output list. The components_ array is of format:
            # [Num_components, Num_features]
            Kernels_list.append(Dict.components_.reshape((C, K, K, 1)))

        # Concatenate the list of individual kernels into a ndarry.
        Kernels = np.concatenate(Kernels_list, axis=3)

        # Convert ndarray of kernels into a tensor. Load using the same datatype 
        # and device as the Samples these kernels will convolve
        Kernels_tensor = torch.tensor(Kernels,dtype=Samples.dtype,device=Samples.device)
        # Must also reorder so that it follows the NCHW format of tensors.
        Kernels_tensor = Kernels_tensor.permute(0, 3, 1, 2)

        if Debug_flag :
            print('Layer ' + str(Layer) + ' Kernels size: ' + str(Kernels_tensor.shape)) 
        
        # Create feature map by convolving over Samples with the filters we made
        # from them.
        Convolve_out = torch.nn.functional.conv2d(Samples, Kernels_tensor)
        
        # Normalize feature map according to activation function (ReLU)
        Convolve_out = torch.nn.functional.relu(Convolve_out)

        # Includes max pooling when specified
        if not M==0:
            Convolve_out = torch.nn.functional.max_pool2d(Convolve_out, M)

        Samples = Convolve_out

        # Append generated filters to return list.
        Filters_output.append(Kernels_tensor)
        
    return Filters_output

def make_dicts(Filter_specs, image_shape, Sample_size, Sample_path, Sample_space_csv):
    sample=[]
    
    # Make the database lose scope in janky way
    if True:
        # Create the sample database
        sample_set = PrototypicalDataset(Sample_path, Sample_space_csv, apply_enhancements=False, n_support=1, n_query=0, image_shape=image_shape)

        sample_ids=list(range(len(sample_set)))
        shuffle(sample_ids)
        sample_ids=sample_ids[:Sample_size]

        # No need for dataloader since MB doesn't exist
        for i in sample_ids:
            sample += sample_set[i][1]

    # Create a THICK tensor
    sample=torch.stack(sample)

    # Create the filters from this image sample
    filters = prepare_dictionaries(sample, Filter_specs, Debug_flag=True)
    return filters

if __name__ == "__main__":
    main()
