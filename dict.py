# Caelan Booker
# Dictionary based kernel initialization for fluke net

import sys
import numpy as np
import scipy as sc
import sklearn.decomposition as skde
import sklearn.feature_extraction as skfe
import cv2
import torch

def prepare_dictionaries (Samples, Filter_specs, Dict_alpha=2, Dict_epochs=1, Dict_minibatch_size=128, Dict_jobs=1) :
    """
    Prepare dictionary filters for the convolution layers of fluke_net.
    
    Parameters:
    Samples ........... A tensor of all the samples used to make the dictionaries. This
                        tensor should be of format:
                        [Num_samples, Channels, Height, Width]
                        The type of these tensors should be 64 bit floats.
    Filter_specs ...... A list stating how many filters must be made and their specifications.
                        The length of the list should be equal to the number of layers in the
                        model. Each list item should be the following format:
                        [Num_out_channels, Kernel_height, Kernel_width]  
                        Num_out_channels can also be though of as the number of kernels for
                        a given layer.
    Return values:
    Filters_output .... A list of tensors. Each tensor is a set of all the kernels for a
                        layer. The tensors are of the following format:
                        [Num_of_kernels, Channels, Kernel_height, Kernel_width]
    """
    
    Filters_output = []
    
    for Layer in range(len(Filter_specs)) :
        # Extract patches from all the samples.
        # First unfold returns view of all slices of size 'Kernel_height', unfolding
        # along the height dimension. Second call handles unfolding along the width
        # dimension with slices of size 'Kernel_width'. The end result is a tensor
        # view of the samples cut into the patches needed for training. Both use a
        # stride of 1.
        # This results in a tensor of the following format:
        # [Num_samples, Channels, Num_height_slices, Num_width_slices, Kernel_height, Kernel_width]
        Patches = Samples.unfold(2, Filter_specs[Layer][1], 1).unfold(3, Filter_specs[Layer][2], 1)

        # Move channels dimension to the front and reshape tensor to following format:
        # [Channel, Num_patches, Patch_data]
        Patches = Patches.permute(1, 0, 2, 3, 4, 5)
        Patches = Patches.reshape(Patches.shape[0], -1, Filter_specs[Layer][1]*Filter_specs[Layer][2])        

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
            Dict = skde.MiniBatchDictionaryLearning(n_components=Filter_specs[Layer][0],  # num of dict elements to extract
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
            Kernels_list.append(Dict.components_.reshape((Filter_specs[Layer][0], Filter_specs[Layer][1], Filter_specs[Layer][2], 1)))

        # Concatenate the list of individual kernels into a ndarry.
        Kernels = np.concatenate(Kernels_list, axis=3)

        # Convert ndarray of kernels into a tensor.
        # Must also reorder so that it follows the NCHW format of tensors.
        Kernels_tensor = torch.from_numpy(Kernels).permute(0, 3, 1, 2) 
        
        # Create feature map by convolving over Samples with the filters we made
        # from them.
        Convolve_out = torch.nn.functional.conv2d(Samples, Kernels_tensor)

        # Normalize feature map according to activation function (ReLU), and        
        # set to Samples for next iteration.
        Samples = torch.nn.functional.relu(Convolve_out)

        # Append generated filters to return list.
        Filters_output.append(Kernels_tensor)
        
    return Filters_output



def main(argv):
    img1 = cv2.imread('whale1.jpg')  # this reads in bgr format
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)  # convert from bgr to rgb
    img2 = cv2.imread('whale2.jpg')  
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    Samples = img1.reshape((1,)+img1.shape)
    Samples = np.append(Samples, img2.reshape((1,)+img1.shape), 0)
    Samples = Samples.astype('float64') / 255
    
    Samples_tensor = torch.from_numpy(Samples)
    Samples_tensor = Samples_tensor.permute(0, 3, 1, 2)
    
    Filter_specs = [[3,3,3],[3,5,5],[10,5,5]]

    Filters_list = prepare_dictionaries(Samples_tensor, Filter_specs)
    
    for Layer in range(len(Filters_list)) :
        
        Filters_list[Layer] = (Filters_list[Layer].permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

        for Filter in range(Filters_list[Layer].shape[0]) :
            cv2.imwrite('filters/f'+str(Layer)+'-'+str(Filter)+'.png', Filters_list[Layer][Filter])


            
if __name__ == "__main__":
    main(sys.argv)
