# Caelan Booker Dictionary initialization for fluke net

import sys
import numpy as np
import scipy as sc
import sklearn.decomposition as skde
import sklearn.feature_extraction as skfe
import cv2
import torch


def prepare_dictionaries (Samples, Filters_per_layer) :
    """
    Prepare dictionary filters for the convolution layers of fluke_net.
    
    Parameters:
    Samples ........... A list of images represented as numpy arrays. These are expected
                        to have 3 channels (r,g,b).
    Layer_count ....... The number of convolution layers in fluke_net.
    Filters_per_layer . A list of ints stating how many filters must be made. The length
                        of this list must equal Layer_count, and the index of an item
                        in the list corresponds to which layer in fluke_net it is for.
    """

    if Samples.dtype is np.dtype('uint8') :
        print("Not floats!")
        Samples = Samples.astype('float64') / 255
    
    Layer_count = len(Filters_per_layer)
    Filters_output = []
    Num_samples = len(Samples)
    Minibatch_size = 128
    patch_height = 3
    patch_width = patch_height
    
    for Layer in range(Layer_count) :
        # Extract patches from all the samples
        Patches = None
        for Image in Samples :
            tmp_patch_arr = skfe.image.extract_patches_2d(Image, (patch_height, patch_width))
            if Patches is None :
                Patches = tmp_patch_arr
            else :
                np.append(Patches, tmp_patch_arr, 0)
                
        # patches should be an array of shape (number of patches, patch_height, patch_width, patch_channels)
        # If working on single channel images, that last dimension may be omitted.               
        # Reshape, also separating the channels (rgb layers on input images)
        Original_patches_shape = Patches.shape
        #print(Original_patches_shape)
        Patches = Patches.reshape((Original_patches_shape[0], -1, Original_patches_shape[3]))
        #print(Patches.shape)
        
        # Fit the dictionary and append the atoms to the list of finished kernels
        Dict_epochs = 1
        Kernels = None
        for Channel in range(Patches.shape[2]) :
            Dict = skde.MiniBatchDictionaryLearning(n_components=Filters_per_layer[Layer],  # num of dict elements to extract
                                                        alpha=2,  # sparsity controlling param
                                                        n_iter=Dict_epochs,  # num of epochs per partial_fit()
                                                        batch_size=Minibatch_size,
                                                        transform_algorithm='omp',
                                                        n_jobs=-1)  # number of parallel jobs to run
            Dict.fit(Patches[:,:,Channel])
            Atoms = Dict.components_
            Reshaped_atoms = Atoms.reshape((Filters_per_layer[Layer], patch_height, patch_width, 1))
            if Kernels is None :
                Kernels = Reshaped_atoms
            else :
                Kernels = np.append(Kernels, Reshaped_atoms, axis=3)

        print(Kernels.shape)
        # remap ?
        Feature_map = []

        Images_tensor = torch.from_numpy(Samples)
        print(Images_tensor.shape)
        Images_tensor = Images_tensor.permute(0, 3, 1, 2)
        print(Images_tensor.shape)
        Kernels_tensor = torch.from_numpy(Kernels)
        Kernels_tensor = Kernels_tensor.permute(0, 3, 1, 2)
        Convolve_out = torch.nn.functional.conv2d(Images_tensor, Kernels_tensor)
        print(Convolve_out.shape)


        """
        for Image in Samples :
            # Convolve over the current Image with each kernel
            Convoled_image = None
            for Kernel in range(Kernels.shape[0]) :
                # Convolve_out should be a 2 dimensional array
                Image_tensor = torch.from_numpy(Image)
                Kernel_tensor = torch.from_numpy(Kernels)
                Image_tensor = Image_tensor.permute(2, 0, 1)
                Image_tensor.unsqueeze_(0)
                Kernel_tensor = Kernel_tensor.permute(0, 3, 1, 2)
                print(Image_tensor.shape)
                print(Image_tensor.type())
                print(Kernel_tensor.shape)
                print(Kernel_tensor.type())
                Convolve_out = torch.nn.functional.conv2d(Image_tensor, Kernel_tensor)
                print(Convolve_out.shape)
                #Convolved_image = sc.ndimage.filters.convolve(Image, Kernels[0,:,:,:])
                #Feature_map.append(Convolved_image)
        """

        # relu ?

        # set samples to activated remaps

        # Append generated filters to return list 
        Filters_output.append(Kernels)
        
    return Filters_output
                    
def main(argv):
    img = cv2.imread('whale1.jpg')  # this reads in bgr format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert from bgr to rgb
    #print(img[0:3,0:3,:])  # all the pixels are 0-255
    Samples = img.reshape((1,)+img.shape)
    Filters_per_layer = [10]
    fils = prepare_dictionaries(Samples, Filters_per_layer)
    for a in range(len(fils)) :
        fils[a] = (fils[a] * 255).astype(np.uint8)
        for i in range(len(fils[a])) :
            cv2.imwrite('filters/f'+str(a)+'-'+str(i)+'.png', fils[a][i])

if __name__ == "__main__":
    main(sys.argv)
