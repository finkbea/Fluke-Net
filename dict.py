# Caelan Booker Dictionary initialization for fluke net

import sys
import numpy as np
import scipy as sc
import sklearn.decomposition as skde
import sklearn.feature_extraction as skfe
import cv2

def prepare_dictionaries (Samples, Layer_count, Filters_per_layer) :
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
    if not len(Filters_per_layer) == Layer_count :
        # Error, Filters_per_layer does not have size equal to Layer_count
        filler_int = 0

    Filters_output = []
    Num_samples = len(Samples)
    Minibatch_size = 128
    patch_height = 3
    patch_width = patch_height
    
    for i in range(Layer_count) :
        # Extract patches from all the samples
        patches = None
        for samp in Samples :
            tmp_patch_arr = skfe.image.extract_patches_2d(samp, (patch_height, patch_width))
            if patches is None :
                patches = tmp_patch_arr
            else :
                np.append(patches, tmp_patch_arr, 0)
                
        # patches should be an array of shape (number of patches, patch_height, patch_width, patch_channels)
        # If working on single channel images, that last dimension may be omitted.

        print(patches.shape)                
        # Reshape, also separating the rgb layers
        r_patches = patches[:,:,:,0].reshape(patches.shape[0], -1)
        g_patches = patches[:,:,:,1].reshape(patches.shape[0], -1)
        b_patches = patches[:,:,:,2].reshape(patches.shape[0], -1)
        print(r_patches.shape)
        """
        r_patches = r_patches.reshape((r_patches.shape[0], r_patches.shape[1], 1))
        g_patches = g_patches.reshape((g_patches.shape[0], g_patches.shape[1], 1))
        b_patches = b_patches.reshape((b_patches.shape[0], b_patches.shape[1], 1))
        print(r_patches.shape)
        patches = np.append(r_patches, g_patches, 2)
        print(patches.shape)
        """
        # Fit the dictionary
        Dict_epochs = 1
        Red_dict = skde.MiniBatchDictionaryLearning(n_components=Filters_per_layer[i],  # num of dict elements to extract
                                                        alpha=2,  # sparsity controlling param
                                                        n_iter=Dict_epochs,  # num of epochs per partial_fit()
                                                        batch_size=Minibatch_size,
                                                        transform_algorithm='omp',
                                                        n_jobs=-1)  # number of parallel jobs to run
        
        Green_dict = skde.MiniBatchDictionaryLearning(n_components=Filters_per_layer[i],
                                                        alpha=2,  
                                                        n_iter=Dict_epochs, 
                                                        batch_size=Minibatch_size,
                                                        transform_algorithm='omp',
                                                        n_jobs=-1) 

        Blue_dict = skde.MiniBatchDictionaryLearning(n_components=Filters_per_layer[i],
                                                        alpha=2,
                                                        n_iter=Dict_epochs, 
                                                        batch_size=Minibatch_size,
                                                        transform_algorithm='omp',
                                                        n_jobs=-1) 
        Red_dict.fit(r_patches)
        Green_dict.fit(g_patches)
        Blue_dict.fit(b_patches)
        Red_atoms = Red_dict.components_
        Green_atoms = Green_dict.components_
        Blue_atoms = Blue_dict.components_
        #print (V.shape)

        # Reshape dictionary stuff?
        Red_kernels = Red_atoms.reshape((Filters_per_layer[i], patch_height, patch_width, 1))
        Green_kernels = Green_atoms.reshape((Filters_per_layer[i], patch_height, patch_width, 1))        
        Blue_kernels = Blue_atoms.reshape((Filters_per_layer[i], patch_height, patch_width, 1))

        Kernels = np.append(np.append(Red_kernels, Green_kernels, axis=3), Blue_kernels, axis=3)
        
        #print(Kernels.shape)
        
        # remap ?

        # relu ?

        # set samples to activated rempas
        Filters_output.append(Kernels)
        
    return Filters_output
                    
def main(argv):
    img = cv2.imread('whale.jpg')  # this reads in bgr format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert from bgr to rgb
    #print(img[0:3,0:3,:])  # all the pixels are 0-255
    Samples = [img]
    Layer_count = 1
    Filters_per_layer = [10]
    fils = prepare_dictionaries(Samples, Layer_count, Filters_per_layer)
    for a in range(len(fils)) :
        fils[a] = (fils[a] * 255).astype(np.uint8)
        for i in range(len(fils[a])) :
            cv2.imwrite('filters/f'+str(a)+'-'+str(i)+'.png', fils[a][i])

if __name__ == "__main__":
    main(sys.argv)
