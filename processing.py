import numpy as np

def rgb2gray(rgb):
    """ Convert an RGB digital image to greyscale. """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def vectorization(img, Cr, Cc, renorm = False):
    """ Vectorize the image as follows. 
        1. Split the original (Mr, Mc) image into S equal-sized patches of 
            shape (Cr, Cc). Then, vectorize each patch and collect all 
            in a (S, Cr*Cc) array, called "vect_patches".
        2. Normalize each (Cr*Cc) vector to the intensity of the corresponding 
            (Cr, Cc) patch. When "renorm" is set to True, save the 
            normalization constants in the array "norm" for final decoding. 
            Otherwise, erase this information by setting "norm" equal to an 
            array of 1s.
        3. Define an array called "states" with shape (S, Cr*Cc), obtained as 
            the elementwise square root of "vect_patches".
    Return the couple (states, norm). """
    
    Mr, Mc = img.shape # Shape of the original image (#rows, #columns)
    
    # The image is split into N patches, each of shape (Cr,Cc)
    patches =  (img.reshape(Mc//Cr, Cr, -1, Cc).swapaxes(1, 2)\
                .reshape(-1, Cr, Cc)) # Shape (S, Cr, Cc)
    
    # Vectorization
    vect_patches = np.reshape(patches,\
                              (patches.shape[0],Cr*Cc)) # Shape (S, Cr*Cc)
    
    # Normalization
    states = np.zeros((patches.shape[0],Cr*Cc)) # Shape (S, Cr*Cc)
    
    norm = np.zeros(patches.shape[0]) # Shape (S, 1)
    for idx in range(patches.shape[0]):
        norm[idx] = vect_patches[idx].sum()
        if norm[idx] == 0:
            raise ValueError('Pixel value is 0') 
        tmp = vect_patches[idx]/norm[idx]
        states[idx] = np.sqrt(tmp)
    if renorm == False:
        norm = np.ones(patches.shape[0])
        
    return (states, norm)

def devectorization(out_freq):
    """ Reconstruct an image using the simulation output. The function 
    operates as follows:
        1. Devectorize each of the S arrays in "out_freq" to 
            a (2**(n2/2), 2**(n2/2)) patch.
        2. Recombine the patches in a single object called "decoded_img" with 
        shape (Mr, Mc). 
    Return an image with number of pixels equal to the length of "out_freq" 
    times the number of patches. """
    
    S = out_freq.shape[0] # Number of patches
    nrow = int(np.sqrt(out_freq.shape[1])) # Number of rows of each patch
    ncol = nrow # Number of columns of each compressed patch
    
    decoded_patches = np.reshape(out_freq,\
                      (out_freq.shape[0], nrow, ncol)) # Shape (S, nrow, ncol)
    
    im_h, im_w = nrow*int(np.sqrt(S)), ncol*int(np.sqrt(S)) # Final shape
    
    decoded_img = np.zeros((im_w, im_h)) # Initialization
    
    idx = 0
    for row in np.arange(im_h - nrow + 1, step=nrow):
        for col in np.arange(im_w - ncol + 1, step=ncol):
            decoded_img[row:row+nrow, col:col+ncol] = decoded_patches[idx]
            idx += 1
            
    return decoded_img