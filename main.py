import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm

from processing import rgb2gray
from processing import vectorization
from processing import devectorization
from circuits import circuit_builder
from circuits import reconstruction

img = mpimg.imread('input/input2.jpg') # We load the input image
# img = np.array(list(zip(*img[::-1]))) # Uncomment for rotating the image

if len(img.shape) != 2: # Check if the image is already in greyscale
    img = rgb2gray(img) # Convert the RGB image to greycsale
    
bitdepth = 8 # Bit depth of the input image
black, white = 0, 2**bitdepth - 1

plt.rcParams.update({'font.size': 12})
# plt.rcParams['figure.figsize'] = [4, 4] # Resize the output canvas

Mr, Mc = img.shape
print('The resolution is', Mr, 'x', Mc, 'pixels.',\
      'There are', np.size(img), 'pixels.')

np.savetxt('output/data/input' + str(int(np.log2(Mr*Mc)))\
           + 'q' + '.txt',img,fmt='%u') # Save the input data as a txt file

plt.imshow(img, cmap='gray', vmin = black, \
           vmax = white, interpolation = 'none')

plt.title('%i qubits (input)' %np.log2(Mr*Mc))
plt.axis('off')
filename = 'output/images/input' + str(int(np.log2(Mr*Mc))) + 'q' + '.jpg'
plt.savefig(filename, bbox_inches='tight') 
plt.show()

Cr,Cc = Mr, Mc
states, norm = vectorization(img, Cr, Cc)
n0 = int(np.log2(Cr*Cc)) # Number of qubits for each circuit

print('The shape of the vectorized image is ', states.shape,\
      '. It will be loaded in ', states.shape[0],\
          ' circuit(s) of ' , n0, ' qubits.', sep='')

for n2 in tqdm(range(n0-2, 0, -2)):
    # shots = 2**(n2+bitdepth)*int(np.sqrt(2**bitdepth)) # Standard choice
    shots = 2**(n2+bitdepth) # Reasonable accuracy
    
    states, norm = vectorization(img, Cr, Cc)
    qcs = circuit_builder(states, n0, n2)
    out_freq = reconstruction(qcs, n2, shots, norm)
    final_img = devectorization(out_freq)*shots
    
    black, white = 0, 2**bitdepth - 1 
    final_img = final_img/final_img.max()*white
    
    np.savetxt('output/data/output' + str(n2)\
               + 'q' + '.txt', final_img,fmt='%u') # Save the data in txt
    
    plt.imshow(final_img, cmap='gray', vmin = black, vmax = white,\
               interpolation = 'none')
    
    plt.title('%i qubits' %n2)
    plt.axis('off')
    filename = 'output/images/output' + str(n2) + 'q' + '.jpg' 
    plt.savefig(filename, bbox_inches='tight') # Save the reconstructed image
    plt.show()