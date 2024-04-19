import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm

from processing import rgb2gray
from processing import vectorization
from processing import devectorization
from circuits import circuit_builder
from circuits import reconstruction

img = mpimg.imread('input/input2.jpg') # Load the input image
# img = np.array(list(zip(*img[::-1]))) # Rotate the image

if len(img.shape) != 2:
    img = rgb2gray(img) # Convert an RGB image to greyscale
    
bitdepth = 8 # Depth
black, white = 0, 2**bitdepth - 1

plt.rcParams.update({'font.size': 12})
# plt.rcParams['figure.figsize'] = [4, 4] # Resize the output canvas

Mr, Mc = img.shape
print('The resolution is', Mr, 'x', Mc, 'pixels.',\
      'There are', np.size(img), 'pixels.')

np.savetxt('output/data/input' + str(int(np.log2(Mr*Mc)))\
           + 'q' + '.txt', img, fmt='%u') # Save
np.save('output/data/input' + str(int(np.log2(Mr*Mc)))\
                   + 'q' + '.npy', img) # Save

plt.imshow(img, cmap='gray', vmin = black, \
           vmax = white, interpolation = 'none')
    
plt.title('%i qubits (input)' %np.log2(Mr*Mc))
plt.axis('off')
filename = 'output/images/input' + str(int(np.log2(Mr*Mc))) + 'q' + '.png'

# Save
plt.savefig(filename, bbox_inches='tight') # Upscale
plt.imsave(fname=filename, arr=img, cmap='gray', \
           vmin = black, vmax = white, format='png') # Faithful

plt.show()

Cr,Cc = Mr, Mc
states, norm = vectorization(img, Cr, Cc)
n0 = int(np.log2(Cr*Cc)) # Number of qubits for each circuit

print('The shape of the vectorized image is ', states.shape,\
      '. It will be loaded in ', states.shape[0],\
          ' circuit(s) of ' , n0, ' qubits.', sep='')

for n2 in tqdm(range(n0-2, 0, -2)):
    # Choices of shots = 
    # 2**(n2+2*bitdepth)                           Ideal,
    # 2**(n2+bitdepth)*int(np.sqrt(2**bitdepth))   Standard
    # 2**(n2+bitdepth)                             Reasonable
    # 2**(n2)*16                                   Noisy                                                      
    shots = 2**(n2+bitdepth)
    
    states, norm = vectorization(img, Cr, Cc)
    qcs = circuit_builder(states, n0, n2)
    out_freq = reconstruction(qcs, n2, shots, norm)
    final_img = devectorization(out_freq)*shots
    
    black, white = 0, 2**bitdepth - 1 
    final_img = final_img/final_img.max()*white
    
    np.savetxt('output/data/output' + str(n2)\
               + 'q' + '.txt', final_img,fmt='%u') # Save
    np.save('output/data/output' + str(n2)\
                   + 'q' + '.npy', final_img) # Save
    
    plt.imshow(final_img, cmap='gray', vmin = black, vmax = white,\
               interpolation = 'none')
    
    plt.title('%i qubits' %n2)
    plt.tight_layout()
    plt.axis('off')
    
    # Save
    filename = 'output/images/output' + str(n2) + 'q' + '.png' 
    # plt.savefig(filename, bbox_inches='tight') # Upscale
    plt.imsave(fname=filename, arr=final_img, cmap='gray', \
               vmin = black, vmax = white, format='png') # Faithful
        
    plt.show()