# returning binary image

import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import numpy as np
import math

labrat = mpimg.imread('../IMG/labrat.jpg')





def do_normalise(x):
    #print(((1 + image)/257)-1)
    x=np.copy(labrat)
    filter_flag0=x[:,:,:]==0

    x[filter_flag0]=1

    return -np.log(1/((x)/257) - 1)

def undo_normalise(image):
    return (1 + 1/(np.exp(-image) + 1) * 257).astype("uint8")
def rotation_matrix(theta):
    return np.c_[
        [1,0,0],
        [0,np.cos(theta),-np.sin(theta)],
        [0,np.sin(theta),np.cos(theta)]
    ]
img_norm = do_normalise(labrat)
img_rot = np.einsum("ijk,lk->ijl", img_norm, rotation_matrix(np.pi))
img = undo_normalise(img_rot)
 
imgplot = plt.imshow(img)
plt.show()