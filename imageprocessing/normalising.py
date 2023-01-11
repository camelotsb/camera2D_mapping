import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import numpy as np

'''def do_normalise(image):
    return -np.log((1/((1 + image)/257)) - 1)'''

def do_normalise(img):
    ar = np.array(img).astype(np.float32)
    for i in range(1000):
        mn = np.min(ar)
        mx = np.max(ar)
        norm = (ar - mn) * (1.0 / (mx - mn))



labrat = mpimg.imread('/home/sb/Documents/Unity _rover_Sim/Linux_Roversim/recorded_data/IMG/robocam_2023_01_05_16_28_32_930.jpg')

img_norm = do_normalise(labrat)

 
imgplot = plt.imshow(img_norm)

plt.show()