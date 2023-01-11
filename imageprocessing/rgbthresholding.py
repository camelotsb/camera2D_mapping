import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import numpy

labrat = mpimg.imread('../IMG/labrat.jpg')

red_sample=numpy.copy(labrat)
blue_sample=numpy.copy(labrat)
green_sample=numpy.copy(labrat)


red_sample[:,:,[1,2]]=0

blue_sample[:,:,[0,1]]=0

green_sample[:,:,[0,2]]=0


fig = plt.figure(figsize=(12,3)) # Create a figure for plotting
plt.subplot(131) # Initialize subplot number 1 in a figure that is 3 columns 1 row
plt.imshow(red_sample) # Plot the red channel
plt.subplot(132) # Initialize subplot number 2 in a figure that is 3 columns 1 row
plt.imshow(green_sample)  # Plot the green channel
plt.subplot(133) # Initialize subplot number 3 in a figure that is 3 columns 1 row
plt.imshow(blue_sample)  # Plot the blue channel
plt.show() 