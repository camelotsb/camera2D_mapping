import matplotlib.image as mpimg
import matplotlib.pyplot as plt 
import numpy

labrat = mpimg.imread('../IMG/labrat.jpg')

ds1=numpy.dot(labrat[...,:3],[0.5,1,1])
ds2=numpy.dot(labrat[...,:3],[1,0.5,1])
fig0=plt.figure(figsize=(12,3))
plt.subplot(131)
plt.imshow(ds1)
plt.subplot(1,3,2)
plt.imshow(ds2)
plt.show()