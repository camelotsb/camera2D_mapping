'''def do_normalise(image):
    return -np.log(1/((1 + image)/257) - 1)
 def undo_normalise(image):
    return (1 + 1/(np.exp(-image) + 1) * 257).astype("uint8")
def rotation_matrix(theta):
    return np.c_[
        [1,0,0],
        [0,np.cos(theta),-np.sin(theta)],
        [0,np.sin(theta),np.cos(theta)]
    ]
img_norm = do_normalise(image)
img_rot = np.einsum("ijk,lk->ijl", img_norm, rotation_matrix(np.pi))
img = undo_normalise(img_rot)
 
imgplot = plt.imshow(img)'''

'''
def simple_threshold(image, threshold=128):
    return ((image > threshold) * 255).astype("uint8")
 
thresholds = [100,120,128,138,150]
 
fig, axs = plt.subplots(nrows=1, ncols=len(thresholds), figsize=(20,5));
gray_im = to_grayscale(image)
                        
for t, ax in zip(thresholds, axs):
    ax.imshow(simple_threshold(gray_im, t), cmap='Greys');
    ax.set_title("Threshold: {}".format(t), fontsize=20);
    ax.set_axis_off();
    '''