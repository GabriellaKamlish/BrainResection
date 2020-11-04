import numpy as np
import nibabel as nib
import torchio as tio
from torchio import transforms
from skimage import io
import matplotlib.pyplot as plt

def make_2d_training_instance(mri_path, segmentation_path):

# find the slice of the mri path that contains greatest 2D slice of resection

# find slice from segmentation path that has the most amount of white in the image (biggest part of resection)
# index this slice number to mri path and this will be the slice_png

    image = tio.ScalarImage(segmentation_path)
    image.load()
    x,y,z = image.shape[1:4]

    label = nib.load(segmentation_path)
    data = label.get_fdata()
    scan = nib.load(mri_path)
    data2 = scan.get_fdata()

    max_White = 0
    for k in range(z):
        imageSlice = data[:,:,k]
        white = np.count_nonzero(imageSlice)
        if white > max_White:
            max_White = white
            best_k = k
    
    # Initialize the subplot panels side by side
    fig, ax = plt.subplots(nrows=2, ncols=1)

    # Show an image in each subplot
    ax[0].imshow(data[:,:, best_k])
    ax[0].set_title('Largest area of resection visualised from label (transverse plane)')
    ax[1].imshow(data2[:,:, best_k])
    ax[1].set_title('Largest area of resection visualised from scan (transverse plane)')

    plt.show()


# to find the hemisphere use the slice_png and run the following to determine right or left
    
    # slice_png_path = None

    # im = Image.open(slice_png_path)
    # xsize, ysize = im.size

    # hemisphere = 'Left'
    # rows, cols = np.where(im == 255)
    # if (cols[cols>(ysize/2)]).size > (cols.size/2):
    #     hemisphere = 'Right'


    # return slice_png_path, hemisphere


make_2d_training_instance('/Users/gabriellakamlish/BrainResection/Exercises/t1_resected.nii.gz', '/Users/gabriellakamlish/BrainResection/Exercises/t1_resection_label.nii.gz')




