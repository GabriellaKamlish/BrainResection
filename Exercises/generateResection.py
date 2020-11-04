import numpy as np
import nibabel as nib
import torchio as tio
import matplotlib.pyplot as plt
from PIL import Image

def make_2d_training_instance(mri_path, segmentation_path):

# find the slice of the mri path that contains greatest 2D slice of resection

# find slice from segmentation path that has the most amount of white in the image (biggest part of resection)
# index this slice number to mri path and this will be the slice_png

    image = tio.ScalarImage(segmentation_path)
    image.load()
    if image.orientation != ('R','A','S'):
        raise Exception('Image orientation is not RAS')

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
    

    largestResectionAreaLabel = np.rot90(data[:,:, best_k])
    largestResectionAreaScan = np.rot90(data2[:,:, best_k])

    # Initialize the subplot panels side by side
    fig, ax = plt.subplots(nrows=2, ncols=1)

    # Show an image in each subplot
    ax[0].imshow(largestResectionAreaLabel)
    ax[0].set_title('Largest area of resection visualised from label (transverse plane)')
    ax[1].imshow(largestResectionAreaScan)
    ax[1].set_title('Largest area of resection visualised from scan (transverse plane)')

    plt.show()

    

# to find the hemisphere use the slice_png and run the following to determine right or left
    
    slice_png_path = largestResectionAreaLabel

    # im = Image.open(slice_png_path)
    # xsize, ysize = slice_png_path.size

    hemisphere = 'right'
    rows, cols = np.where(slice_png_path == 255)
    if (cols[cols>128]).size > (cols.size/2):
        hemisphere = 'left'

    print('The resection is mainly in the {} hemisphere'.format(hemisphere))
    # return slice_png_path, hemisphere


make_2d_training_instance('/Users/gabriellakamlish/BrainResection/Exercises/t1_resected.nii.gz', '/Users/gabriellakamlish/BrainResection/Exercises/t1_resection_label.nii.gz')




