import numpy as np
import nibabel as nib
import torchio as tio
import matplotlib.pyplot as plt
from PIL import Image

def make_2d_training_instance(mri_path, segmentation_path):

# find the slice of the mri path that contains greatest 2D slice of resection

# find slice from segmentation path that has the most amount of white in the image (biggest part of resection)
# index this slice number to mri path and this will be the slice_png

    label = tio.ScalarImage(segmentation_path)
    scan = tio.ScalarImage(mri_path)
    transform = tio.ToCanonical()

    if label.orientation != ('R','A','S'):
        label = transform(label)
    elif scan.orientation != ('R','A','S'):
        scan = transform(scan)

    label_data = label.data
    scan_data = scan.data

    data = label_data.numpy()
    data2 = scan_data.numpy()

    x,y,z = data.shape[1:4]

    max_White = 0
    for k in range(z):
        imageSlice = data[0,:,:,k]
        white = np.count_nonzero(imageSlice)
        if white > max_White:
            max_White = white
            best_k = k
    

    largestResectionAreaLabel = np.rot90(data[0,:,:, best_k])
    largestResectionAreaScan = np.rot90(data2[0,:,:, best_k])
    
    slice_png = Image.fromarray(largestResectionAreaScan)
    if slice_png.mode != 'RGB':
        slice_png = slice_png.convert('RGB')

    slice_png_path = '/Users/gabriellakamlish/BrainResection/Exercises/largestResectionAreaScan.png'
    slice_png.save(slice_png_path)
    
    # Initialize the subplot panels side by side
    fig, ax = plt.subplots(nrows=2, ncols=1)

    # Show an image in each subplot
    ax[0].imshow(largestResectionAreaLabel)
    ax[0].set_title('Largest area of resection visualised from label (transverse plane)')
    ax[1].imshow(largestResectionAreaScan)
    ax[1].set_title('Largest area of resection visualised from scan (transverse plane)')

    plt.show()
    
# to find the hemisphere use the slice_png and run the following to determine right or left

    hemisphere = 'left'
    rows, cols = np.where(largestResectionAreaLabel != 0)
    if (cols[cols>(largestResectionAreaLabel.shape[1]/2)]).size > (cols.size/2):
        hemisphere = 'right'

    print('The resection is located in the {} hemisphere of the brain'.format(hemisphere))
    return slice_png_path, hemisphere

if __name__ == "__main__": 
    mri_path = input('Path to the mri: ')
    segmentation_path = input('Path to the label: ')
#   /Users/gabriellakamlish/BrainResection/Exercises/t1_resected.nii.gz
#   /Users/gabriellakamlish/BrainResection/Exercises/t1_resection_label.nii.gz  
    make_2d_training_instance(mri_path, segmentation_path)
