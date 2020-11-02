import numpy as np
from PIL import Image
import nibabel as nib
import numpy as np

def make_2d_training_instance(mri_path, segmentation_path):

# find the slice of the png path that contains greatest 2D slice of resection
# find slice from segmentation path that has the most amount of white in the image (biggest part of resection)
# index this slice number to mri path and this will be the slice_png

    label = nib.load(segmentation_path)
    data = nib.get_fdata()
    x,y,z = data.shape
  
    print((x,y,z))
# to find the hemisphere use the slice_png and run the following to determine right or left
    
    slice_png_path = None

    im = Image.open(slice_png_path)
    xsize, ysize = im.size

    hemisphere = 'Left'
    rows, cols = np.where(im == 255)
    if (cols[cols>(ysize/2)]).size > (cols.size/2):
        hemisphere = 'Right'


    # return slice_png_path, hemisphere


make_2d_training_instance('t1_resected.nii.gz', 't1_resection_label.nii.gz')
