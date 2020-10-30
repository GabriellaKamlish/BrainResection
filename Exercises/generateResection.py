import numpy as np
from PIL import Image

def make_2d_training_instance(mri_path, segmentation_path):

    im = Image.open(segmentation_path)
    xsize, ysize = im.size

    hemisphere = 'Left'
    rows, cols = np.where(im == 255)
    if (cols[cols>(ysize/2)]).size > (cols.size/2):
        hemisphere = 'Right'


    slice_png_path = None

    return slice_png_path, hemisphere