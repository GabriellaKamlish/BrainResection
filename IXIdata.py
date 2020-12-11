# Get the size of the segmentation image

# Generate a random square within those confines

# Check the indices of the square and compare to indices of segmentation

# If there are any indices in square that arent in segmentation then restart

# otherwise apply the square to the actual T1 image


import torchio as tio
import numpy as np 
import random 
from PIL import Image

def generate_resection(segmentation_path, T1_path):

    label = tio.ScalarImage(segmentation_path)
    T1 = tio.ScalarImage(T1_path)
    transform = tio.ToCanonical()

    if label.orientation != ('R','A','S'):
        label = transform(label)
    elif T1.orientation != ('R','A','S'):
        T1 = transform(T1)

    label_data = label.data
    T1_data = T1.data

    label_array = label_data.numpy()
    T1_array = T1_data.numpy()

    x,y,z = T1_array.shape[1:4]
    print(x,y,z)

    label_loc = np.nonzero(label_array)
    nonzero_x = label_loc[1]
    nonzero_y = label_loc[2]
    nonzero_z = label_loc[3]

    # choose random indice in the length on one of the nonzero
    # choose random number to be length of square
    # check if corresponding indices are in nonzero array, if yes then continue if not new random indice
    while True:
        N = random.randint(0,nonzero_x.shape)
        Square_length = random.randint(10,50)
        Sq_x_dim = [nonzero_x[N], nonzero_y[N],nonzero_z[N]]
        if ((nonzero_x[N] + Square_length) in nonzero_x) and ((nonzero_y[N] + Square_length) in nonzero_y) and ((nonzero_z[N] + Square_length) in nonzero_z):
            square_image = np.zeros((x,y,z), np.uint8)
            square_image[nonzero_x[N]:nonzero_x[N]+ Square_length, nonzero_y[N]:nonzero_y[N]+ Square_length, nonzero_z[N]:nonzero_z[N]+ Square_length] = 255
            Image.fromarray(square_image).save('test.png')
            False
        else:
            continue
        
    





generate_resection('/Users/gabriellakamlish/BrainResection/IXI/brain_segs/IXI002-Guys-0828-T1_NeuroMorph_Parcellation_gif_brain_seg.nii.gz', '/Users/gabriellakamlish/BrainResection/IXI/T1/IXI002-Guys-0828-T1.nii.gz')
