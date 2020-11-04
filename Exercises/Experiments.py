import SimpleITK as sitk
import torchio
# from resector.parcellation import get_resectable_hemisphere_mask
from torchio.datasets import FPG
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# fpg data

FPG_data = FPG()
# FPG.plot(FPG_data)
image = FPG_data.t1
print(image)
# generate resect files using

# resect /Users/gabriellakamlish/.cache/torchio/fpg/t1.nii.gz /Users/gabriellakamlish/.cache/torchio/fpg/t1_seg_gif.nii.gz t1_resected.nii.gz t1_resection_label.nii.gz



image = tio.ScalarImage(segmentation_path)
image.load()

transform = tio.ToCanonical()
ras_sample = transform(image)
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

