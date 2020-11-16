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
# generate resect files using

# resect /Users/gabriellakamlish/.cache/torchio/fpg/t1.nii.gz /Users/gabriellakamlish/.cache/torchio/fpg/t1_seg_gif.nii.gz t1_resected.nii.gz t1_resection_label.nii.gz


# mv /Users/gabriellakamlish/.cache/torchio/fpg/t1.nii.gz 