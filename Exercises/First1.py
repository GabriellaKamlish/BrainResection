import SimpleITK as sitk
import torchio
import numpy
from resector.parcellation import get_resectable_hemisphere_mask
from torchio.datasets import FPG
import nibabel as nib
import matplotlib.pyplot as plt

# fpg data
FPG_data = FPG()
# FPG.plot(FPG_data)

# 3D Nifti images
test_image=nib.load('t1_resected.nii.gz').get_fdata()
test_mask=nib.load('t1_resection_label.nii.gz').get_fdata()
# (176, 256, 256)

# fig, (ax1, ax2) = plt.subplots(1,2, figsize = (12, 6))
# ax1.imshow(test_image)
# ax1.set_title('Image')
# ax2.imshow(test_mask)
# ax2.set_title('Mask')


# print(torchio.datasets.FPG().seg.path)
# # path to brain parcellation data from fpg
# parcellation_path = '/tmp/noise/002_S_0295_I118671_t1_pre_NeuroMorph_Parcellation.nii.gz'

# # labelling using resector
# hemisphere = 'right'

# mask = get_resectable_hemisphere_mask(
#     parcellation_path,
#     hemisphere,
# )

# sitk.WriteImage(mask, f'/tmp/noise/002_S_0295_I118671_t1_pre_resectable_{hemisphere}_seg.nii.gz')

# resect /Users/gabriellakamlish/.cache/torchio/fpg/t1.nii.gz /Users/gabriellakamlish/.cache/torchio/fpg/t1_seg_gif.nii.gz t1_resected.nii.gz t1_resection_label.nii.gz