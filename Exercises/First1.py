import torchio
from torchio.datasets import FPG
import SimpleITK as sitk

FPG_data = FPG()
print(FPG_data.filenames['t1'])
# image = sitk.ReadImage(FPG_data.filenames['t1'])