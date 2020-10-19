import torchio
from torchio.datasets import FPG
import SimpleITK as sitk
from resector.parcellation import get_resectable_hemisphere_mask

FPG_data = FPG()
parcellation_path = '/tmp/noise/002_S_0295_I118671_t1_pre_NeuroMorph_Parcellation.nii.gz'
hemisphere = 'right'

mask = get_resectable_hemisphere_mask(
    parcellation_path,
    hemisphere,
)

sitk.WriteImage(mask, f'/tmp/noise/002_S_0295_I118671_t1_pre_resectable_{hemisphere}_seg.nii.gz')


