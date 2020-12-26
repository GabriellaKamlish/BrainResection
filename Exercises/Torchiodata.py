import torchio as tio 

transforms = [
     tio.ToCanonical(),  # to RAS
     tio.Resample((1, 1, 1)),  # to 1 mm iso
]

ixi_dataset = tio.datasets.IXI(
    'IXI/',
     modality=('T1'),
     transform=tio.Compose(transforms),
     download=True,
)
print('Number of subjects in dataset:', len(ixi_dataset))  # 577
sample_subject = ixi_dataset[0]
print('Keys in subject:', tuple(sample_subject.keys()))  # ('T1', 'T2')
print('Shape of T1 data:', sample_subject['T1'].shape)  # [1, 180, 268, 268]


