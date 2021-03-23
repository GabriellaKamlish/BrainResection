from pathlib import Path
import torch
import torchio as tio
from model import GifNetEncoder


torch.set_grad_enabled(False)

this_dir = Path(__file__).parent
weights_path = this_dir / 'weights.pth'

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
model = GifNetEncoder()
state_dict = torch.load(weights_path)
model.load_state_dict(state_dict)
model.eval().to(device)

transform = tio.Compose((
    tio.ToCanonical(),
    tio.RescaleIntensity((-1, 1)),
))

colin = tio.datasets.Colin27()
t1_preprocessed = transform(colin.t1)
inputs = t1_preprocessed.data.unsqueeze(0).to(device)
features = model(inputs)

print('Features:', features.shape)
