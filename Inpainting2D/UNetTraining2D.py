from UNetDataset import UNetDataset
from Unet2d import UNet
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import smtplib, ssl

# IXI dataset path
MNI_data_path = '/Users/gabriellakamlish/BrainResection/IXI/IXI_MNI'
IXI_dataset = UNetDataset(MNI_data_path)

dataloader = DataLoader(IXI_dataset, batch_size=5,shuffle=True, num_workers=0)

neural_net = UNet()

criterion = nn.MSELoss()
optimizer = optim.Adam(neural_net.parameters(), lr=0.001)

training_iterations = 1

for epoch in range(training_iterations):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        print(inputs.dtype)

        inputs = inputs.float()
        print(inputs.dtype)
        outputs = neural_net(inputs)
        print(outputs.dtype)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 5 == 4:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')

