import os
from os import listdir
from os.path import isfile, join
import torch
import torchio as tio
import numpy as np
import cv2
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

class ImageAugmentor:
    def __init__(self, images_path, batch_size = 10, dim = (181,181), n_channels=1, shuffle = False):
        
        self.images_path = images_path
        self.image_names = [f for f in listdir(images_path) if isfile(join(images_path, f))]
        self.image_names.remove('.DS_Store')
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.n_channels = n_channels
        self.on_epoch_end()


    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Generate indexes of the batch
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]

        # Generate Data
        return self.__data_generation(indexes)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_names))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, idxs):
        X_batch = np.empty((self.batch_size, self.dim[0], self.dim[1]))
        y_batch = np.empty((self.batch_size, self.dim[0], self.dim[1]))

        for i, idx in enumerate(idxs):
            img_name = os.path.join(self.images_path, self.image_names[idx])
            image = Image.open(img_name)
            image = ImageOps.grayscale(image)
            image = np.asarray(image)
            image_copy = image.copy()

            # Get mask associated to that image
            masked_image = self.__createMask(image_copy)

            # normalise
            X_batch[i,] = masked_image/255
            y_batch[i] = image/255
        
        return X_batch, y_batch

    def __createMask(self, img):
        ## Prepare masking matrix
        mask = np.full((181,181,1), 255, dtype=np.uint8)
        # Get random x locations to start line
        x1, x2 = np.random.randint(1, 181), np.random.randint(1, 181)
        # Get random y locations to start line
        y1, y2 = np.random.randint(1, 181), np.random.randint(1, 181)
        # Draw black rectangle on the white mask
        cv2.rectangle(mask,(x1,y1),(x2,y2),(1,1,1),-1)
        # Perforn bitwise and operation to mak the image

        masked_image = cv2.bitwise_and(img.astype(int), mask.astype(int))
        return masked_image


if __name__ == "__main__":
    augmented_brains = ImageAugmentor('/Users/gabriellakamlish/BrainResection/IXI/T1/Slices', batch_size=10, shuffle= True)
    print(len(augmented_brains))
    a_b = augmented_brains[5]
    print(a_b[0][0].shape)
    

    sample_masks, sample_labels = augmented_brains[1]
    # sample_masks, sample_labels = traingen[sample_idx]
    sample_images = [None]*(len(sample_masks)+len(sample_labels))
    sample_images[::2] = sample_labels
    sample_images[1::2] = sample_masks

    fig = plt.figure(figsize=(16., 8.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(4, 5),  # creates 2x2 grid of axes
                    axes_pad=0.3,  # pad between axes in inch.
                    )

    for ax, image in zip(grid, sample_images):
        ax.imshow(image)

    plt.show()