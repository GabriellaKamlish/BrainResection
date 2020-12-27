import torchio as tio
import numpy as np 
import random 
from PIL import Image
import torch 
import nibabel as nib
import scipy.misc
import matplotlib.pyplot as plt
import cv2

def generate_resection(segmentation_path, T1_path):

    label = tio.ScalarImage(segmentation_path)
    T1 = tio.ScalarImage(T1_path)
    transform = tio.ToCanonical()

    label_data = label.data
    T1_data = T1.data

    label_array = label_data.numpy()
    T1_array = T1_data.numpy()
    # Remove 4th dimension from nifty
    T1_array = np.squeeze(T1_array)
    label_array = np.squeeze(label_array)
    
    planes = T1_array.shape
    # print(x,y,z)

    # first value fixed makes frontal plane P->A
    # second value fixed makes transverse plane I->S
    # third value fixed makes sagittal plane L->R

    # check min and max value for front, trans, sagittal, where label is non zero 
    # iterate through and take slice of label, find np.nonzero of new cross section
    # choose random pixel from new np.nonzero 
    # create a black image and place random sized rectangle starting from pixel
    # find itersection between label slice and rectangle (save as new label)
    # apply intersection to t1 slice and save

    it = 0
    for plane in planes:
        it += 1
        for i in range(plane-1):
            # takes slice dependent on orientation
            if it == 1:
                label_slice = label_array[i,:,:]
                T1_slice = T1_array[i,:,:]
            elif it == 2:
                label_slice = label_array[:,i,:]
                T1_slice = T1_array[:,i,:]
            else:
                label_slice = label_array[:,:,i]
                T1_slice = T1_array[:,:,i]
            # finds location of non zero pixels
            label_loc = np.nonzero(label_slice)
            # number of non zero pixels
            label_loc_size = label_loc[0].size
            # skips any slices which do not contain brain 
            if label_loc_size != 0:
                # generate random value in label_loc size
                random_start = random.randint(0,label_loc_size-1)
                # extract point of random value
                resection_start_y = label_loc[0][random_start]
                resection_start_z = label_loc[1][random_start]
                # generate random rectangle size
                resection_width = random.randint(25,100)
                resection_height = random.randint(25,100)
                # create resection image and remake label to be same dtype
                if it == 1 or it == 2:
                    resection_image = np.zeros((256, 150), np.uint8)
                    label = np.zeros((256, 150), np.uint8)

                else:
                    resection_image = np.zeros((256, 256), np.uint8)
                    label = np.zeros((256, 256), np.uint8)

                # apply resection to image
                resection_image[resection_start_y:resection_start_y+resection_width, resection_start_z:resection_start_z+resection_height] = 255
                # remake label to be same data type as resection image
                label[label_loc[0],label_loc[1]] = 255
                # bitwise and to get intersection of resection and label
                img_bwa = cv2.bitwise_and(resection_image,label)
                # apply resection to T1
                resec_loc = np.nonzero(img_bwa)
                T1_resected = T1_slice.copy()
                T1_resected[resec_loc[0],resec_loc[1]]=0

                # save new label and t1
                if it == 1:
                    plt.imsave('TEST/Labels/frontal_label_slice_{}.png'.format(i), img_bwa)
                    plt.imsave('TEST/T1_resected/frontal_T1_resected_slice_{}.png'.format(i), T1_resected)
                    plt.imsave('TEST/T1/frontal_T1_slice_{}.png'.format(i), T1_slice)

                elif it == 2:
                    plt.imsave('TEST/Labels/transverse_label_slice_{}.png'.format(i), img_bwa)
                    plt.imsave('TEST/T1_resected/transverse_T1_resected_slice_{}.png'.format(i), T1_resected)
                    plt.imsave('TEST/T1/transverse_T1_slice_{}.png'.format(i), T1_slice)
               
                else:
                    plt.imsave('TEST/Labels/sagittal_label_slice_{}.png'.format(i), img_bwa)
                    plt.imsave('TEST/T1_resected/sagittal_T1_resected_slice_{}.png'.format(i), T1_resected)
                    plt.imsave('TEST/T1/sagittal_T1_slice_{}.png'.format(i), T1_slice)
            else:
                continue


if __name__ == "__main__":
    generate_resection('/Users/gabriellakamlish/BrainResection/IXI/brain_segs/IXI002-Guys-0828-T1_NeuroMorph_Parcellation_gif_brain_seg.nii.gz', '/Users/gabriellakamlish/BrainResection/IXI/T1/IXI002-Guys-0828-T1.nii.gz')
