import numpy as np
from PIL import Image

image = np.zeros((256, 256), np.uint8)
image[120:200, 120:136] = 255
Image.fromarray(image).save('square.png')

im = Image.open('square.png')
isize = im.size

result = 'Top'
white = np.where(image == 255)
x_coords = white[0]
if (x_coords[x_coords>128]).size > (x_coords.size/2):
    result = 'Bottom'

print(result)
        