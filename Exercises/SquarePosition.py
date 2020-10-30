import numpy as np
from PIL import Image

image = np.zeros((256, 256), np.uint8)
image[120:200, 120:136] = 255
Image.fromarray(image).save('square.png')

im = Image.open('square.png')
x,y = im.size

result = 'Top'
rows, cols = np.where(image == 255)
if (rows[rows>128]).size > (rows.size/2):
    result = 'Bottom'

print(result)
        