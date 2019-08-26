# -*- coding: utf-8 -*-

import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import utl

# reading in images
im_list = []
images = glob.glob(os.path.join("./test_images/", "*.jpg"))
for i in range(len(images)):
    img = np.array(Image.open(images[i]))
    im_list.append(img)
    plt.imshow(im_list[i])
    # plt.show()

# convert image to gray scale
plt.imshow(utl.grayscale(im_list[1]), cmap='gray')
plt.show()
