# -*- coding: utf-8 -*-

import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import utl

# reading in images
im_list = []
images = glob.glob(os.path.join("./test_images/", "*.jpg"))
num_images = len(images)
fig, axs = plt.subplots(len(images), 5, figsize=(5, 8))
for i in range(num_images):
    img = np.array(Image.open(images[i]))
    im_list.append(img)
    axs[i, 0].imshow(im_list[i])
    axs[i, 0].axis("off")

# convert image to gray scale
gray_im_list = []
for i in range(num_images):
    gray_im_list.append(utl.grayscale(im_list[i]))
    axs[i, 1].imshow(gray_im_list[i], cmap='gray')
    axs[i, 1].axis("off")

# define a kernel size and apply Gaussian smoothing
kernel_size = 5
gSmooth_im_list = []
for i in range(num_images):
    gSmooth_im_list.append(utl.gaussian_blur(gray_im_list[i], kernel_size))
    axs[i, 2].imshow(gSmooth_im_list[i], cmap='gray')
    axs[i, 2].axis("off")

# applies the Canny transform
edges_im_list = []
low_threshold = 50
high_threshold = 150
for i in range(num_images):
    edges_im_list.append(utl.canny(gray_im_list[i], low_threshold, high_threshold))
    axs[i, 3].imshow(edges_im_list[i], cmap='gray')
    axs[i, 3].axis("off")

# create a masked edges image
mask = np.zeros_like(edges_im_list[0])
ignore_mask_color = 255
im_shape = edges_im_list[0].shape
vertices = np.array([[(30, im_shape[0]),
                      (2.4 * im_shape[1] / 5, im_shape[0] / 2),
                      (2.6 * im_shape[1] / 5, im_shape[0] / 2),
                      (im_shape[1] - 30, im_shape[0])]],
                    dtype=np.int32)
masked_im_list = []
for i in range(num_images):
    masked_im_list.append(utl.region_of_interest(edges_im_list[i], vertices))
    axs[i, 4].imshow(masked_im_list[i], cmap='gray')
    axs[i, 4].axis("off")

plt.show()
