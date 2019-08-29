# -*- coding: utf-8 -*-

import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
import utl
from consts import *


def main():
    # reading in images
    im_list = []
    images = glob.glob(os.path.join("./test_images/", "*.jpg"))
    num_images = len(images)
    fig, axs = plt.subplots(len(images), NUM_PLOTS, figsize=FIGURE_SIZE)
    plt.subplots_adjust(wspace=W_SPACE, hspace=H_SPACE)
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
    gaussian_im_list = []
    for i in range(num_images):
        gaussian_im_list.append(utl.gaussian_blur(gray_im_list[i], KERNEL_SIZE))
        axs[i, 2].imshow(gaussian_im_list[i], cmap='gray')
        axs[i, 2].axis("off")

    # applies the Canny transform
    edges_im_list = []
    for i in range(num_images):
        edges_im_list.append(utl.canny(gaussian_im_list[i], LOW_THRESHOLD, HIGH_THRESHOLD))
        axs[i, 3].imshow(edges_im_list[i], cmap='gray')
        axs[i, 3].axis("off")

    # create a masked edged image
    masked_im_list = []
    for i in range(num_images):
        masked_im_list.append(utl.region_of_interest(edges_im_list[i], VERTICES))
        axs[i, 4].imshow(masked_im_list[i], cmap='gray')
        axs[i, 4].axis("off")

    # Define the Hough transform parameters
    hough_im_list = []
    for i in range(num_images):
        line_img = utl.hough_lines(masked_im_list[i], RHO, THETA, THRESHOLD, MIN_LINE_LENGTH, MAX_LINE_GAP)
        hough_im_list.append(utl.weighted_img(im_list[i], line_img))
        axs[i, 5].imshow(hough_im_list[i])
        axs[i, 5].axis("off")

    plt.show()
    os.mkdir("test_images_output")
    plt.savefig('test_images_output/images.png')


if __name__ == '__main__':
    main()
