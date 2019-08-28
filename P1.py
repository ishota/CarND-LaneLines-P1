# -*- coding: utf-8 -*-

import os
import glob
import matplotlib.pyplot as plt
from PIL import Image
import utl
import cv2
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
        lines = cv2.HoughLinesP(masked_im_list[i], RHO, THETA, THRESHOLD, np.array([]), MIN_LINE_LENGTH, MAX_LINE_GAP)
        line_img = np.zeros((masked_im_list[i].shape[0], masked_im_list[i].shape[1], 3), dtype=np.uint8)
        utl.draw_lines(line_img, lines, thickness=5)
        color_edges = np.dstack((masked_im_list[i], masked_im_list[i], masked_im_list[i]))
        hough_im_list.append(utl.weighted_img(color_edges, line_img))
        axs[i, 5].imshow(hough_im_list[i])
        axs[i, 5].axis("off")

    plt.show()


if __name__ == '__main__':
    main()
