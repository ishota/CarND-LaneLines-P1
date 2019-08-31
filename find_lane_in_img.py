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
    plot_count = 0
    fig, axs = plt.subplots(NUM_PLOTS, len(images), figsize=FIGURE_SIZE)
    plt.subplots_adjust(wspace=W_SPACE, hspace=H_SPACE)
    for i in range(num_images):
        img = np.array(Image.open(images[i]))
        im_list.append(img)
        axs[plot_count, i].imshow(im_list[i])
        axs[plot_count, i].axis("off")
    plot_count += 1

    # convert image to gray scale
    gray_im_list = []
    for i in range(num_images):
        gray_im_list.append(utl.grayscale(im_list[i]))
        axs[plot_count, i].imshow(gray_im_list[i], cmap='gray')
        axs[plot_count, i].axis("off")
    plot_count += 1

    # define a kernel size and apply Gaussian smoothing
    gaussian_im_list = []
    for i in range(num_images):
        gaussian_im_list.append(utl.gaussian_blur(gray_im_list[i], KERNEL_SIZE))
        axs[plot_count, i].imshow(gaussian_im_list[i], cmap='gray')
        axs[plot_count, i].axis("off")
    plot_count += 1

    # applies the Canny transform
    edges_im_list = []
    for i in range(num_images):
        edges_im_list.append(utl.canny(gaussian_im_list[i], LOW_THRESHOLD, HIGH_THRESHOLD))
        axs[plot_count, i].imshow(edges_im_list[i], cmap='gray')
        axs[plot_count, i].axis("off")
    plot_count += 1

    # compute a masked edged image
    masked_im_list = []
    for i in range(num_images):
        masked_im_list.append(utl.region_of_interest(edges_im_list[i], VERTICES))
        axs[plot_count, i].imshow(masked_im_list[i], cmap='gray')
        axs[plot_count, i].axis("off")
    plot_count += 1

    # compute a line image
    for i in range(num_images):
        line_img = utl.hough_lines(masked_im_list[i], RHO, THETA, THRESHOLD, MIN_LINE_LENGTH, MAX_LINE_GAP)
        axs[plot_count, i].imshow(line_img)
        axs[plot_count, i].axis("off")
    plot_count += 1

    # applies find line pipeline
    hough_im_list = []
    for i in range(num_images):
        hough_im_list.append(utl.find_lane_line(im_list[i], line_color=[255, 0, 0], is_improved=IS_IMPROVED))
        axs[plot_count, i].imshow(hough_im_list[i])
        axs[plot_count, i].axis("off")
    plot_count += 1

    if not os.path.exists('test_images_output'):
        os.mkdir('test_images_output')
    plt.savefig('test_images_output/images.jpg')


if __name__ == '__main__':
    main()
