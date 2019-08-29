# -*- coding: utf-8 -*-

import utl
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from consts import *


def main():

    # divide videos into img
    num_frame = utl.save_frame('test_videos/' + VID_NAME + '.mp4', 'test_videos/img_' + VID_NAME + '/', VID_NAME)

    # detect lane line
    for n in range(num_frame):
        img_output_dirs = "test_videos/img_" + VID_NAME + "/"
        if not os.path.exists(img_output_dirs):
            os.mkdir(img_output_dirs)
        os.makedirs(img_output_dirs)
        image = mpimg.imread(img_output_dirs + VID_NAME + '_' + '{0:03d}.jpg'.format(n))
        plt.imshow(utl.find_lane_line(image))
        vid_output_dirs = "test_videos_output/img_" + VID_NAME
        if not os.path.exists(vid_output_dirs):
            os.mkdir(vid_output_dirs)
        os.makedirs(vid_output_dirs)
        plt.savefig('test_videos_output/img_' + VID_NAME + '/' + VID_NAME + '_{0:03d}.jpg'.format(n))

    # create video from img
    utl.convert_frame_to_video('test_videos_output/img_' + VID_NAME + '/', num_frame, VID_NAME, 'test_videos_output/')


if __name__ == '__main__':
    main()
