# -*- coding: utf-8 -*-

import utl
from consts import *


def main():

    # divide videos into img
    image_list = utl.get_frame_list('test_videos/' + VID_NAME + '.mp4')

    # detect lane line
    line_color = [0, 0, 255]
    for n, image in enumerate(image_list):
        image_list[n] = utl.find_lane_line(image, line_color, is_improved=IS_IMPROVED)

    # create video from img
    utl.convert_frame_to_video(image_list, VID_NAME, 'test_videos_output/')


if __name__ == '__main__':
    main()
