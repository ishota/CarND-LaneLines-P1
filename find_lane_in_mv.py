# -*- coding: utf-8 -*-

import utl


def main():


    # divide videos into img
    # num_frame = utl.save_frame('test_videos/solidWhiteRight.mp4', 'test_videos/img_solidWhiteRight/', 'solidWhiteRight')
    num_frame = 214

    # create video from img
    utl.convert_frame_to_video('test_videos/img_solidWhiteRight/', int(num_frame), 'solidWhiteRight', 'test_videos_output/', 'output')


if __name__ == '__main__':
    main()
