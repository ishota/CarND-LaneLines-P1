import numpy as np

# setting for subplot
NUM_PLOTS = 7
FIGURE_SIZE = (12, 8)
W_SPACE = 0.01
H_SPACE = 0.01

# parameter for Gaussian smoothing
KERNEL_SIZE = 5

# parameters for Canny transform
LOW_THRESHOLD = 20
HIGH_THRESHOLD = 200

# parameters for mask (img size = 960 x 540)
VERTICES = np.array([[(0, 540), (460, 320), (500, 320), (960, 540)]], dtype=np.int32)

# parameters for Hough Transform
RHO = 1
THETA = np.pi / 180
THRESHOLD = 25
MIN_LINE_LENGTH = 15
MAX_LINE_GAP = 20


# parameters for file name
VID_NAME = 'solidWhiteRight'
# VID_NAME = 'solidYellowLeft'
# VID_NAME = 'challenge'

# trigger for improved method
IS_IMPROVED = True

# trigger for advanced method
