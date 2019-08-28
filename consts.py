import numpy as np

# setting for subplot
NUM_PLOTS = 6
FIGURE_SIZE = (16, 8)
W_SPACE = 0.01
H_SPACE = 0.01

# parameter for Gaussian smoothing
KERNEL_SIZE = 5

# parameters for Canny transform
LOW_THRESHOLD = 30
HIGH_THRESHOLD = 180

# parameters for mask size = 930 x 540
VERTICES = np.array([[(30, 540), (460, 300), (499, 300), (930, 540)]], dtype=np.int32)

# parameters for Hough Transform
RHO = 1
THETA = np.pi / 180
THRESHOLD = 15
MIN_LINE_LENGTH = 30
MAX_LINE_GAP = 80
