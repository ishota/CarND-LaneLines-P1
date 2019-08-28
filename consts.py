import numpy as np

# setting for subplot
NUM_PLOTS = 6
FIGURE_SIZE = (16, 8)
W_SPACE = 0.01
H_SPACE = 0.01

# parameter for Gaussian smoothing
KERNEL_SIZE = 5

# parameters for Canny transform
LOW_THRESHOLD = 50
HIGH_THRESHOLD = 150

# parameters for mask

# parameters for Hough Transform
RHO = 1
THETA = np.pi / 180
THRESHOLD = 15
MIN_LINE_LENGTH = 40
MAX_LINE_GAP = 20
