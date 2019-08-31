import cv2
from consts import *


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=None, thickness=10, is_improved=False, x_range=[20, 910]):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    if is_improved:
        if color is None:
            color = [255, 0, 0]

        # separate line dots left and right
        left_lines = []
        right_lines = []
        left_x = []
        left_y = []
        right_x = []
        right_y = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 <= img.shape[1]/2 and x2 <= img.shape[1]/2:
                    left_lines.append(line)
                    left_x.append(x1)
                    left_y.append(y1)
                    left_x.append(x2)
                    left_y.append(y2)
                else:
                    right_lines.append(line)
                    right_x.append(x1)
                    right_y.append(y1)
                    right_x.append(x2)
                    right_y.append(y2)
        left_x = np.array(left_x)
        left_y = np.array(left_y)
        right_x = np.array(right_x)
        right_y = np.array(right_y)

        # liner approximation method
        if left_x.shape[0] > 2 and left_y.shape[0] > 2:
            left_res = np.polyfit(left_x, left_y, 1)
            left_min_x = int((539 - left_res[1]) / left_res[0])
            left_max_x = 460
            left_points = np.poly1d(left_res)([left_min_x, left_max_x])
            cv2.line(img, (left_min_x, int(left_points[0])), (left_max_x, int(left_points[1])), color, thickness)

        if right_x.shape[0] > 2 and right_y.shape[0] > 2:
            right_res = np.polyfit(right_x, right_y, 1)
            right_min_x = 500
            right_max_x = int((539 - right_res[1]) / right_res[0])
            right_points = np.poly1d(right_res)([right_min_x, right_max_x])
            cv2.line(img, (right_min_x, int(right_points[0])), (right_max_x, int(right_points[1])), color, thickness)
    else:
        if color is None:
            color = [255, 0, 0]
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, line_color=None, is_improved=False):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    if line_color is None:
        draw_lines(line_img, lines, is_improved=is_improved)
    else:
        draw_lines(line_img, lines, line_color, is_improved=is_improved)

    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


def find_lane_line(img, line_color=None, is_improved=False):
    gray_im = grayscale(img)
    gaussian_im = gaussian_blur(gray_im, KERNEL_SIZE)
    edges_im = canny(gaussian_im, LOW_THRESHOLD, HIGH_THRESHOLD)
    masked_im = region_of_interest(edges_im, VERTICES)
    line_im = hough_lines(masked_im, RHO, THETA, THRESHOLD, MIN_LINE_LENGTH, MAX_LINE_GAP, line_color, is_improved)
    hough_im = weighted_img(img, line_im)
    return hough_im


def get_frame_list(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    image_list = []
    for n in range(int(count)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, n)
        ret, frame = cap.read()
        if ret:
            image_list.append(frame)
        else:
            return image_list


def convert_frame_to_video(image_list, name, result_dir_path):

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    shape1 = image_list[0].shape[1]
    shape2 = image_list[0].shape[0]
    video = cv2.VideoWriter('{}{}.mp4'.format(result_dir_path, name), fourcc, 20.0, (shape1, shape2))

    for image in image_list:
        video.write(image)
