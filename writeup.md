# **Finding Lane Lines on the Road** 

## Writeup

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report

[result]: ./test_images_output/images.jpg

---

### 1. Finding lane line pipeline. 

My pipeline consisted of 6 steps. 

1. Convert the images to grayscale. 
2. Apply Gaussian smoothing to be any odd number for suppressing noise and spurious gradients by averaging. 
3. Detect edges in image by Canny edge detection algorithm. 
4. Extract the area around the lane lines in the image by covering it with a trapezoidal black image. 
5. Implement a Hough transform on image created by step 4. 
6. Perform a linear approximation on the points obtained by the Hough transform. 

#### Details of draw_lines extension by linear approximation. 

By performing linear approximation, it is possible to represent where there is no lane lines in the image. 
First, finds whether the end point of the lane line obtained by the Hough transform is on the left or right side of the image. 
Second, find linear approximation function on the let and right respectively. 
Third, compute an inverse of the function and find the region of the function in the image, especially in the area not covered with the black trapezoid. 

#### Result of the proposed pipeline

A ``find_lane_in_img.py`` in this project will make a result of the proposed pipeline for sample images. 
The result is, from top to bottom, original image, image converted to grayscale, Gaussian smoothed image, 
image detected edges by Canny's algorithm, image extracted from a trapezoidal range surrounded lane lines, 
image after performed linear approximation, and original image with the approximate line. 

![alt_txt][result]

A ``find_lane_in_mv.py`` in this project will make a result of the proposed pipeline for sample video. 
After implement it, you can watch video with the approximate line at ``test_videos_output/`` directory.

### 2. Potential shortcomings with the pipeline

#### Depends on the accuracy of edge detection algorithm

When the edge detection accuracy of the Canny algorithm is low, the accuracy of the line acquired by the Hough transform is also low.
Since linear approximation is performed using the end points of the line obtained by the Hough transform, 
the result of the linear approximation depends on the result of the Hough transform.
In particular, if the end points are biased, the error in the slope of the approximate line tends to increase. 

For example, when the difference between the road color and the lane line color is small, it is often impossible to detect at Canny edge algorithms.
In this case, even if linear approximation is used, the lane cannot be detected.

#### Cannot detect curve lines. 

Since it is a linear approximation, the detection error for straight roads is low, but for curved roads is high. 
Here, the detection error is expressed by averaging the squares of the sample points and the approximate function.

### 3. Possible improvements to the pipeline

#### Use past image frames

The line lanes appear regularly and continuously a video. 
So, we can use the past frame to improve the detection accuracy of the current frame.
Here, I propose a discount rate (gamma < 1) is applied to the past frame so as not too be influenced current frame. 
In other words, using afterimages.

#### Fit a curve using polynomial approximation

Find the function of a curved lane lines by using polynomial approximation.
We increase the number of edge points for approximation by using afterimages,
so the approximation error is considered to be small. 
