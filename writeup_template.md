## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration2.jpg "Calibration"
[image2]: ./test_images/straight_lines1.jpg "Original image"
[image3]: ./output_images/h_channel.jpg "H channel"
[image4]: ./output_images/s_channel.jpg "S channel"
[image5]: ./output_images/l_channel.jpg "L channel"
[image6]: ./output_images/x_gradient.jpg "x gradient"
[image7]: ./output_images/y_gradient.jpg "y gradient"
[image8]: ./output_images/gradient_magnitude.jpg "Gradient magnitude"
[image9]: ./output_images/combined_gradients.jpg "Combined gradient threshold"
[image10]: ./output_images/yellow_mask.jpg "Yellow threshold"
[image11]: ./output_images/white_mask.jpg "White threshold"
[image12]: ./output_images/yellow_white_mask.jpg "Yellow and white thresholds"
[image13]: ./output_images/color_thresh_h.jpg "Channel H color threshold"
[image14]: ./output_images/color_thresh_l.jpg "Channel L color threshold"
[image15]: ./output_images/color_thresh_s.jpg "Channel S color threshold"
[image16]: ./output_images/channels_thresh_combined.jpg "Channels HSL color threshold combined"
[image17]: ./output_images/grad_color_thresh.jpg "Information separated by threshold"
[image18]: ./output_images/roi.jpg "Region of interest"
[image19]: ./output_images/masked_roi.jpg "Masked region of interest"
[image20]: ./output_images/perspective_vertices.jpg "Perspective vertices selection"
[image21]: ./output_images/warped.jpg "Perspective change"
[image22]: ./output_images/window_search.jpg "Centroids defining lanes"
[image23]: ./output_images/fitted_warped.jpg "Fitted polygon bound by lanes"
[image24]: ./output_images/fitted_unwarped.jpg "Fitted lane unwarped"
[image25]: ./output_images/undistorted.jpg "Undistorted image"
[video1]: ./project_output.mp4 "Video"
[video2]: ./challenge_output.mp4 "Challenge video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the fourth code cell of the IPython notebook located in "./Advanced_lane_lines.ipynb" (or in lines # through # of the file called `some_file.py`).
The actual code of the function is in the file "./helper.py".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgPoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will show how I applied the distortion correction to one of the test images:
![alt text][image2]
![alt text][image25]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

First I transformed the image from RGB to HSL colorspace and applied contrast limited adaptive histogram equalization over the three channels to smooth the contrast of the image.
I used a combination of specific color, independent channel values and gradient thresholds to generate a binary image (thresholding steps from lines 18 to 61 of the first cell).
I also applied a morphologicla close operation after the gradients threshold and the final one to fill in any possible gaps inside the lines.
Here's an example of my output for each step included in the overall threshold.

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]
![alt text][image17]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes opencv function `warpPerspective()`, which appears in line 64. I chose to hardcode the source and destination points in the following manner:

```python
low_y = imshape[0] - 50
high_y = imshape[0] / 2 + 72

src_vertices = np.array(
    [(565, 470),
     (275, 670),
     (1035, 670),
     (720, 470),
    ])
roi_vertices = np.array(
    [(220,low_y),
     (imshape[1]/2-25, high_y), 
     (imshape[1]/2+60, high_y), 
     (imshape[1]-130,low_y)
    ], dtype=np.int32)
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. I also drawed the region of interest.

![alt text][image18]
![alt text][image19]
![alt text][image20]
![alt text][image21]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Once the image has been warped I check if we already have a lane detected, if not I do a full search using a sliding window method.
We run a convolutional kernel (5x5) with value 1 to identify areas with lots of pixels.
The algorithm reads the convoluted image and moves the window calculating the sum of all values within that window.
For each row it calculates the median to rule out any outliers and areas with a sum lower than a certain threshold.
It then proceeds to calculate the centroid on each remaning area and appends them to a vector.
Once the process is finished we have a collection of points that we use to fit the current image.
If we already had a line detected, the process is similar but much faster, since we only look in an area around the best fit line.
The functions are `window_search()` and `guided_window_search()` and can be found in the "./helper.py" file.

Once all of this is done we save the current fit coefficients from the latest 5 frames.
To update the best fit, avoid sudden changes (jitter) and prevent old values from having too much impact we use a weigthed average of
the coefficients over the last 5 frames, giving more weight to recent fits.

![alt text][image22]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Radius of curvature is calculated inside the function `update_line()` in "./helper.py", lines 313-319.
The position of the vehicle is calculated in the second cell of the notebook, lines 86-99.
To calculate the radius of curvature I just followed the instructions given, applying the formula
and transforming from pixel space to real world space.
For the position of the vehicle we calculate the pixel distance between the left and the right fit on the bottom of the image.
Then we calculate the difference from the pixel in the center of the image (center of the car), and the center of the lane.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the second cell of the notebook, lines 109 through 117. Here is an example of my result on a test image:

![alt text][image23]
![alt text][image24]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result.](./project_output.mp4)

Here's a [link to the challenge video result.](./challenge_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I think the pipeline is generally good, I spent a lot of time trying to understand potential problems and shortcomings of several approaches.
There are still some points I couldn't work on further due to time constraints:

- Darker parts of the road due to sudden changes in image contrast will make the thresholds not detect the lanes (as seen in the challenge video).
- Lack of an automatic method to create a region of interest and most importantly transform source points.
- When driving over sharp turns where one of the lanes gets out of the region of interest will not work.
