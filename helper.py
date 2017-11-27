import glob

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage.measurements import center_of_mass

def camera_calibration(cal_path, output_images):
    # Camera calibration
    # Prepare dimensions as well as object points and image points arrays
    nx = 9
    ny = 6
    objPoints = []
    imgPoints = []

    # Prepare object points
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    # Loop through calibration images
    images = glob.glob(cal_path+'calibration*.jpg')
    for fname in images:
        img = cv2.imread(fname)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        # If corners found add them to their corresponding array
        if ret:
            imgPoints.append(corners)
            objPoints.append(objp)

    # If found, draw corners and save
    if ret:
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        cv2.imwrite(output_images+fname.split('/')[-1], img)
        
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, gray.shape[::-1], None, None)
    return mtx, dist

def abs_sobel_thresh(image, orient='x', kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    sobel = cv2.Sobel(image, cv2.CV_64F, orient=='x', orient=='y', ksize=kernel)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def mag_thresh(image, kernel=3, thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel)
    mag = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel = np.uint8(255*mag/np.max(mag))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def dir_threshold(image, kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel)
    direction = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(direction)
    binary_output[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    return binary_output

def color_threshold(image, thresh=(0, 255)):
    # Apply threshold to input image
    s_binary = np.zeros_like(image)
    s_binary[(image >= thresh[0]) & (image <= thresh[1])] = 1
    return s_binary

def clahe(image, clipLimit=4.0, tileGridSize=(12,12)):
    """Apply contrast limited adaptive histogram equalization to a channel.
    Parameters
    ----------
    channel : np.ndarray, ndim=2
        Input representation.
    clipLimit : float
        Limit for contrast clipping.
    tileGridSize : 2-tuple
        Size of the kernel.
    Returns
    -------
    Z : np.ndarray
        The contrast adapted channel.
    """
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(image)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def conv_img(img, conv_kernel):
    conv_kernel = conv_kernel[::-1, ::-1]
    conv = convolve2d(img, conv_kernel, 'same', 'fill', 0)
    return conv

def perspective_transform(src_vertices, dst_vertices):
    src = np.float32(src_vertices)
    dst = np.float32(dst_vertices)
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    return M, M_inv

def window_search(img, min_thresh=3000, verbose=False, img_path=None):
    imgshape = img.shape[::-1]
    window_width = 30 
    window_height = 60
    left_x_centroids = np.array([])
    left_y_centroids = np.array([])
    right_x_centroids = np.array([])
    right_y_centroids = np.array([])
    
    # Check lane lines
    if verbose:
        aux_img = np.dstack(( np.copy(img*255), np.copy(img*255), np.copy(img*255)))
        
    midpoint = int(imgshape[0]/2)
    for y in range(imgshape[1]-1,0,-window_height):
        aux_left_x_centroids = np.array([])
        aux_left_y_centroids = np.array([])
        aux_right_x_centroids = np.array([])
        aux_right_y_centroids = np.array([])
        for x in range(0,midpoint,window_width):
            left_x = midpoint + x
            left_section = img[y-window_height:y,
                          x:x+window_width]
            right_section = img[y-window_height:y,
                          left_x:left_x+window_width]
            left_val = left_section.sum()
            right_val = right_section.sum()
            if left_val > min_thresh:
                left_mass = center_of_mass(left_section == left_section.max())
                left_abs_mass = [x + int(left_mass[1]),
                                 y-window_height+int(left_mass[0])]
                aux_left_x_centroids = np.append(aux_left_x_centroids,
                                                 left_abs_mass[0])
                aux_left_y_centroids = np.append(aux_left_y_centroids,
                                                 left_abs_mass[1])

            if right_val > min_thresh:
                right_mass = center_of_mass(right_section == right_section.max())
                right_abs_mass = [left_x + int(right_mass[1]),
                                 y-window_height+int(right_mass[0])]
                aux_right_x_centroids = np.append(aux_right_x_centroids,
                                                  right_abs_mass[0])
                aux_right_y_centroids = np.append(aux_right_y_centroids,
                                                  right_abs_mass[1])
        
        if len(aux_left_x_centroids) > 0:
            left_median = np.median(aux_left_x_centroids)
            left_criteria = np.zeros_like(aux_left_x_centroids, np.bool_)
            left_criteria[(aux_left_x_centroids > left_median * 0.9) & 
                          (aux_left_x_centroids < left_median * 1.1)] = True
            if left_criteria.any():
                left_x_cent = int(np.mean(aux_left_x_centroids[left_criteria]))
                left_y_cent = int(np.mean(aux_left_y_centroids[left_criteria]))
                left_x_centroids = np.append(left_x_centroids, left_x_cent)
                left_y_centroids = np.append(left_y_centroids, left_y_cent)
                if verbose:
                    cv2.circle(aux_img, (left_x_cent, left_y_cent),
                               6, (255,0,0), -1)
            
        if len(aux_right_x_centroids) > 0:
            right_median = np.median(aux_right_x_centroids)
            right_criteria = np.zeros_like(aux_right_x_centroids, np.bool_)
            right_criteria[(aux_right_x_centroids > right_median * 0.9) & 
                           (aux_right_x_centroids < right_median * 1.1)] = True
            if right_criteria.any():
                right_x_cent = int(np.mean(aux_right_x_centroids[right_criteria]))
                right_y_cent = int(np.mean(aux_right_y_centroids[right_criteria]))
                right_x_centroids = np.append(right_x_centroids, right_x_cent)
                right_y_centroids = np.append(right_y_centroids, right_y_cent)
                if verbose:
                    cv2.circle(aux_img, (right_x_cent, right_y_cent),
                               6, (255,0,0), -1)
            
    if verbose:
        cv2.imwrite(img_path + 'window_search.jpg', aux_img)
        
    return left_x_centroids, left_y_centroids, right_x_centroids, right_y_centroids
        
def guided_window_search(img, line, min_thresh=2000, verbose=False):
    window_width = 200 
    window_height = 60
    x_centroids = np.array([])
    y_centroids = np.array([])
    
    if verbose:
        aux_img = np.dstack(( np.copy(img*255), np.copy(img*255), np.copy(img*255)))
        
    for y in range(img.shape[0]-1,0,-window_height):
        x = np.polyval(line.best_fit, y)
        x_min = int(x-window_width//2)
        x_max = int(x+window_width//2)
        y_min = y-window_height
        y_max = y
        section = img[y_min:y, x_min:x_max]
        val = section.sum()
        if val > min_thresh:
            mass = center_of_mass(section == section.max())
            abs_mass = [x_min + int(mass[1]), y_min + int(mass[0])]
            x_centroids = np.append(x_centroids, abs_mass[0])
            y_centroids = np.append(y_centroids, abs_mass[1])
            if verbose:
                cv2.rectangle(aux_img, (x_min,y_min),
                                     (x_max, y_max), (0,255,0), 2)
                cv2.circle(aux_img, (abs_mass[0], abs_mass[1]),
                   6, (255,0,0), -1)
    
    if verbose:
        for centroid in line.recent_centroids_fitted[-1]:        
            cv2.circle(aux_img, (int(centroid[0]), int(centroid[1])),
                6, (0,0,255), -1)
            
    if verbose:
        plt.imshow(aux_img)
        plt.show()
        
    return x_centroids, y_centroids

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # centroid values of the last n fits of the line
        self.recent_centroids_fitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #recent fits
        self.recent_fits = []
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

def update_line(img, line, x_centroids, y_centroids, line_base_pos, 
                ym_per_pix=30/720, xm_per_pix=3.7/790):
    # Define conversions in x and y from pixels space to meters
    # Assume xm is always 3.7
    # ym_per_pix: meters per pixel in y dimension
    # xm_per_pix: meters per pixel in x dimension
    
    if len(x_centroids) > 2:
        line.detected = True
        if len(line.recent_centroids_fitted) >= 5:
            line.recent_centroids_fitted = line.recent_centroids_fitted[1:]
            
        line.recent_centroids_fitted.append(list(zip(x_centroids,
                                                     y_centroids)))
        line.bestx = np.mean([centroid[0] for values in line.recent_centroids_fitted
                                   for centroid in values])
        line.current_fit = np.polyfit(y_centroids, x_centroids, 2)
        
        if len(line.recent_fits) >= 5:
            line.recent_fits = line.recent_fits[1:]
        line.recent_fits.append(line.current_fit)
        
        if line.best_fit is None:
            line.best_fit = line.current_fit
        else:
            # Find best fit by averaging last fitted coefficients using weights
            # Recent fits will have higher weights
            weights = np.linspace(1, 2, len(line.recent_fits))
            line.best_fit = np.average(line.recent_fits, axis=0, weights=weights)
            #aux_x_centroids = [centroid[0] for values in line.recent_centroids_fitted
            #                  for centroid in values]
            #aux_y_centroids = [centroid[1] for values in line.recent_centroids_fitted
            #                  for centroid in values]
            #line.best_fit = np.polyfit(aux_y_centroids, aux_x_centroids, 2)
            
        ploty = np.linspace(0, img.shape[0]-1)
        plotx = np.polyval(line.best_fit, ploty)
        y_eval = np.max(ploty)
        fit_cr = np.polyfit(ploty*ym_per_pix, plotx*xm_per_pix, 2)
        
        # Calculate the new radii of curvature
        curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
        line.radius_of_curvature = curverad
        line.line_base_pos = line_base_pos
        
    else:
        line.detected = False
        
    return line
