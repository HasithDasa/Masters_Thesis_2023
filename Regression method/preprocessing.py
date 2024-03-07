# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 13:34:30 2023

@author: Jakob Dieckmann
@email: j.dieckmann@bimaq.de

@description: I want to preprocess my images for further work with it.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
import cv2
from scipy.ndimage import rotate, median_filter
from datetime import datetime 
from scipy import stats
import argparse
from jenkspy import JenksNaturalBreaks
from scipy.stats import linregress
from skimage.restoration import inpaint



"""
My prefered structure is that parameters
  that I need are defined in this variable 
    
"""

params = {
    "canny_min_length"  :  5, 
    "folder_path"       : r"D:\Academic\MSc\Thesis\Project files\Project Complete\data\new data\npy\copied_images\New folder\corotating",
    "file_name"         : "irdata_0018_0239.npy",
    # "folder_path"       : r"C:\Users\jdi\Documents\Seafile\KleineDaten",
    "min_arc_length"    : 100,
    "noise_std"         : 0, 
    "data_date"         : "231508"
    }
    

def main():
    """ The program is executed from here """
    
    file_str = os.path.join(params['folder_path'], params['file_name'])
    img_pre = np.load(file_str)
    img = preprocess(img_pre, params=params)
    img = add_noise(img, params=params)
    plot_np_image(img)
    return None 

def preprocess(img_pre, params=params):
    """ Here the image is preprocessed, including outlier removel and derotation"""
    
    img = img_pre.copy()
    img = campain_special(img)
    hist, bin_edges = plot_histogram(img)
    img = remove_background(img, hist, bin_edges)
    # img = remove_outliers(img)
    edges = find_edges(img)
    le, te, angle = assign_edges(edges)
    # plot_np_image_with_edges(img, edges)
    img = rotate_and_crop(img, angle)
    img = remove_background(img, hist, bin_edges)
    # plot_np_image(img, "This is the preprocessed image") 

    # img = add_noise(img, params=params)
    return img

def add_noise(img, params=params):
    
    ## Adding noise
    if params["noise_std"]:
        non_zero_mask = img != 0
        noise = np.random.normal(0, params["noise_std"], img.shape)
        img[non_zero_mask] += noise[non_zero_mask]
        # plot_np_image(img, "Noised image")
    return img

def remove_dead_on_blade(img, te, le):
    for i, col in enumerate(img.T):
        active = col[int(te[i]): int(le[i])]
        active[active==0] = np.mean(active[active != 0])
    return img


def remove_dead_pixels(img):
    """This remove zero valued pixels if all other pixels around have a value"""
    cus_filter = np.array([[1, 1, 1], 
                           [1, 9, 1],
                           [1, 1, 1]])
    """Here, a custom filter (cus_filter) is defined. This filter seems designed for a convolution operation.
    The center value (9) being distinctively larger suggests that this filter 
    might be used to detect pixels that stand out from their neighbors."""

    # all non-zero values in the original img are set to 1, while zero values remain zero.
    bin_img = np.where(img != 0, 1, img)
    filt = cv2.filter2D(bin_img, -1, cus_filter)
    # Find the coordinates where filt equals 8
    row, col = np.where(filt == 8)
    
    img_no_dead = img.copy()
    """For every pixel identified as a 'dead pixel' (those surrounded by non-zero values), 
    its value is replaced by the average of the pixel immediately above and the pixel immediately below it. """
    img_no_dead[row, col] = (img_no_dead[row - 1, col] + img_no_dead[row+1, col] ) /2
    return img_no_dead

def campain_special(img, params=params):
    # img = np.rot90(img, 2)
    img = np.flipud(img)
    img = img[:, 0:500] # The measurement was very close to the tower

    return img


def assign_edges(edges):
    slopes = np.zeros(edges.shape[0])
    std_errors = np.zeros(edges.shape[0])
    """For each edge, a linear regression is performed using the linregress function from the scipy.stats module. 
    The independent variable x is simply an array of indices, while the dependent variable y is the edge's values."""
    for i, edge in enumerate(edges):
        slopes[i], intercept, r_value, p_value, std_errors[i] = linregress(x=np.arange(edge.shape[0]), y=edge)

    """slopes[i]: The slope of the regression line, stored in the previously initialized slopes array. 
    std_errors[i]: The standard error of the estimate, stored in the previously initialized std_errors array."""

    """Based on the standard errors computed for each edge's regression fit, the edge with the minimum standard error is assigned as the leading edge (le), 
    and the one with the maximum standard error is assigned as the trailing edge (te)."""
    le = edges[np.argmin(std_errors)]
    te = edges[np.argmax(std_errors)]
    """The angle of the leading edge (le) is calculated using the arctan function. 
    This gives the angle between the leading edge and the horizontal axis"""
    angle = np.arctan(slopes[np.argmin(std_errors)])
    le -= 10 # avoids effects at the edges of rotor blade
    te += 15

    return le, te, np.rad2deg(angle) 

def find_edges(img):
  
    contours = find_contours(img)
    # print("We have found {} contours in the derotated".format(len(contours)))
    # plot_np_image_with_contours(img, contours)
    
    ## The next part is only for formatting contours as a readable numpy array
    edges = np.zeros((len(contours), img.shape[1]))
    for k, edge in enumerate(contours):

        edge = edge.reshape(-1, 2)

        _, imp_indice = np.unique(edge[:, 0], return_index =True)
        edge = edge[imp_indice] # consits of unique edge coordinates

        
        width = img.shape[1] - 1
        if edge[-1, 0] != width: #last value
            y = edge[-1, 1]
            edge = np.append(edge, [[width, y]], axis = 0)
        for i in np.arange(img.shape[1]): # all the other values
            if edge[i,0] != i:
                edge = np.insert(arr=edge, obj=i, values=[i, edge[i,1]], axis=0)
        edges[k] = edge[:, 1]
    
    ## Reduce to two contours 
    con_means = np.array([np.mean(edge) for edge in edges])
    jnb = JenksNaturalBreaks(2) # Asking for 2 clusters
    jnb.fit(con_means)
    
    edge1 = np.stack(edges[con_means>jnb.inner_breaks_])
    edge2 = np.stack(edges[con_means<=jnb.inner_breaks_])
    edge1 = np.mean(edge1, axis = 0)
    edge2 = np.mean(edge2, axis = 0)
    edges = np.stack((edge1, edge2), axis= 0)

    # print("edges", edges)
    return edges


    # ## Reduce to to two contours 
    # con_means = np.array([np.mean(con[:, 0, 1]) for con in contours])
    # jnb = JenksNaturalBreaks(2) # Asking for 4 clusters
    # jnb.fit(con_means)
 
    
def remove_background(img, hist, bin_edges):
    # This removes the background. The function is derived from looking at the histogramm.
    # It assumes that in the histogram there are two bubbles. One from the background and one from the content. 
    # It is important to note that I want to drastically remove, because I rather have the edges cut a little stricter.
    # This makes my algorithms more robust, because there are not artefacts from the edges
    
    # Example: 000012345433000014567123400000 should be cut in a way that only the second hill between 4 and 4 stays
    
    
    threshold = img.shape[0] * img.shape[1] / hist.shape[0] / 3
    
    # First removal of bad stuff
    hist[hist < threshold] = 0
    for i in np.arange(2, hist.shape[0]-2):
        if np.any(hist[i-2:i] == 0) and np.any(hist[i+1:i+2+1] == 0):
            hist[i] = 0
            
    # Find interesting part
    # Assumption: interesting part is warmer
    # look where interesting starts 
    i = hist.shape[0] - 20 
    while np.all(hist[i:i+20] < threshold * 6):
        i -= 1    
    hist[i+1:] = 0
    
    # look how long interesting part goes
    while np.any(hist[i-19: i+1] != 0):
        i-= 1
    
    # cut the edge a little tighter
    while hist[i] < threshold:
        i += 1
             
    hist[:i] = 0
    
    lower_lim = bin_edges[np.min(np.nonzero(hist))]
    upper_lim = bin_edges[np.max(np.nonzero(hist))+1]
    # img[(img<lower_lim) | (img>upper_lim)] = 0
    img[(img<lower_lim)] = 0
    img[(img>upper_lim)] = upper_lim
    
    # plot_np_image(img, title="Background removed")
    
    img = remove_dead_pixels(img)
    
    return img    
    
def remove_outliers(T_std):
# Taken from Dollinger
# 1. Erstellen einer Maske, in der alle Pixel außerhalb eines definierten 
#    Bildbereichs schwarz, der restliche Bildbereich weiß dargestellt wird
# 2. Im originalbild werden die defekten Bildbereiche weiß gefärbt
# 3. 'Bildreperatur' der definierten Pixel unter Anwendung von inpaint.inpaint_biharmonic    
    
    T_std_mask = T_std.copy()
    threshold = [np.mean(T_std) - 3 * np.std(T_std), np.mean(T_std) + 3 * np.std(T_std)]
    T_std_mask = np.where(np.bitwise_or(T_std_mask <= threshold[0], T_std_mask >= threshold[1]), 1, 0)
    T_std_defect = T_std.copy()
    
    for layer in range(T_std_defect.shape[-1]):
        T_std_defect[np.where(T_std_mask == 1)] = 0
        
    T_std_clean = inpaint.inpaint_biharmonic(T_std_defect, T_std_mask, multichannel=False)
    
    return T_std_clean
    

def rotate_and_crop(image, angle):
    # Rotate the image
    rotated_image = rotate(image, angle, reshape=False, mode='constant', order=1)
    wr, hr = rotatedRectWithMaxArea(image.shape[1], image.shape[0], np.deg2rad(angle))
    h_offset = int((rotated_image.shape[0] - hr) /2)
    w_offset = int((rotated_image.shape[1] - wr) /2)

    print("h_offset:", h_offset)
    print("h_offset_:", h_offset+int(hr))
    print("w_offset:", w_offset)
    print("w_offset:", w_offset+int(wr))

    cropped_rotated_image = rotated_image[h_offset:h_offset+int(hr), w_offset:w_offset+int(wr)]
    
    ## make sure that there are borders in y direction
    cropped_rotated_image[:2, :] = 0
    cropped_rotated_image[-2:, :] = 0
    
    return cropped_rotated_image

def rotatedRectWithMaxArea(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
      return 0,0
      
    width_is_longer = w >= h
    side_long, side_short = (w,h) if width_is_longer else (h,w)
      
    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
      # half constrained case: two crop corners touch the longer side,
      #   the other two corners are on the mid-line parallel to the longer line
      x = 0.5*side_short
      wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
    else:
      # fully constrained case: crop touches all 4 sides
      cos_2a = cos_a*cos_a - sin_a*sin_a
      wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a
      
    return wr,hr

def plot_histogram(img):
    
    hist, bin_edges = np.histogram(img, bins=1000)
    
    fig, ax = plt.subplots()
    ax.bar(bin_edges[1:], hist)
    # Plot the histogram
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram' + ", created: {}".format(datetime.now().strftime("%H:%M:%S")))
    plt.show()
    return hist, bin_edges
  
def find_contours(img):
    """ Plot all the lines """ 
    
    median_img = median_filter(img, size=5)
    edges = cv2.Canny(median_img.astype(np.uint8), 22, 30)
    
    # Find the contours in the edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out small contours
    filtered_contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) >= params["min_arc_length"]]
    
    return filtered_contours
        
def plot_np_image(img, title="Figure X"):
    """ This function plots an numpy image with title image """
    
    fig, ax = plt.subplots()
    im = ax.imshow(img, cmap='gray', vmin=240, vmax=290)
    fig.suptitle(title + ", created: {}".format(datetime.now().strftime("%H:%M:%S")))
    ax.set_xlabel("x position on rotor blade")
    ax.set_ylabel("y position on rotor blade")
    # Create a colorbar object
    cbar = fig.colorbar( im, ax=ax, label="The intensity")
    
    # Customize the colorbar
    cbar.ax.set_ylabel('Colorbar Label', fontsize=12)
    cbar.ax.tick_params(labelsize=10)  
    
    plt.show()
    return fig, ax

def plot_np_image_with_contours(img, contours, title="with contours"):
    """ This function plots an numpy image with title image and shows the detected contours """
     
    contour_img = img.copy()
    cv2.drawContours(contour_img, contours, -1, img.max()+5, 2)
    
    fig, ax = plt.subplots()
    im = ax.imshow(contour_img, vmin=290, vmax=img.max())
    fig.suptitle(title + ", created: {}".format(datetime.now().strftime("%H:%M:%S")))
    ax.set_xlabel("x position on rotor blade")
    ax.set_ylabel("y position on rotor blade")
    # Create a colorbar object
    cbar = fig.colorbar( im, ax=ax, label="The intensity")
    
    # Customize the colorbar
    cbar.ax.set_ylabel('Colorbar Label', fontsize=12)
    cbar.ax.tick_params(labelsize=10)  
    plt.show()
    
    return fig, ax

def plot_np_image_with_edges(img, edges, title="with edges"):
    """ This function plots an numpy image with title image and shows the detected contours """
    
    edge_img = img.copy()
    edge_val = np.max(edge_img) + 10
    for edge in edges:
        for x, y in enumerate(edge):
            edge_img[int(y)-2: int(y)+3, x] = edge_val
    
    fig, ax = plt.subplots()
    im = ax.imshow(edge_img, vmin=290, vmax=edge_img.max())
    fig.suptitle(title + ", created: {}".format(datetime.now().strftime("%H:%M:%S")))
    ax.set_xlabel("x position on rotor blade")
    ax.set_ylabel("y position on rotor blade")
    # Create a colorbar object
    cbar = fig.colorbar( im, ax=ax, label="The intensity")
    
    # Customize the colorbar
    cbar.ax.set_ylabel('Colorbar Label', fontsize=12)
    cbar.ax.tick_params(labelsize=10)  
    plt.show()
    
    return fig, ax
    
def draw_rectangle_on_image(img, rect):
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    
    # Draw the minimum area rectangle on the original image
    cv2.drawContours(img,[box],0,(0,0,255),2)
    
    return img

# def derotate(img, params):
    
#     """ This function recognices the rotor blade and derotates it."""
    
#     img_ori = img.copy()
#     contours = find_contours(img)
#     print("We have found {} contours.".format(len(contours)))
#     contur_img = cv2.drawContours(img, contours, -1, color=(255,255,255), thickness=5)
    
#     # Get the minimum area rectangle that contains all the contours
#     rect = cv2.minAreaRect(np.concatenate(contours))
#     draw_rectangle_on_image(img, rect)
    
#     img = rotate_and_crop(img, angle=rect[2])
    
#     # img = rotate(img_ori, angle=rect[2], mode="constant")
    
#     # If rotor blade is vertically
#     if np.mean(np.std(img, axis=0)) < np.mean(np.std(img, axis=1)):
#         img = np.rot90(img, k=1)
    
#     return img

if __name__ == "__main__":
    main()
    
