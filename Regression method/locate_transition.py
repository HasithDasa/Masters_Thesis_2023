# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 08:28:43 2023

@author: Jakob Dieckmann
@email: j.dieckmann@bimaq.de

@description: This python script locates the laminar turbulent transition with 
    state of the art methods first implemented by Dollinger-Gleichauf.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.ndimage import gaussian_filter
from preprocessing import preprocess, plot_np_image, find_edges, assign_edges, remove_dead_on_blade, plot_np_image_with_edges, add_noise
from scipy.optimize import curve_fit
from scipy.special import erf, erfinv
import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings
# warnings.filterwarnings("error")



"""  Parameters from the program are stored here """

params = {
    
    "folder_path"       : r"D:\Academic\MSc\Thesis\Project files\Project Complete\data\new data\npy\copied_images\New folder\corotating",
    "file_name"         : "irdata_0021_0440.npy",
    "start_x_position"  : 200, # Start des Bereiches in dem Transition bestimmt wird
    "stripe_width"      : 100, # Breite des Bereiches in dem Transition bestimmt wird
    "smooth"            : 0, #If zero then no smoothing is applied
    "loc_method"        : "intensity", # this is "intensity" or "intensity" gradient
    "noise_std"         : 0, #If zero then no noise is applied
    "inf"               : 1e-3,
    }

def main():
    file_str = os.path.join(params['folder_path'], params['file_name'])
    img_pre = np.load(file_str)
    img = preprocess(img_pre, params)
    img = add_noise(img, params=params)
     
    # locate_transitions(img)
    edges = find_edges(img)
    plot_np_image_with_edges(img, edges)
    le, te, angle = assign_edges(edges)  
    params["le"], params["te"] = int(np.min(le)), int(np.max(te))    
    ioc = extract_intensity(img, params=params)
    ioc_trans, par_err, ax = locate_transition(ioc, params=params)  # par_err parameter for the error function 
    mu_pos_trans = np.average(ioc_trans) + params["te"] # without this addition it is reference to the trailing edge
    # mu_pos_trans = params["te"]
    std_pos_trans = np.std(ioc_trans)
    CNR = calc_CNR(img, ioc_trans, params=params)
    
    # print transition location
    plot_image_and_transition(img, pos=mu_pos_trans)
    
    
    print("The transition is determined at {} pixels from the top".format(mu_pos_trans))
    print("The transition is determined at {} pixels from the trailing edge".format(np.average(ioc_trans)))
    print("The standard deviation of the result is {} ".format(std_pos_trans))
    print("The CNR of the image is {} .".format(CNR))
    
    
def locate_transition(ioc, params=params):
    """ This function locates the transition """
    
    method=params["loc_method"]
    
    #Smoothen for further analysis
    ioc_not_smooth = ioc.copy()

    """This line conditionally applies a Gaussian filter to the ioc data if the smooth parameter in the params dictionary is not zero. 
    If smooth is zero, the original ioc remains unchanged."""
    ioc = gaussian_filter(ioc, params["smooth"]) if params["smooth"] else ioc
    
    # # Calculate gradient of intensity
    # try:
    #     ioc_grad = np.gradient(f=ioc, axis=1)
    # except:
    #     sys.exit()
    
    # Plot the function that is used for analysis
    if params["loc_method"] == "intensity":
        fig, ax = plot_int_chord(ioc[-1, :], title="The intensity of last column over chord length")
    # else:
    #     fig, ax = plot_int_chord(ioc_grad[-1, :], title="The intensity gradient of last column over chord length")

    # We are plotting to a curve
    # independent variable: position on chord
    # dependend variable:   intensity (intensity gradient)
    st = ioc.shape[0]
    ioc_trans = np.zeros(shape=(st))
    par_err = np.zeros(shape=(st, 4)) # parameters of the error function
    xdata = np.arange(ioc.shape[1]) # Chord length
    start = int(0.7 * (params["le"] - params["te"]))
    for i, (int_col, grad_col) in enumerate(zip(ioc, ioc)):  # I changed here because ioc_grad can't be calculated in some cases "(ioc, ioc_grad)"
        try:
            bounds= ((0, 0, 0, int(np.min(int_col))), (10, int(int_col.shape[0]), 25, np.ceil(np.max(int_col))))
            if params["loc_method"] == "intensity":
                popt, pcov = curve_fit(CDF, xdata=xdata, ydata=int_col, p0=[1, start, 6, bounds[0][3]], maxfev=10000, bounds=bounds) # p0 = [a, mu, sigma, y_off]
            else:
                popt, pcov = curve_fit(PDF, xdata=xdata, ydata=grad_col, p0=[1, start, 6, bounds[0][3]], maxfev=10000, bounds=bounds) # p0 = [a, mu, sigma, nonsense]
            ioc_trans[i] = popt[1]
            par_err[i] = popt
        except:
            print("The curve fit didn't work. Please change the settings. Increase smoothing. Systems exit. I will try to cheat here! Be careful!")
            print("The data I was working with params \n {}, and int_col \n {} and i \n {}".format(params, int_col, i))
            ioc_trans[i] = ioc_trans[i-1]
            par_err[i] = par_err[i-1]
            popt = par_err[i-1]
            
    if params["loc_method"] == "intensity":
        plot_xxx_function(ax, "cdf", popt[0], popt[1], popt[2], popt[3])
    else:
        plot_xxx_function(ax, "pdf", popt[0], popt[1], popt[2])
    return ioc_trans, par_err, ax

def preparation_of_image(img, params=params):
    img = preprocess(img, params)
    edges = find_edges(img)
    le, te, angle = assign_edges(edges)  
    remove_dead_on_blade(img, te, le)
    params["le"], params["te"] = int(np.min(le)), int(np.max(te))
    ioc = extract_intensity(img, params=params)
    return img, ioc, le, te

def preparation_of_sim(img, params=params):
    edges = find_edges(img)
    le, te, angle = assign_edges(edges) 
    params["le"], params["te"] = int(np.min(le)), int(np.max(te))
    ioc = extract_intensity(img, params=params)
    return img, ioc, le, te

    
def extract_intensity(img, params=params):
    # Cuts to observed area only
    # For only one column this columns is taken and params is ignored
    x_pos = params["start_x_position"] if img.shape[1] > 1 else 0
    st = params["stripe_width"] if img.shape[1] > 1 else 1
    ioc = np.transpose(img[params["te"]+1:params["le"], x_pos:x_pos+st])
    return ioc
    

def calc_CNR(img, ioc_trans, params=params):
    transition = np.average(ioc_trans) + params["te"]
    ## Calculates the CNR with the formula from Felix Dissertation
    lam_pos = params["le"] - round((params["le"] - transition) / 2 )
    tur_pos = params["te"] + round(( transition - params["te"]) / 2)
    h_stripe = 5 # stripe height ist 2*h_stripe
    tur = img[tur_pos-h_stripe: tur_pos+h_stripe, params["start_x_position"]: params["start_x_position"]+ params["stripe_width"]]    
    lam = img[lam_pos-h_stripe: lam_pos+h_stripe, params["start_x_position"]: params["start_x_position"]+ params["stripe_width"]] 
    
    CNR = np.abs(np.mean(tur) - np.mean(lam)) / np.sqrt(np.std(tur)**2 + np.std(lam)**2)
    
    return CNR
 
def plot_int_chord(int_over_chord, int_over_chord2=None, title="The temparature along the chord"):
    """ This plots a curve with y-intensity and x chord position """

    fig, ax = plt.subplots()
    fig.suptitle(title + ", created: {}".format(datetime.now().strftime("%H:%M:%S"))) 
    ax.plot(np.arange(len(int_over_chord)), int_over_chord, label="Measured {} at x={} pixels" \
        .format(params["loc_method"], params["start_x_position"] + params["stripe_width"]))
    if int_over_chord2 is not None: ax.plot(np.arange(len(int_over_chord2)), int_over_chord2, color="green" )      
    ax.set_xlabel("chord length in pixel")
    ax.set_ylabel(params["loc_method"])
    return fig, ax
    
# Guassian cumulative distribution function
def CDF(c, a, mu, sigma, y_off):
    """ This function provides the cumulative distribution function that is 
        a scaled gaussian error function  (erf) """
        
    # Scaling variable  : a # 
    # µ                 : mu 
    # σ                 : sigma
    # y-offset          : y_off
    I = (1/2) * a * erf((c - mu)/(np.sqrt(2)*sigma)) + y_off     
    return I

def inv_CDF(c, a, I, sigma, y_off):
    if isinstance(I, np.ndarray):
        # if I is a NumPy array, create a new array of the same shape
        A = np.zeros_like(I)
        for i, Ii in enumerate(I):
            term = 2 * (Ii - y_off) / a 
            term = np.clip(term, -0.9999, 0.9999, out=term)
            A[i] = c - np.sqrt(2) * sigma * erfinv(term)
    else:
        # if I is a scalar, calculate the quantile directly
        term = 2 * (I - y_off) / a 
        term = max(-0.999999999, min(0.999999999, term))
        # term = np.clip(term, -0.9999, 0.9999, out=term)
        A = c - np.sqrt(2) * sigma * erfinv(term)
    return A



# Guassian distribution function
def PDF(x, a, mu, sigma, nonsense):
    """ This function provides Probabilistic gaussion function. "The bell curve" """
    
    return a*np.exp(-(x-mu)**2/(2*sigma**2))


def plot_xxx_function(ax = None, xxx="cdf", a=1, mu=100, sigma=20, y_off=300):
    """ Helpfunction: plots the pdf """
    
    if ax is None:
        raise TypeError("Please specify the axis to plot on")
    x = np.linspace(0, ax.get_xlim()[1] - 7, 1000) # Define a range of x values
    # y = PDF(x, a, mu, sigma)  
    if xxx == "cdf":
        y = CDF(x, a, mu, sigma, y_off)   # Evaluate the function at each x value
    else:
        y = PDF(x, a, mu, sigma)
    ax.plot(x, y, label='Plot of the ' + xxx + " function.")     # Plot x versus y
    ax.legend()
    plt.show()                # Display the plot
    
def plot_image_and_transition(img, pos, title="Image and Transition"):
    img_print = img
    img_print[ int(pos) , ::20] = 1
    plot_np_image(img_print, title=title)
    return None
    
    

if __name__ == "__main__":
    main()
