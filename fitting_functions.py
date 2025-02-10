'''
Generic fitting functions 
'''

from Function_files.math_functions import zoom
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, fftconvolve

def dbl_exp_decay(x, y_1, y_2, T1, T2, offset):
    """
    Generates approximate T1 decay with given parameters
    
    Parameters
    ----------

    x : 1D array 
        Positional arguments for gaussian
    y_0 : Single value
        Y intercept / start of decay profile
    T1 : Single value
        T1 lifetime of excited state
    T2 : Single value
        Additional dephasing term of excited state
    off : single value
        Offset of the data

    Returns
    -------

    1D array of height values for the positional arguments given in x
    """

    return (y_1 * np.exp(-x/T1)) + (y_2 * np.exp(-x/T2)) + offset

def exp_decay(x, y_0, T1, offset):
    """
    Generates approximate T1 decay with given parameters
    
    Parameters
    ----------

    x : 1D array 
        Positional arguments for gaussian
    y_0 : Single value
        Y intercept / start of decay profile
    T1 : Single value
        T1 lifetime of excited state
    off : single value
        Offset of the data

    Returns
    -------

    1D array of height values for the positional arguments given in x
    """

    return (y_0 * np.exp(-x/T1)) + offset

def find_tau(y: list[float], x: list[float]=[], modifier: float=0.9):
    """
    # TO DO - add functionality for working wih three or more pulses

    Find the difference in time between two pulses and return
    the indexes of these points.

    Uses find_trigger

    Parameters
    ----------

    y : array like
        Pulsed signal
    x : array like
        Time data / data to calculate difference between pulses
    modifier : single value
        Threshold multiplier for trigger (percentage of maximum value of 
        data)
    
    Returns
    -------

    centres : list of int
        Indexes of centres of pulses
    tau     : single value float
        Difference between centres for x data

    """
    indexes = find_trigger(y, modifier, edge='both')
    indexes = np.sort(indexes)
    pulses = len(indexes)//2
    centres = [(indexes[(idx*2)+1] + indexes[idx*2])//2 for idx in range(pulses)]
    if x:
        tau = x[centres[1]] - x[centres[0]]
        return tau, centres
    else:
        return centres
    
def find_trigger(data, modifier: float=0.9, edge: str='rise'):
    """
    Find rising or falling edge of a trigger signal

    Parameters
    ----------

    data : array like
        Trigger signal
    modifier : single value
        Threshold multiplier for trigger (percentage of maximum value of 
        data)
    edge : rise, fall, both
        rise = rising edge
        fall = falling edge
        both = all edges
    
    Returns
    -------

    position : int
        index of trigger point

    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    if modifier:
        threshold = np.max(data) * modifier
    else:
        threshold = np.max(data) * 0.9

    sign = data >= threshold
    search = np.round(fftconvolve(sign, [1, -1]))

    if edge == 'rise':
        position = int(np.min(np.where(search == 1)[0]))
    elif edge == 'fall':
        position = int(np.max(np.where(search == -1)[0]))
    else:
        rising = np.where(search == 1)[0]
        falling = np.where(search == -1)[0]
        position = np.append(rising, falling)
        
    return position

def fit_dbl_exp_decay(x, y, params=None, meth=None, lims=(-np.inf, np.inf)):
    """
    Fits data to an approximate double exponetial decay curve
    
    Parameters
    ----------

    x : 1D array 
        x values of orginal data
    y : 1D array
        y values corresponding to x values
    params : 1D array, optional
        Guess values for T1 decay; y_0, T1
    meth : Single string {'lm', 'trf', 'dogbox'}, optional
        Method to use for optimisation. See 
        scipy.optimize.curve_fit for details
    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on parameters. Defaults to 
        no bounds. 
        See scipy.optimize.curve_fit for details

    Returns
    -------

    fit : 1D array
        Fitted variables
    fit_err : 1D array
        Uncertainty in fitted variables
    """
    fit, success = curve_fit(dbl_exp_decay, x, y, p0=params, method=meth, bounds=lims)
    fit_err = np.sqrt(np.diag(success))

    return fit, fit_err

def fit_exp_decay(x, y, params=None, meth=None, lims=(-np.inf, np.inf)):
    """
    Fits data to an approximate T1 decay curve
    
    Parameters
    ----------

    x : 1D array 
        x values of orginal data
    y : 1D array
        y values corresponding to x values
    params : 1D array, optional
        Guess values for T1 decay; y_0, T1
    meth : Single string {'lm', 'trf', 'dogbox'}, optional
        Method to use for optimisation. See 
        scipy.optimize.curve_fit for details
    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on parameters. Defaults to 
        no bounds. 
        See scipy.optimize.curve_fit for details

    Returns
    -------

    fit : 1D array
        Fitted variables
    fit_err : 1D array
        Uncertainty in fitted variables
    """
    fit, success = curve_fit(exp_decay, x, y, p0=params, method=meth, bounds=lims)
    fit_err = np.sqrt(np.diag(success))

    return fit, fit_err

def fit_gauss(x, amp, params=None, meth=None, lims:tuple=(-np.inf, np.inf)):
    """
    Fits a gaussian function to the data

    Parameters
    ----------

    x : 1D array 
        x values of orginal data
    amp : 1D array
        Amplitude values corresponding to x values
    params : 1D array, optional
        Guess values for Lorentzian function; amp, amp_0, x_0, gamma
    meth : Single string {'lm', 'trf', 'dogbox'}, optional
        Method to use for optimisation. See 
        scipy.optimize.curve_fit for details
    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on parameters. Defaults to 
        no bounds. 
        See scipy.optimize.curve_fit for details

    Returns
    -------

    fit_data : 1D Array
        Fitted variables: height, sigma, mean
    fit_err : 1D Array
        Uncertainty in fitted variables
    """
    fit, success = curve_fit(gaussian, x, amp, p0=params, method=meth, bounds=lims)
    fit_err = np.sqrt(np.diag(success))

    return fit, fit_err
            
def fit_gls(x, amp:float, params=None, meth=None, lims=(-np.inf, np.inf)):
    """
    Fits a Voigt profile to the data using the GLS method
    
    Parameters
    ----------

    x : 1D array 
        x values of orginal data
    amp : 1D array
        Amplitude values corresponding to x values
    params : 1D array, optional
        Guess values for Voigt profile; y_0, amp_g, x_0g, sigma, amp_l, x_0l, gamma, eta
    meth : Single string {'lm', 'trf', 'dogbox'}, optional
        Method to use for optimisation. See 
        scipy.optimize.curve_fit for details
    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on parameters. Defaults to 
        no bounds. 
        See scipy.optimize.curve_fit for details

    Returns
    -------

    fit : 1D array
        Fitted variables
    fit_err : 1D array
        Uncertainty in fitted variables
    """
    fit, success = curve_fit(pseudo_voigt, x, amp, p0=params, method=meth, bounds=lims)
    fit_err = np.sqrt(np.diag(success))

    return fit, fit_err

def fit_lorentz(x, y, params=None, meth=None, lims:tuple=(-np.inf, np.inf)):
    """
    Fits a Lorentzian function to the data
    
    Parameters
    ----------

    x : 1D array 
        x values of orginal data
    y : 1D array
        Amplitude values corresponding to x values
    params : 1D array, optional
        Guess values for Lorentzian function; amp_0, x_0, gamma
    meth : Single string {'lm', 'trf', 'dogbox'}, optional
        Method to use for optimisation. See 
        scipy.optimize.curve_fit for details
    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on parameters. Defaults to 
        no bounds. 
        See scipy.optimize.curve_fit for details

    Returns
    -------

    fit : 1D array
        Fitted variables
    fit_err : 1D array
        Uncertainty in fitted variables
    """
    fit, success = curve_fit(lorentzian, x, y, p0=params, method=meth, bounds=lims)
    fit_err = np.sqrt(np.diag(success))

    return fit, fit_err

def fit_Ngauss(x, y, params=None, meth=None, lims:tuple=(-np.inf, np.inf)):
    """
    Fits N number of Gaussian to the data
    
    Parameters
    ----------

    x : 1D array 
        x values of orginal data
    y : 1D array
        y values corresponding to x values
    params : 1D array, optional
        Guess values for gaussians in list: amp, y_0, x_0, sigma
    meth : Single string {'lm', 'trf', 'dogbox'}, optional
        Method to use for optimisation. See 
        scipy.optimize.curve_fit for details
    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on parameters. Defaults to 
        no bounds. 
        See scipy.optimize.curve_fit for details

    Returns
    -------

    fit : 1D array
        Fitted variables
    fit_err : 1D array
        Uncertainty in fitted variables
    """
    fit, success = curve_fit(N_gaussian, x, y, p0=params, method=meth, bounds=lims)
    fit_err = np.sqrt(np.diag(success))

    return fit, fit_err

def fit_rise_time(x, y, params=None, meth=None, lims=(-np.inf, np.inf)):
    """
    Fits a straight line function to the data to 
    
    Parameters
    ----------

    x : 1D array 
        x values of orginal data
    y : 1D array
        y values corresponding to x values
    params : 1D array, optional
        Guess values for straight line; a, b
    meth : Single string {'lm', 'trf', 'dogbox'}, optional
        Method to use for optimisation. See 
        scipy.optimize.curve_fit for details
    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on parameters. Defaults to 
        no bounds. 
        See scipy.optimize.curve_fit for details

    Returns
    -------

    fit : 1D array
        Fitted variables
    fit_err : 1D array
        Uncertainty in fitted variables
        
    """
    fit, success = curve_fit(rising_edge, x, y, p0=params, method=meth, bounds=lims)
    fit_err = np.sqrt(np.diag(success))

    return fit, fit_err

def fit_straight(x, y, params=None, meth=None, lims=(-np.inf, np.inf)):
    """
    Fits a straight line function to the data to 
    
    Parameters
    ----------

    x : 1D array 
        x values of orginal data
    y : 1D array
        y values corresponding to x values
    params : 1D array, optional
        Guess values for straight line; a, b
    meth : Single string {'lm', 'trf', 'dogbox'}, optional
        Method to use for optimisation. See 
        scipy.optimize.curve_fit for details
    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on parameters. Defaults to 
        no bounds. 
        See scipy.optimize.curve_fit for details

    Returns
    -------

    fit : 1D array
        Fitted variables
    fit_err : 1D array
        Uncertainty in fitted variables
    """
    fit, success = curve_fit(straight, x, y, p0=params, method=meth, bounds=lims)
    fit_err = np.sqrt(np.diag(success))

    return fit, fit_err

def gaussian(x, amp:float, y_0:float, x_0:float, sigma:float):
    """
    Generates Gaussian with given parameters
    
    Parameters
    ----------

    x : 1D array 
        Positional arguments for gaussian
    amp : Single value
        Maximum value of gaussian
    y_0 : Single value
        Y Offset
    x_0 : Single value
        Centre of Gaussian peak
    sigma : Single value
        Standard deviation of Gaussian

    Returns
    -------

    1D array of height values for the positional arguments given in x
    """

    return amp * np.exp(-((x - x_0) ** 2) / (2 * sigma ** 2)) + y_0

def lorentzian(x, amp:float, y_0:float, x_0:float, gamma:float):
    """
    Generates Lorentzian function with given parameters.

    Parameters
    ----------

    x : array_like
        Input range of frequencies
    amp : single value
        Height of peak
    y_0 : Single value
        Y offset
    x_0 : single value
        Central frequency of Lorentzian
    gamma : single value
        FWHM of Lorentzian
    
    Returns
    -------

    out : 1-D array
        Output amplitudes as function of x

    """
    return (amp * ((0.5*gamma)**2/((x-x_0)**2 + (0.5*gamma)**2))) + y_0

def N_gaussian(x, *params):
    """
    Generates sum of N Gaussians with given parameters
    
    Parameters
    ----------

    x : 1D array 
        Positional arguments for Gaussian
        
    params : list of values containing (in order) 

    amp : Single value
        Maximum value of gaussian
    y_0 : Single value
        Y Offset
    x_0 : Single value
        Centre of Gaussian peak
    sigma : Single value
        Standard deviation of Gaussian

    Returns
    -------

    1D array of height values for the positional arguments given in x

    """
    assert len(params) % 4 == 0, 'params must be a multiple of 4'
    y = np.zeros_like(x)
    for i in range(0, len(params), 4):
        y = y + gaussian(x, params[i], params[i+1], params[i+2], params[i+3])

    return y

def N_lorentzian(x, *params):
    """
    Generates sum of N Lorentzians with given parameters
    
    Parameters
    ----------

    x : 1D array 
        Positional arguments for Lorentzian
        
    params : list of values containing (in order) 

    amp : Single value
        Maximum value of Lorentzian
    y_0 : Single value
        Y Offset
    x_0 : Single value
        Centre of Lorentzian peak
    gamma : Single value
        Standard deviation of Lorentzian

    Returns
    -------

    1D array of height values for the positional arguments given in x

    """
    assert len(params) % 4 == 0, 'params must be a multiple of 4'
    y = np.zeros_like(x)
    for i in range(0, len(params), 4):
        y = y + lorentzian(x, params[i], params[i+1], params[i+2], params[i+3])

    return y

def peak_find(y, x=None, top_tol=None, dist=None, prom_tol=None, lims=None):
    """
    Find peaks in data

    Parameters
    ----------

    y : array to find peaks
    x : corresponding x-values
    args : peak finding conditions (see scipy.signal.find_peaks)

    Returns
    -------

    peaks_data : indexes of peaks for each data set

    """
    lower = 0
    upper = None

    if lims:
        lower, upper = zoom(x, bounds=lims)

    data_max = np.max(y[lower:upper])

    if prom_tol:
        prom_tol *= data_max
    if top_tol:
        top_tol *= data_max

    peaks, _ = find_peaks(y[lower:upper], height=top_tol, distance=dist, prominence=prom_tol)
    
    return peaks

def pseudo_voigt(x:list[float], y_0:float, amp_g:float, x_0g:float,
                sigma:float, amp_l:float, x_0l:float, gamma:float, eta:float):
    """
    Generates Pseudo Voigt profile with given parameters using GLS method.

    Parameters
    ----------

    x : array_like
        Input range of frequencies
    y_0 : single value
        Profile Y offset
    amp_g : single value
        Amplitude of Gaussian component
    x_0g : single value
        Central frequency of Gaussian
    sigma : single value
        Standard deviation of Gaussian
    amp_l : single value
        Amplitude of Lorentzian component
    x_0l : single value
        Central frequency of Lorentzian
    Gamma : 
        FWHM of Lorentzian
    Eta : Single Value
        Weighting factor of Gaussian to Lorentzian

    Returns
    -------

    out : 1-D array
        Output amplitudes as function of x

    """
    pv = (eta * amp_g * (np.exp(-((x - x_0g) ** 2) / (2 * sigma ** 2)))) + ((1-eta)
        * amp_l * ((0.5*gamma)**2/((x-x_0l)**2 + (0.5*gamma)**2))) + y_0

    return pv

def rising_edge(t, amp, t_0, t_r):
    """
    Generates a rising edge with the given parameters.

    Parameters
    ----------

    t : array_like
        Input range of time
    amp : single value
        Maximum value of rise
    t_0 : Single value
        time offset
    t_r : single value
        rise time
    
    Returns
    -------

    out : 1-D array
        Output amplitudes as function of t 

    """
    return amp * (1 - np.exp(-(t-t_0)/t_r))

def straight(x, a, b):
    """
    Generates straight line function with given parameters.

    Parameters
    ----------

    x : array_like
        Input range of x values
    a : single value
        gradient or slope of line
    b : single value
        Y-interecept value
    
    Returns
    -------

    out : 1-D array
        y values as a function of x

    """
    return a*x + b

def print_T1_fit(fit_data):
    """
    Print out the values extracted from a T1 fit

    <fit_data>:
        tuple of fit and fit error
    
    """
    print('T1 Fit Params:')
    print("I0 : {0:2.2f} mV, T1 : {1:2.2f} us, Y0 : {2:2.2f} mV \n".format(fit_data[0][0]*1E3, fit_data[0][1]*1E6, fit_data[0][2]*1E3))
    print('Fit Error data:')
    print("dI0 : {0:2.2f} mV, dT1 : {1:2.3f} us, dY0 : {2:2.2f} mV \n".format(fit_data[1][0]*1E3,fit_data[1][1]*1E6, fit_data[1][2]*1E3))