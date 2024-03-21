'''
Generic smoothing and filtering functions 
'''

from scipy.signal import fftconvolve
from Function_files.fitting_functions import gaussian
import numpy as np

def band_pass(N_low, low, N_high, high):
    """
    Generates band-pass windowed sinc for filtering data.

    Parameters
    ----------

    N_low : array_like
        Length of filter
    fc : single value
        frequency cut-off
    
    Returns
    -------

    out : 1-D array
        y values as a function of x

    """

    bpf = fftconvolve(low_pass(N_low, low), high_pass(N_high, high))
    bpf = bpf / np.sum(bpf)

    return bpf

def blackman(N: int):
    """
    Generates blackman window

    Parameters
    ----------

    N : array_like
        Length of window
    
    Returns
    -------

    out : 1-D array
        y values as a function of x

    """
    if not N % 2:
        N += 1

    n = np.arange(N)

    return 0.42 - 0.5 * np.cos(2*np.pi * n / (N-1)) + 0.08 * np.cos(4 * np.pi * n / (N-1))

def high_pass(N, fc):
    """
    Generates high-pass windowed sinc for filtering data below the cut-off
    frequency.

    Parameters
    ----------

    N : array_like
        Length of filter
    fc : single value
        frequency cut-off
    
    Returns
    -------

    out : 1-D array
        y values as a function of x

    """
    if not N % 2:
        N += 1

    high_pass =  blackman(N) * sinc_filter(N, fc)
    high_pass = high_pass / np.sum(high_pass)
    high_pass = -high_pass
    high_pass[(N-1)//2] += 1

    return high_pass

def low_pass(N, fc):
    """
    Generates low-pass windowed since for filtering data above the cut-off
    frequency.

    Parameters
    ----------

    N : array_like
        Length of filter
    fc : single value
        frequency cut-off
    
    Returns
    -------

    out : 1-D array
        y values as a function of x

    """
    if not N % 2:
        N += 1

    low_pass = blackman(N) * sinc_filter(N, fc)

    return low_pass / np.sum(low_pass)

def moving_av(N, type:str='square'):
    """
    Generates moving average window of type

    Parameters
    ----------

    N : array_like
        Length of window
    type : string
        Type of window to use:
        Square
        Gaussian
        Blackman
    
    Returns
    -------

    out : 1-D array
        y values as a function of x

    """
    if type == 'square':
        window = np.ones(N)
    if type == 'gaussian':
        window = gaussian(np.arange(N), 1, 0, N/2, (N-1)/5)
    if type == 'blackman':
        window = blackman(N)

    return window / np.sum(window)

def sinc_filter(N, fc):
    """
    Generates sinc filter with length N and frequency fc

    Parameters
    ----------

    N : array_like
        Length of filter
    fc : single value
        frequency

    Returns
    -------

    out : 1-D array
        y values as a function of x

    """
    if not N % 2:
        N += 1

    n = np.arange(N)

    return np.sinc(2 * fc * (n - (N-1)/2))

def smooth_data(data, N: int=100, type:str='square'):
    """
    Generates sinc filter with length N and frequency fc

    Parameters
    ----------

    data : input array of data to be smoothed    
    N : array_like
        Length of filter
    type : string - name of smoothing function
        Square
        Gaussian
        Blackman
    
    Returns
    -------

    out : 1-D array of data smoothed

    """
    # create a boxcar window and then create a list of smoothed data
    avg_window = moving_av(N, type)
    length_window = avg_window.shape[0] // 2
    
    return fftconvolve(avg_window, data)[length_window-1:-length_window]