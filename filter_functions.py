'''
Generic smoothing and filtering functions 
'''

from scipy.signal import fftconvolve
from Function_files.fitting_functions import gaussian
import numpy as np

def band_pass(N, low, high):
    """
    Generates a band-pass window sinc for filtering data 
    with frequency cut-offs at low and high

    Parameters
    ----------
    N : int
        Length of filter
    low :
        Low frequency cut-off
    high : 
        High frequency cut-off
    
    Returns
    -------
    bpf : array
        Normalised band-pass function of length N

    """

    bpf = fftconvolve(low_pass(N, low), high_pass(N, high))
    bpf = bpf / np.sum(bpf)

    return bpf

def blackman(N: int):
    """
    Generates blackman window

    Parameters
    ----------
    N : int
        Length of window
    
    Returns
    -------
    out : array
        Normalised Blackman window of length N

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
    N : int
        Length of filter
    fc : int
        Frequency cut-off
    
    Returns
    -------
    out : array
        Normalised high-pass windowed sinc filter of length N

    """
    # N must be even
    if not N % 2:
        N += 1

    high_pass =  blackman(N) * sinc_filter(N, fc)
    high_pass = high_pass / np.sum(high_pass)
    high_pass = -high_pass
    high_pass[(N-1)//2] += 1

    return high_pass

def low_pass(N, fc):
    """
    Generates low-pass windowed sinc for filtering data above the cut-off
    frequency.

    Parameters
    ----------
    N : int
        Length of filter
    fc : int
        Frequency cut-off
    
    Returns
    -------
    out : array
        Normalised low-pass windowed sinc filter of length N

    """
    if not N % 2:
        N += 1

    low_pass = blackman(N) * sinc_filter(N, fc)

    return low_pass / np.sum(low_pass)

def create_window(N, mode:str='square'):
    """
    Generates window for smoothing depending on user input

    Parameters
    ----------
    N : int
        Length of window
    mode : string
        Function to use for window > square, gaussian, blackman
    
    Returns
    -------
    out : array
        Normalised smoothing window of length N

    """
    if mode == 'square':
        window = np.ones(N)
    if mode == 'gaussian':
        window = gaussian(np.arange(N), 1, 0, N/2, (N-1)/5)
    if mode == 'blackman':
        window = blackman(N)

    return window / np.sum(window)

def sinc_filter(N, fc):
    """
    Generates a sinc filter with length N and cut-off frequency fc

    Parameters
    ----------
    N : array
        Length of filter
    fc : int
        Frequency

    Returns
    -------
    out : array
        Normalised sinc filter of length N

    """
    if not N % 2:
        N += 1

    n = np.arange(N)

    return np.sinc(2 * fc * (n - (N-1)/2))

def smooth_data(data, N: int=100, mode:str='square'):
    """
    Filter a given array of data using chosen type of smoothing function.
    Uses fftconvolve for speed and mirrors data to remove edge effects.
    
    Parameters
    ----------
    data : array
        Input array of data to be smoothed    
    N : int
        Length of filter
    mode : string
        Smoothing function to use > square, gaussian, blackman
    
    Returns
    -------
    out : array
        Filtered data

    """
    # create a boxcar window and then create a list of smoothed data
    avg_window = create_window(N, mode)
    # pad data to avoid edge effects
    N_pad = len(data)//2
    padded = np.concatenate((data[N_pad:], data, data[:N_pad]))

    return fftconvolve(avg_window, padded)[N_pad:-N_pad]