'''
Sean Keenan, PhD Physics
Quantum Memories Group, Heriot-Watt University, Edinburgh
2023

Functions designed to perform mathematical operations on data
'''

import numpy as np
from scipy.fftpack import fft, fftfreq
from scipy.integrate import simpson

import numpy as np

def average_arrays(list_of_arrays:list):
    '''
    Calculate the average and standard deviation for a list of numpy arrays
    excluding any arrays that contain inf or nan values.
    
    list_of_arrays : list of numpy arrays
        List containing numpy arrays

    '''
    if not list_of_arrays:
        raise ValueError("Input list is empty")

    temp_sum = np.zeros_like(list_of_arrays[0])
    temp_sum_sq = np.zeros_like(list_of_arrays[0])
    count = 0
    for array in list_of_arrays:
        if not np.isinf(np.sum(array)) and not np.isnan(np.sum(array)):
            temp_sum += array
            temp_sum_sq += np.square(array)
            count += 1

    if count == 0:
        raise ValueError("No valid arrays found in the input list")

    avg = temp_sum / count
    variance = (temp_sum_sq / count) - np.square(avg)
    std_dev = np.sqrt(variance)

    return avg, std_dev

def bin_data(data, N: int = 10, edge: bool = False):
    """
    Bin the data and return mean
    
    Parameters
    ----------

    data : list of data to average
    bins : number of bins to group data into
    edge : choose to include right or left edge of bin.

    Returns
    -------

    mean : value of data

    """
    minimum = min(data)
    maximum = max(data)
    bins = np.linspace(minimum, maximum, N+1) 
    binned = np.digitize(data, bins, right=edge)

    return data[binned == np.bincount(binned).argmax()].mean()

def corrected_pulse_area(dataset_1, indexes:list[int], dataset_2=None):
    '''
    Calculate the corrected and normalised pulse area.

    First zero the data sets using a simple binning function on a
    section of the data set with no pulse. Then calculate the area 
    of the zeroed datasets before normalising.

    Parameters
    ----------
    
    dataset_1 :
        excel file for the transmitted and reference pulses
    dataset_2 :
        excel file for the correction pulse data
    indexes :
        list of indexes for the corresponding column data
        
    [0] trans :
        index for the transmitted data
    [1] ref :
        index for the reference data
    [2] time :
        index for the time data
    [3] start :
        start index of data set to correct
    [4] start :
        stop index of data set to correct

    Returns
    ----------

    normalise

    '''
    control = 0
    zeroed_1 = zero_data(dataset_1[:, indexes[0]], [indexes[3], indexes[4]])
    zeroed_2 = zero_data(dataset_1[:, indexes[1]], [indexes[3], indexes[4]])
    
    area_1 = simpson(y=zeroed_1, x=dataset_1[:,indexes[2]])
    area_2 = simpson(y=zeroed_2, x=dataset_1[:,indexes[2]])
    
    if dataset_2 is not None:
        control = np.abs(simpson(y=dataset_2[:,indexes[0]], x=dataset_2[:,indexes[2]]))
        
    return normalise(dataset_1=area_1, reference=area_2, dataset_2=control)

def find_longest(data_list):
    """
    Find longest list and length within a lst
    
    Parameters
    ----------

    data_list : list of data

    Returns
    -------

    longest : longest list
    length : length of longest list
    """
    longest = max(data_list, key = lambda i: len(i))
    length = max(map(len, data_list))

    return longest, length

def calc_fft(time, amplitude):
    """
    Perform FFT calculation for amplitude component and generate the frequency
    from times.

    Parameters
    ----------
    time : 1D data array / list to use as reference
    amplitude : 1D data array / list of transmission data

    Returns
    -------
    frequencies : list of frequency values from input time data
    ffts : list of fft calculated from input amplitude data

    """
    N = len(time)
    T = time[1] - time[0]
    fftd = fft(amplitude)
    frequencies = fftfreq(N, T)

    return frequencies, fftd

def normalise(dataset_1, dataset_2, reference=1):
    """
    Normalise a set of data by subtracting a control set and dividing by
    a reference (optional).

    Parameters
    ----------
    dataset_1 : data array to normalise
    dataset_2 : control data to subtract
    reference : reference data to divide by

    Returns
    -------
    normalised dataset

    """
    return np.divide(np.subtract(dataset_1, dataset_2), reference)

def OD_calc(ref_data, trans_data, c_factor: float=1):
    """
    Perform OD calculation for transmission data and adjust the reference
    using the correction factor if neccesary

    Parameters
    ----------
    reference : data array to use as reference
    transmission : data array of transmission data
    correction : correction factor for the reference data

    Returns
    -------
    calculated optical depth

    """
    return np.log((ref_data * c_factor)/trans_data)

def zero_data(data, indexes: list[int]=[0, -1]):
    """
    Correct for background on a data set and zero

    Parameters
    ----------
    data : list / array - data to perform zoom
    indexes : list - lower and upper bounds of the region to correct for

    Returns
    -------
    zeroed data 

    """
    correction = bin_data(data[indexes])

    return data + abs(correction)

def zoom(data, bounds:tuple=()):
    """
    Zoom in on a particular area of interest in a dataset

    Parameters
    ----------
    data : list / array - data to perform zoom
    bounds : tuple - lower and upper bounds of the region of interest

    Returns
    -------
    start, stop : start and stop index for the zoomed data

    """
    start = np.argmin(abs(data - bounds[0]))
    stop = np.argmin(abs(data - bounds[1]))

    return start, stop