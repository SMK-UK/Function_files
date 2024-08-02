'''
Sean Keenan, PhD Physics
Quantum Memories Group, Heriot-Watt University, Edinburgh
2023

Functions designed to plot data

V.1.0
'''

from numpy import argmin, linspace, min
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, to_rgba
import matplotlib.patches as mpatches
import matplotlib.pyplot as mp
import numpy as np
import os
from Function_files.fitting_functions import exp_decay

mp.style.use(r"C:\Users\sk88\Documents\Python\Function_files\signature.mplstyle")

# colour map for plotting scope data
scope_colours = ['gold', 'limegreen', 'orange', 'royalblue']
scope_rgba = [to_rgba(colour) for colour in scope_colours]
og_cmap = mp.get_cmap('tab10')

# Get the colors from the existing colormap
og_colors = og_cmap(np.linspace(0, 1, og_cmap.N))
# Combine existing colors with predefined colors
combined_colors = np.vstack([scope_rgba, og_colors])
# Create a new colormap from the combined colors
custom_cmap = ListedColormap(combined_colors)

def plot_scan(data_dict:dict, y_key:str, xerr_key:str= None, yerr_key:str= None, dp:int= 2, scale_x:int=1, scale_y:int=1):
    '''
    Plot a scan over a range of x values corresponding to 
    specific y values with errorbars if required. x data is
    given as key (string) and so will be converted to an int 
    and rounded.

    data_dict : dictionary
        Dictionary with the x data points as keys and y 
        data points as values
    y_key : string
        Corresponding key for desired y data points to 
        extract
    xerr_key : string
        Corresponding key for the x data error
    yerr_key : string
        Corresponding key for the y data error
    dp : int
        x data points from strings to int and round
    scale_x : int
        scale the data in x
    scale_y : int
        scale the data in y

    '''
    fig, ax = mp.subplots()
    for key in data_dict:
        x = round(float(key), dp) * scale_x
        y_data = data_dict[key][y_key] * scale_y 
        if xerr_key:
            error_x = data_dict[key][xerr_key] * scale_y
        else:
            error_x = 0
        if yerr_key:
            error_y = data_dict[key][yerr_key] * scale_y
        else:
            error_y = 0
        
        ax.errorbar(x, y_data, error_y, error_x, fmt='.b')

    return fig, ax

def plot_scope(time, channel_data, titles=[], multi: bool=False):
    '''
    Plot scope data.

    TO DO: fix bug for plotting only 1 array

    <time>:
        time channel data goes here
    <channel_data>:
        channel data goes here as list
    <titles>:
        list of titles corresponding to channel_data
        goes here
    <multi>:
        Choose to plot individual or on top of one another

    '''
    # set labels if they exist or not
    labels = []
    if titles:
        for title in titles:
            labels.append(title)
    if not titles or len(titles) < len(channel_data):
        for index in range(len(titles), len(channel_data), 1):
            labels.append(f'Channel {index+1}')
    # chosose plot type
    if multi:

        num = len(channel_data)
        fig, ax = mp.subplots(nrows=num, ncols=1, sharex='all')
        # shared labels
        fig.tight_layout(w_pad=2, rect=[0.05, 0.05, 1, 1])
        fig.supxlabel('Time ($\mu$s)')
        fig.supylabel('Voltage (V)')

        for index, axis in enumerate(ax):
            axis.set_title(labels[index])
            axis.plot(time, channel_data[index], color=custom_cmap(index))
    else:
        fig, ax = mp.subplots()
        for index, data in enumerate(channel_data):
            ax.plot(time, data, color=custom_cmap(index), label=labels[index])
            ax.legend()
        ax.set(xlabel='Time ($\mu$s)', ylabel='Voltage (V)')
    
    return fig, ax

def plot_spectra(x_data, y_data, data_indexes = [], 
                 keys = list[str], 
                 shifter: float=0, 
                 axis_lbls = None, 
                 sec_axis = True, 
                 save = False, 
                 data_labels = [], 
                 lims: tuple = (), 
                 woi: list = [], 
                 res = 80):
    """
    TO DO:

    - add ability to change the sec_axis via input
    - update the code to handle functionality externally and make it more universal
        e.g. saving figures, labels etc.
    
    Plot and display spectra that are combined in plots according to the keys

    <x_data>:
        Wavelength data or similar
    <y_data>:
        Absorption data or similar
    <keys>:
        List of strings to discriminate between separate plots
    <shifter>:
        Choose to shift in y each data set by a predetermined amount
    <axis_lbls>:
        Define labels to give each axis
    <sec_axis>: 
        Boolean to display a secondary axis; default is designed to display wavenumbers and wavelength
    <save>:
        Boolean to save the figures
    <data_labels>:
        List of file names corresponding to each y_data set
    <lims>:
        Cut the data to display only a sub section in x
    <woi>:
        List of vertical lines of interest to add to the plot
    <res>:
        Set the resolution for plotting

    """
    data_lbl = None

    for m, key in enumerate(keys):
        fig, ax = mp.subplots()
        ax.grid(True, color='silver', linewidth=0.5)
        if axis_lbls:
            ax.set_title('Halfwave Plate: ' + key)
            ax.set(xlabel=axis_lbls[0], ylabel=axis_lbls[1])
        if sec_axis:
            sec_ax = ax.secondary_xaxis('top', functions=
                                        (lambda x: 1e7/x, lambda x: 1e7/x))
            sec_ax.set_xlabel('Wavelength (nm)')
        if woi:
            for woi_set in woi:
                for vline in woi_set[0]:
                    ax.axvline(x=vline, linestyle=woi_set[1], 
                               color=woi_set[2], linewidth='1')
        shift = 0
        for o, x in enumerate(x_data[m]):
            
            plot_colour = mp.cm.winter(linspace(0, 1, len(x_data[m])))
            if lims:
                lower, upper = zoom(x, lims)
                x = x[lower:upper]
                y = y_data[m][o][lower:upper]
            else:
                y = y_data[m][o]
            y -= (min(y) - shift)
            if data_labels:
                data_lbl = os.path.split(data_labels[o])[1]
            ax.plot(x, y, color=plot_colour[o],
                    linestyle='-', alpha=0.8, label=data_lbl)
            if data_indexes:
                ax.plot(x[data_indexes[m][o]], y[data_indexes[m][o]], 
                        color='red',
                   marker='x', linestyle='None', alpha=1, 
                   label='_nolegend_')     
            shift += shifter

        ax.legend(bbox_to_anchor=(1.01, 1), loc='best', fontsize=8)
        fig.tight_layout()

        if save:
            folder = os.path.split(data_labels[0])[0]
            region = str(round(x[lower])) + '_' + str(round(x[upper])) + '_' + key
            name = os.path.join(folder, region + '.png')

            fig.savefig(fname=name, dpi=res, format='png', bbox_inches='tight')

def plot_TPA(data_sets, data_keys, axis_dict, set_labels):
    '''
    Plot TPA data for a given experiment
    

    <data_sets>:
        time channel data goes here
    <channel_data>:
        channel data goes here as list
    <i>:
        trigger and channel indexes goes here as dictionary

    '''
    figs = []
    axes = []
    for data_set in data_sets:
        fig, ax = mp.subplots(nrows=2, sharex=True)
        figs.append(fig)
        axes.append(axes)
        for index, subset in enumerate(data_set):
            for key in data_keys:

                if "cph" in key:
                    axis = 0
                    cp_axis = axis_dict['cph']
                else:
                    axis = 1
                    cp_axis = axis_dict['cpl']
                if "sph" in key:
                    mark = 6
                    colour = 'blue'
                else:
                    mark = 7
                    colour = 'red'
                ax[axis].plot(set_labels[index], subset[key], marker=mark, color=colour)
                ax[axis].set(title=f'Control E-Field along d$_{cp_axis}$')
                ax[axis].legend([f"SP E-field along d$_{axis_dict['sph']}$", f"SP E-field along d$_{axis_dict['spl']}$"])

    return figs, axes

def plot_T1_trigger(time_data, channel_data:list, i:dict=None):
    '''
    Plot T1 data where trigger has been selected.
    Used to check the trigger location

    <time_data>:
        time channel data goes here
    <channel_data>:
        channel data goes here as list
    <i>:
        trigger and channel indexes goes here as dictionary

    '''
    # default to plot all
    if not i:
        i = {'trig':0,
             'ref_off':-1,
             'off':-1,
             'ramp':-1}

    fig, ax = mp.subplots(ncols=2, nrows=2, sharex=True)
    # tight layout and shared labels
    fig.tight_layout(w_pad=2, rect=[0, 0.05, 1, 1])
    fig.supxlabel('Time ($\mu$s)')

    # plot linear reference data
    ax[0][0].set_title('Reference')
    ax[0][0].plot(time_data[i['trig']:i['ramp']], channel_data[0][i['trig']:i['ramp']], label='raw', alpha=0.8)
    ax[0][0].plot(time_data[i['trig']+i['ref_off']:i['ramp']], channel_data[0][i['trig']+i['ref_off']:i['ramp']], label='cut', alpha=0.8)
    ax[0][0].set(ylabel='Voltage (V)')
    # plot logarithmic reference data
    ax[1][0].set_title('Reference')
    ax[1][0].plot(time_data[i['trig']:i['ramp']], channel_data[0][i['trig']:i['ramp']], label='raw', alpha=0.8)
    ax[1][0].plot(time_data[i['trig']+i['ref_off']:i['ramp']], channel_data[0][i['trig']+i['ref_off']:i['ramp']], label='cut', alpha=0.8)
    ax[1][0].set(ylabel='log scale (a.u.)')
    ax[1][0].set_yscale('log')
    # plot linear transmitted data
    ax[0][1].set_title('Transmitted')
    ax[0][1].plot(time_data[i['trig']:i['ramp']], channel_data[1][i['trig']:i['ramp']], label='raw', alpha=0.8)
    ax[0][1].plot(time_data[i['trig']+i['off']:i['ramp']], channel_data[1][i['trig']+i['off']:i['ramp']], label='cut', alpha=0.8)
    ax[0][1].set(ylabel='Voltage (V)')
    ax[0][1].legend()
    # plot logarithmic transmitted data
    ax[1][1].set_title('Transmitted')
    ax[1][1].plot(time_data[i['trig']:i['ramp']], channel_data[1][i['trig']:i['ramp']], label='raw', alpha=0.8)
    ax[1][1].plot(time_data[i['trig']+i['off']:i['ramp']], channel_data[1][i['trig']+i['off']:i['ramp']], label='cut', alpha=0.8)
    ax[1][1].set(ylabel='log scale (a.u.)')
    ax[1][1].set_yscale('log')

    return fig, ax

def plot_T1_fit(time, data, fit_params, scale=[1, 1, 1]):
    '''
    Plot T1 fitted data on top of experimental data
    
    '''
    if type(fit_params) == tuple:
        fit = fit_params[0]
    else:
        fit = fit_params
    
    time = time * scale[1]
    fit = [x*y for x, y in zip(fit, scale)]

    fig, ax = mp.subplots(nrows=1, ncols=2)
    
    ax[0].set_title('Reference Fit')
    ax[0].plot(time, data - fit[-1], color='blue', alpha=0.5, label='Exp. Data')
    ax[0].plot(time, exp_decay((time), *fit) - fit[-1], color='orange', linestyle='--', alpha=1, label='Fit')
    ax[0].ticklabel_format(axis='both', style='sci', scilimits=(0,0), useMathText = True)
    ax[0].legend()

    ax[1].set_title('Log Scale Fit')
    ax[1].plot(time, data - fit[-1], color='blue', alpha=0.5, label='Exp. Data')
    ax[1].plot(time, exp_decay((time), *fit) - fit[-1], color='orange', linestyle='--', alpha=1, label='Fit')
    ax[1].set_yscale('log')
    ax[1].legend()

    return fig, ax

def plot_T2_trigger(time_data, channel_data, i:dict=None):
    '''
    Plot T2 data where trigger has been selected.
    Used to check the trigger location

    <time_data>:
        time channel data goes here
    <channel_data>:
        channel data goes here
    <i>:
        trigger and channel indexes goes here as dictionary

    '''
    # default to plot all
    if not i:
        i = {'trig': 0,
             'off': 0,
             'ramp': -1}

    fig, ax = mp.subplots()
    # tight layout and shared labels
    fig.tight_layout(w_pad=2, rect=[0, 0.05, 1, 1])

    # plot echo data
    ax.set_title('Stimulated Emission')
    ax.plot(time_data, channel_data, label='original data', alpha=0.8)
    ax.plot(time_data[i['trig']+i['off']:i['ramp']], channel_data[i['trig']+i['off']:i['ramp']], label='echo selected', alpha=0.8)
    ax.set(xlabel=('Time ($\mu$s)'), ylabel='Voltage (V)')
    ax.legend(loc='best')

    return fig, ax
   
def zoom(data, bounds:tuple=()):
    """
    zoom in on a particular area of interest in a dataset

    Parameters
    ----------
    data : list / array - data to perform zoom
    bounds : tuple - lower and upper bounds of the region of interest

    Returns
    -------
    start, stop : start and stop index for the zoomed data

    """
    start = argmin(abs(data - bounds[0]))
    stop = argmin(abs(data - bounds[1]))

    return start, stop