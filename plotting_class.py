'''
Sean Keenan
2024

Class for plotting and saving data

V.0.1

TODO
    - add checking for duplicate saves to prevent overwrite
    - add remaining plot functions and update
    - add formatting functionality so users can change from default easily

'''

from numpy import argmin, linspace, min
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, to_rgba
import matplotlib.patches as mpatches
import matplotlib.pyplot as mp
import numpy as np
import os

from typing import Any, Dict, List, Tuple, Union

from Function_files.addresses import Init_Directories
dirs = Init_Directories()

from Function_files.fitting_functions import exp_decay
from Function_files.math_functions import zoom

mp.style.use(dirs.mplstyle)

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

class Plotter:

    def __init__(self):

        self.dir = dirs.base                                # set directory
        self.folder = 'folder_name/'                        # folder name
        self.fname = 'file_name'                            # file name
        self.format = 'png'                                 # format of saved plots
        self.res = 80                                       # set resolution of plots
        self.scale_x = 1                                    # set scaling factor for x-axis
        self.scale_y = 1                                    # set scaling factor for y-axis
        self.precision = 2                                  # set precision for rounding
        self.title = ''                                     # title for plots
        self.x_label = 'x axis'                             # x axis label
        self.sec_x_label = None                             # second x axis label
        self.y_label = 'y axis'                             # y axis label

    def scale(self, axis='x'):

        prefix = {
            1E-9: 'G',
            1E-6: 'M',
            1E-3: 'k',
            1:'',
            1E3:'m',
            1E6:'$\\mu$',
            1E9:'n',
            1E12:'p'
            }
        
        if axis == 'x':
            return prefix[self.scale_x]

        if axis == 'y':
            return prefix[self.scale_y]

    def plot_aom_eff(self, 
                     data:np.array, 
                     max:int=1
                     ):
        """
        Plot the efficiency curve data for an AOM.

        Parameters
        ----------
        data : array
            array of data containing x values, single and 
            double pass values
        max : int or float
            maximum power before the AOM

        Returns
        -------
        fig, ax: 
            Figure and axes handles for the plot 
        
        """
        fig, ax = mp.subplots()
        ax.plot(data[:,0] * self.scale_x, (data[:,1]/max) *100, 'x', label='SP')
        ax.plot(data[:,0] * self.scale_x, (data[:,2]/max) *100, 'x', label='DP')
        ax.set(title=f'{self.title}')
        ax.set(xlabel=f'{self.x_label}', ylabel='Efficiency (\\%)')
        ax.legend(loc='upper left')

        return fig, ax

    def plot_scan(self, 
                  data_dict:dict, 
                  y_key:str, 
                  xerr_key:str= None, 
                  yerr_key:str= None):
        '''
        Plot a scan over a range of x values corresponding to 
        specific y values with errorbars if required. x data is
        given as key (string) and so will be converted to an int 
        and rounded.

        Parameters
        ----------
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

        Returns
        -------
        fig, ax: 
            Figure and axes handles for the plot

        '''
        x = [round(float(key) * self.scale_x, self.precision) for key in data_dict]
        y = [data_dict[key].get(y_key) * self.scale_y for key in data_dict]
    
        x_err = [data_dict[key].get(xerr_key, 0) * self.scale_x for key in data_dict] \
                if xerr_key else [0] * len(data_dict)

        y_err = [data_dict[key].get(yerr_key, 0) * self.scale_y for key in data_dict] \
             if yerr_key else [0] * len(data_dict)
        
        fig, ax = mp.subplots()
        ax.errorbar(x, y, y_err, x_err, fmt='.', color='C0')
        ax.set(title=f'{self.title}')
        ax.set(xlabel=f'{self.x_label}', ylabel=f'{self.y_label}')

        return fig, ax
    
    def plot_scope(self, 
                   time:np.array, 
                   channel_data:Union[np.array, list[np.array]], 
                   titles:list=[], 
                   multi:bool=False):
        '''
        Plot scope data.

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
        # plot single array
        if not isinstance(channel_data, list):
            channel_data = [channel_data]
        # scale data
        channel_data = [data * self.scale_y for data in channel_data] 
        # scale time
        time = time * self.scale_x

        # set labels if they exist or not
        labels = titles[:]
        labels.extend([f'Channel {i+1}' for i in range(len(labels), len(channel_data))])

        # chosose plot type
        if multi:
            num = len(channel_data)
            fig, ax = mp.subplots(nrows=num, ncols=1, sharex=True)
            # shared labels
            fig.tight_layout(w_pad=2, rect=[0.05, 0.05, 1, 1])
            fig.supxlabel(f'Time {self.scale("x")}s')
            fig.supylabel(f'Voltage ({self.scale("y")}V)')

            for index, axis in enumerate(ax):
                axis.set_title(labels[index])
                axis.plot(time, channel_data[index], color=custom_cmap(index))
        else:
            fig, ax = mp.subplots()
            for index, data in enumerate(channel_data):
                ax.plot(time, data, color=custom_cmap(index), label=labels[index])
                ax.legend()
            ax.set(xlabel=f'Time {self.scale("x")}s', 
            ylabel=f'Voltage ({self.scale("y")}V)')
        
        return fig, ax
    
    def plot_spectra(self,
                     x_data, 
                     y_data, 
                     data_indexes = [],  
                     shifter: float=0,  
                     sec_axis = True, 
                     data_labels = [], 
                     lims: tuple = (), 
                     woi: list = []):
        """
        Plot temperature dependent spectra with peaks highlighted
        and subsequent spectra shifted if given as arguments.
        
        Parameters
        ---------
        x_data: array or list of arrays
            X data to plot - length should match y_data
        y_data: array or list of arrays
            Y data to plot - length should match x_data
        data_indexes: array
            Indexes of peaks to mark
        shifter: int
            Value to shift subsequent y_data by
        sec_axis: bool
            Display secondary axis 
        data_labels: list
            List of labels for each set of data
        lims: tuple
            Corresponding x values to trim data
        woi: list
            List of x values to highlight by plotting 
            vertical lines

        Returns
        -------
        fig, ax: 
            Figure and axes handles for the plot

        """
        # fades plot colours instead of using default
        plot_colour = mp.cm.winter(linspace(0, 1, len(x_data)))
        
        fig, ax = mp.subplots()
        shift = 0
        for index, x_values in enumerate(x_data):
            # cut data to region of interest
            if lims:
                lower, upper = zoom(x_values, lims)
            else:
                lower = 0
                upper = None
            
            x = x_values[lower:upper]
            y = y_data[index][lower:upper]
            y -= (min(y) - shift)
            shift += shifter
            # labels for legend
            if data_labels:
                label = data_labels[index]
            else:
                label = '_nolegend_'
            # plot the data
            ax.plot(x, y, 
                    color=plot_colour[index], linestyle='-', 
                    alpha=0.8, label=f'{label}')
            # plot markers where applicable
            if data_indexes:
                ax.plot(x[data_indexes[index]], y[data_indexes[index]], 
                        color='red', marker='x', linestyle='None', 
                        alpha=1, label='_nolegend_')  
        # add secondary axis (wavelength / wavevector)
        if sec_axis:
            sec = ax.secondary_xaxis('top', 
                               functions=(lambda x: 1E7/x, 
                                          lambda x: 1E7/x))
            sec.set_xlabel(self.sec_x_label)
        # add vlines to indicate x values of interest
        if woi:
            for woi_set in woi:
                for vline in woi_set[0]:
                    ax.axvline(x=vline, linestyle=woi_set[1], 
                               color=woi_set[2], linewidth='1', alpha=0.3)
        # format the plot
        #ax.set(title=f'{self.title}')
        #ax.set(xlabel=f'{self.x_label}', ylabel=f'{self.y_label}')
        #ax.legend(bbox_to_anchor=(1.01, 1), loc='best', fontsize=8)     # legend outside of plot area
        #fig.tight_layout()
        ax.set(xlabel=f'{self.x_label}', ylabel=f'{self.y_label}')
        ax.legend()
        ax.get_legend().remove()     
        fig.tight_layout()

        region = '_' + str(round(x_values[lower])) + '_' + str(round(x_values[upper]))
        self.fname = self.fname + region

        return fig, ax
    
    def plot_T1_fit(self, time, data, fit_params):
        '''
        Plot T1 fitted data on top of experimental data

        Parameters
        ----------
        time: array
            Time data for corresponding fit_params
        data: array
            Y data for the corresponding fit_params
        fit_params: list / tuple
            Fit parameters to use in exp_decay function 

        Returns
        -------
        fig, ax: 
            Figure and axes handles for the plot

        '''
        if type(fit_params) == tuple:
            fit = fit_params[0]
        else:
            fit = fit_params

        # make label for plots
        scale_time = time * self.scale_x

        labels = ['Reference Fit', 'Log Scale Fit']                    # plot labels
        scales = ['linear', 'log']

        fig, ax = mp.subplots(nrows=1, ncols=2)
        for index, label in enumerate(labels):
        
            ax[index].set(title=f'{label}')
            ax[index].plot(scale_time, (data - fit[-1]) * self.scale_y, color='C0', alpha=1, label='Exp. Data')
            ax[index].plot(scale_time, (exp_decay((time), *fit) - fit[-1]) * self.scale_y, color='C1', linestyle='--', alpha=1, label='Fit')
            ax[index].set_yscale(f'{scales[index]}')
            ax[index].set(ylabel=f'Voltage ({self.scale("y")}V)')
            ax[index].legend()

        ax[0].set(xlabel=f'Time ({self.scale("x")}s)')

        return fig, ax
    
    def plot_T1_trigger(self, time:np.array, channel_data:np.array, i:dict=None):
        '''
        Plot T1 data where trigger has been selected.
        Used to check the trigger location

        Parameters
        ----------
        time: array
            Time channel data goes here
        channel_data: nd.array
            Channel data goes as a set of arrays
        i: int
            Trigger and channel indexes goes here as dictionary

        Returns
        -------
        fig, ax: 
            Figure and axes handles for the plot

        '''
        # scale time
        time = time * self.scale_x
        channel_data = [data * self.scale_y for data in channel_data]
        label = ['Reference', 'Transmitted']                    # plot labels
        # default to plot all
        if not i:
            i = {'trig':0,
                'ref_off':-1,
                'off':-1,
                'ramp':-1}
        
        fig, ax = mp.subplots(ncols=2, nrows=2, sharex=True)
        # tight layout and shared labels
        fig.tight_layout(w_pad=2, rect=[0, 0.05, 1, 1])
        fig.supxlabel(f'Time ({self.scale("x")}s)')

        plot_key = 'ref_off'                                    # plot reference data first
        for index, data in enumerate(channel_data):
            # plot linear data
            ax[0][index].set(title=f'{label[index]}')
            ax[0][index].plot(time[i['trig']:i['ramp']], data[i['trig']:i['ramp']], label='raw', alpha=0.8)
            ax[0][index].plot(time[i['trig']+i[plot_key]:i['ramp']], data[i['trig']+i[plot_key]:i['ramp']], label='cut', alpha=0.8)
            ax[0][index].set(ylabel=f'Voltage ({self.scale("y")}V)')
            # plot logarithmic data
            ax[1][index].set(title=f'{label[index]}')
            ax[1][index].plot(time[i['trig']:i['ramp']], data[i['trig']:i['ramp']], label='raw', alpha=0.8)
            ax[1][index].plot(time[i['trig']+i[plot_key]:i['ramp']], data[i['trig']+i[plot_key]:i['ramp']], label='cut', alpha=0.8)
            ax[1][index].set(ylabel='log scale (a.u.)')
            ax[1][index].set(yscale='log')

            plot_key = 'off'                                    # swap to plot transmitted data

        return fig, ax
    
    def plot_T2_trigger(self, time, channel_data, i:dict=None):
        '''
        Plot T2 data where trigger has been selected.
        Used to check the trigger location

        Parameters
        ----------
        time: array
            Time channel data goes here
        channel_data: list
            Channel data goes here as list
        i: int
            Trigger and channel indexes goes here as dictionary

        Returns
        -------
        fig, ax: 
            Figure and axes handles for the plot

        '''
        # scale time
        time = time * self.scale_x
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
        ax.plot(time, channel_data, label='original data', alpha=0.8)
        ax.plot(time[i['trig']+i['off']:i['ramp']], channel_data[i['trig']+i['off']:i['ramp']], label='echo selected', alpha=0.8)
        ax.set(xlabel=f'Time ({self.scale("x")}s)', ylabel=f'Voltage ({self.scale("y")}V)')
        ax.legend(loc='best')

        return fig, ax

    def plot_XRD(self, data:list, labels:list[str]=None):
        '''
        Plot XRD data for a given crystal

        Parameters
        ----------
        data: array
            x, y data 
        labels: list[str]
            list of labels for data

        Returns
        -------
        fig, ax: 
            Figure and axes handles for the plot

        '''
        assert len(data)%2 == 0, "Data must contain x,y multiples"
        
        fig, ax = mp.subplots()
        for i in range(len(data)//2):       # plot multiple x, y combinations
            ax.plot(data[i*2], data[i*2+1], label=labels[i])
        ax.set(xlabel=self.x_label, ylabel=self.y_label)
        ax.legend(loc='best')

        return fig, ax
    
    def plot_XRD_stick(self, data:list, labels:list[str]=None):
        '''
        Plot XRD data for a given crystal as a stem plot

        Parameters
        ----------
        data: array
            x, y data - peak locations
        labels: list[str]
            list of labels for data

        Returns
        -------
        fig, ax: 
            Figure and axes handles for the plot

        '''
        n = len(data)
        assert n%2 == 0, "Data must contain x,y multiples"
        m = n//2
        fig, ax = mp.subplots(nrows=m)
        for i in range(m):       # plot multiple x, y combinations
            st_vars = ax[i].stem(data[i*2], data[i*2+1], markerfmt='', label=labels[i], linefmt=f'C{i}')
            st_vars[1].set_linewidth(2)
            st_vars[2].set_visible(False)
            ax[i].legend(loc='upper right')
            
        return fig, ax
    
    def save_fig(self, figure):

        path = dirs.join(self.dir, self.folder, self.fname) + f'.{self.format}'
        figure.savefig(fname=path, dpi=self.res, format=self.format, bbox_inches='tight')
        
        return print('figure saved!')
    
    @staticmethod
    def zoom(data, bounds:tuple=()):
        """
        Zoom in on a particular area of interest in a dataset

        Parameters
        ----------
        data: array 
            Data to perform zoom
        bounds: tuple
            Lower and upper bounds of the region of interest

        Returns
        -------
        lims: tuple
            minimum index and maximum index for the zoomed data

        """
        lims = [argmin(abs(data - bounds[0])), 
                argmin(abs(data - bounds[1]))]

        return lims
