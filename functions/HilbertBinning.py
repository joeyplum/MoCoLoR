import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import hilbert
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter

class HilbertBinning():
    """
        Sorting respiratory signals into arbitrary bins for respiratory cycle analysis. Takes a given respiratory signal from
        bellows or diaphragm height, reorganizes the data into a single cosine cycle using a hilbert transform, excludes data
        that deviate too far from the average respiratory cycle and sorts the data into bins.

        Input arguments:
        signal      > the raw respiratory data in units of amplitude
        smoothing   > the degree to smooth the data for better hilbert results using a uniform convolution
        plots       > displays plots for debugging to see goodness of fit
        bin_type    > determines how to bin data
            'Fixed'    > divides all points into a fixed number of bins
            'Dynamic'  > divides all points into bins with a given number of data points
        fixed_bins  > number of bins if using fixed method
        dynamic_pts > number of points per bin using dynamic method

        Outputs:
        binned_indx >
        perc_ex     > percent of data excluded from binning

    """

    def __init__(self, signal, smoothing=100):
        # Smooth signal with an average filter
        self.signal = np.convolve(signal, np.ones(smoothing), mode='same')

        # Hilbert transform
        signal_hilbert = hilbert(signal)
        self.phase = np.angle(signal_hilbert)
        self.order = np.argsort(self.phase)
        self.signal_sorted = self.signal[self.order]

        # Fit data to sinusoidal phase
        self.fit_data()

    def cos_func(self, x, a, b):
        return a*np.cos(b*x)

    def fit_data(self):
        """
        Fit the signal to a cosine to determine the mean and standard deviation of the signal amplitude
        """

        sorted_signal_median = median_filter(self.signal_sorted, 2000)

        params, params_cov = curve_fit(self.cos_func, self.phase[self.order], self.signal_sorted, p0=[200000,1/(2*np.pi)])
        self.amp  = params[0]
        self.freq = params[1]

        stdev = np.diag(params_cov)
        self.amp_err  = stdev[0]

    def outliers(self, n_std):
        """
        Determines signal points to exclude from binning. Excludes points if they are more than 2 standard deviations from the mean signal
        """
        # Identify outliers
        indx = (self.signal>(self.cos_func(self.phase, self.amp, self.freq)+n_std*self.amp_err)) |\
            (self.signal<(self.cos_func(self.phase, self.amp, self.freq)-n_std*self.amp_err))
        self.outliers = indx
        
        self.perc_ex = np.sum(indx) / self.signal.size * 100
    
    def sort_fixed_bin(self, n_bins, stdev=1):
        """
        Sorts the signal into n bins according to both the phase and amplitude of the signal. First sorts all points into n bins
        of equal phase witdth, with bins spaced evenly starting at -pi (the bin at +pi is the same as at -pi). Next removes 
        outliers with amplituds outside of 2 standard deviations from the mean.
        Input:
        n_bins  > the number of bins to sort the signal. Should be even to ensure inspiration and expiration both have their own bin

        Output:
        bin_hot > a Nxn binary array, where N is the signal length and n the number of bins. A 1 in the nth column indicates that
                  point goes in the nth bin. If a row has no column with a 1, that row is an outlier
        """
        self.outliers(stdev)

        self.bin_array = np.zeros_like(self.signal)

        N = self.bin_array.size
        bin_centers = np.linspace(-np.pi, np.pi, n_bins+1) 

        # Vector_method to put in bins
        # Make a Nxn_bins matrix to find which bin center the phase is closest to
        cent_matrix  = np.tile(bin_centers, [N, 1])
        phase_matrix = np.tile(np.reshape(self.phase, [self.phase.size, 1]), [1, n_bins+1])
        self.bin_array = np.argmin(np.abs(cent_matrix-phase_matrix),-1)
        self.bin_array[self.bin_array==n_bins] = 0 # bin centered on -pi is the same as +pi

        # Hot-encode the array
        bin_hot = np.zeros([self.bin_array.size, self.bin_array.max()+1], dtype=int)
        bin_hot[np.arange(self.bin_array.size), self.bin_array] = 1

        # Make column 0 be for outliers
        outlier_tile = np.tile(np.reshape(self.outliers, [self.outliers.size, 1]), [1, n_bins])
        bin_hot[outlier_tile==1] = 0
        bin_hot = np.concatenate((np.reshape(self.outliers, [self.outliers.size, 1]), bin_hot,), axis=1)

        self.bin_hot = bin_hot

        return bin_hot

    def sort_dynamic_bin(self, n_pts, stdev=1):
        """
        Sorts the signal into bins according to a sliding window such that each bin contains the same number of n_pts points. Points
        can belong to multiple bins due to overlapping windows. The number of possible window frames is N-n_pts-o, where N is the 
        total number of points in the signal, n_pts is the width of the window and o is the number of outlier points.
        Input:
        n_pts   > Number of signal points to include per window

        Output:
        bin_hot > a Nxn binary array, where N is the signal length and n the number of windows. A 1 in the nth column indicates that 
                  point is included in the nth window. A point may be included in multiple windows

        Not currently implemented - AMM 2023/10/10
        """
        # self.outliers(stdev)
        # out_sorted = self.outliers[self.order]

        # keep_pts = np.zeros([self.signal.size, n_pts], dtypes='bool')
        # for i in range(n_pts):
        #     for j in range(self.signal.size):



        # return bin_hot
    
    def set_dynamic_fps(self, fps):
        """
        Dynamic binning returns the maximum fps which may be inefficient for reconstruction. This sets a definite framerate for 
        reconstructions.

        Not currently implemented - AMM 2023/10/10
        """
        #return self.bin_hot[:,:,fps]
        

    def plot_signal(self):
        """
        Mainly for debugging to see how the fits / bins perform. Plots the original signal and instantaneous phase.
        """
        plt.figure(figsize=(16,3))
        ax1 = plt.subplot(2,1,1)
        ax1.plot(self.signal)

        ax2 = plt.subplot(2,1,2, sharex=ax1)
        ax2.plot(self.phase)
        plt.xlabel("Acquisition Number")

    def plot_phase_resample(self, n_std=2):
        """
        Plots the original signal, colored by acquisition time. Also plots data re-sampled along a single phase with the same
        color codes to show where individual oscillations fit on the phase.
        """
        color = np.linspace(1, 10, np.size(self.signal))

        plt.figure()
        plt.subplot(2,1,1)
        plt.scatter(np.linspace(0, self.signal.size, self.signal.size) ,self.signal, c=color, s=0.5)

        plt.subplot(2,1,2)
        plt.scatter(self.phase, self.signal, c=color, s=0.5)
        x = np.linspace(-np.pi, np.pi, self.signal.size)
        plt.plot(x, self.cos_func(x, self.amp, self.freq), c='k')
        plt.plot(x, self.cos_func(x, self.amp, self.freq)+n_std*self.amp_err, c='k')
        plt.plot(x, self.cos_func(x, self.amp, self.freq)-n_std*self.amp_err, c='k')
        plt.xlabel('Phase')
        plt.xlabel('Amplitude')

    def plot_fixed_bin(self, n_bins):
        """
        Plots the data interpolated to a single phase. Data are color-coded by bin and set to gray for outliers. 
        """
        hsv = matplotlib.colormaps['prism']
        c = [[0.5, 0.5, 0.5]] 
        for i in range(n_bins):
            c.append(hsv(i/n_bins))

        cmap = LinearSegmentedColormap.from_list("Bin Colors", c)

        plt.figure()
        plt.subplot(2,1,1)
        plt.scatter(np.arange(self.signal.size), self.signal, c=np.argmax(self.bin_hot, -1), cmap=cmap, s=0.5)
        plt.xlabel('Acquisition number')
        plt.ylabel('Amplitude')

        plt.subplot(2,1,2)
        plt.scatter(self.phase, self.signal, c=np.argmax(self.bin_hot, -1), cmap=cmap, s=0.5)
        plt.xlabel('Phase')
        plt.ylabel('Amplitude')