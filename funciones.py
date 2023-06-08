# coding: utf-8


# Useful functions

import numpy as np
import scipy.signal as sig

# =============================================================================
# PLOTTING
# =============================================================================

    
def labels_bands(filter=0, Hz = False):
    """
    Generates labels for frequency bands.

    Parameters
    ----------
    filter : int, optional
        Filter type to determine the frequency bands. 
        0 for octave filter, 1 for one-third octave filter.
        Default is 0.
    Hz : bool, optional
        Determines if the labels should include the 'Hz' unit.
        If True, labels will include 'Hz'; if False, labels will not include 'Hz'.
        Default is False.

    Returns
    -------
    labels : list
        List of labels for the frequency bands.

    Notes
    -----
    The function generates labels for frequency bands based on the specified filter type.
    If `filter` is 0, the labels are generated for octave filter bands.
    If `filter` is 1, the labels are generated for one-third octave filter bands.
    The labels can include the 'Hz' unit if `Hz` is set to True.
    """

    if filter == 0: # Octaves filter
        center = ['31.5', '63', '125', '250', '500', '1 k', '2 k', '4 k', '8 k', '16 k']
    if filter == 1: # Third octaves filter
        center = ['25', '31.5', '40', '50', '63', '80', '100', '125', '160', '200', 
                  '250', '315','400', '500', '630', '800', '1 k', '1.25 k', '1.6 k', 
                  '2 k', '2.5 k', '3.15 k', '4 k', '5 k', '6.3 k', '8 k', '10 k', 
                  '12.5 k', '16 k', '20 k']
    if Hz:
        labels = []
        for i in center:
            labels.append(str(i + 'Hz'))
        return labels
    else:
        return center

# =============================================================================
# FOURIER
# =============================================================================

def spectrum(x, fs, dB=False):
    """
    Parameters
    ----------
    x : array
        Input array.
    fs : int
        Sampling frequency.
    dB : bool, optional
        If True, the output X is in decibels (dB). Default is False.

    Returns
    -------
    freq : array
        Array of frequencies.
    X : array
        Fourier transformed array.
    """

    X = np.fft.rfft(x)
    X = np.abs(X)
    X /= max(X)
    if dB==True:
        X = 20*np.log10(X)

    freq = np.fft.rfftfreq(x.size, 1/fs)
    
    return freq, X


# =============================================================================
# ADJUSTMENTS AND ANALYSIS
# =============================================================================


def zero_padding(x, N):
    if x.ndim == 1: # Vector
        ceros = np.zeros(N)
        return np.hstack([x, ceros])
    elif x.ndim == 2 and x.shape[1] == 2: # Stereo signal
        ceros = np.zeros([N, 2])
        return np.vstack([x, ceros])
    
def rms(x):

    """
    Calculates the root mean square (RMS) value of a signal.
    Parameters
    ----------
    x : array-like
        Signal to calculate the RMS value.

    Returns
    -------
    float
        RMS value.

    """
    
    return np.sqrt(np.mean(x**2))

def rms_windows(x, fs, time_window):
    """
    Calculates the root mean square (RMS) value of signal segments.

    Parameters
    ----------
    x : array-like
        Input signal.
    fs : int or float
        Sampling frequency of the signal.
    time_window : float
        Length of each segment in seconds.

    Returns
    -------
    result : array
        Array of RMS values for each segment.

    Notes
    -----
    The function calculates the RMS value of signal segments with a specified time window.
    The input signal `x` is divided into segments of length `time_window` seconds.
    If the length of the input signal is not divisible by the segment length, zero-padding is applied.
    The RMS value is calculated for each segment, and the results are returned as an array.
    The last segment might have a different length if the input signal size is not divisible by `time_window`.
    The sampling frequency `fs` is used to determine the segment length in samples.
    """

    W = int(time_window * fs)
    N = x.size
    x_original = x
    
    if N % W != 0:
        ceros = W - N % W
        x = zero_padding(x, ceros)
        N = x.size
    
    Nwin = N // W
    result = np.zeros(Nwin)
    
    for i in range(Nwin):
        result[i] = rms(x[W*i:W*(i+1)])
    
    result[-1] = rms(x_original[W*(Nwin-1):])
    return result


def least_squares(x, y, n=1):
    """
    Performs a least squares fit of a polynomial function to the given data points.

    Parameters
    ----------
    x : array-like
        Input array representing the x-coordinates of the data points.
    y : array-like
        Input array representing the y-coordinates of the data points.
    n : int, optional
        Degree of the polynomial to fit. Default is 1 (linear).

    Returns
    -------
    p : array
        Coefficients of the polynomial fit.
    """
    A = np.vstack([x, np.ones(len(x))]).T
    p = np.linalg.lstsq(A, y, rcond=None)[0]
    return p

# =============================================================================
# FILTERS
# =============================================================================
    


def butter_bandpass(N, flow, fhigh, fs):
    """
    Returns the second-order section (SOS) coefficients of a Butterworth bandpass filter.

    Parameters
    ----------
    N : int
        Order of the filter.
    flow : float
        Lower cutoff frequency of the bandpass filter.
    fhigh : float
        Upper cutoff frequency of the bandpass filter.
    fs : float
        Sampling frequency.

    Returns
    -------
    sos : array
        Second-order section (SOS) coefficients of the Butterworth bandpass filter
    """
    # The frequency is normalize in [0, pi]
    wlow = flow / (0.5 * fs)
    whigh = fhigh / (0.5 * fs)
    return sig.butter(N, [wlow, whigh], 'bandpass', output='sos')


def bandpass_filter(signal, flow, fhigh, fs, N=5, filtfilt=False):
    """
    Applies a 'sos' bandpass filter.
    Parameters
    ----------
    signal : 1-D array
        Signal to be filtered.
    flow : int or float
        Lower cutoff frequency.
    fhigh : int or float
        Upper cutoff frequency.
    fs : int
        Signal sampling frequency.
    N : int, optional
        Filter order. Default is 5.

    Returns
    -------
    y : 1-D array
    Filtered signal.
    """

    sos = butter_bandpass(N, flow, fhigh, fs)
    if filtfilt:
        y = sig.sosfiltfilt(sos, signal)
    else: 
        y = sig.sosfilt(sos, signal)
    return y

def octaves(f0=31.5, normalized=True):
    """
    Returns the central frequencies for octave bands.

    Parameters
    ----------
    f0 : float, optional
        Central frequency of the lower band in Hz. Default is 31.5.

    Returns
    -------
    list
    Central frequencies of octave bands.
    """
    if normalized:
        center = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        return center
    else:
        return [round(f0 * 2 ** i, 1) for i in range(0, 10)]

def thirds(f0=25, normalized=True):
    """
    Returns the central frequencies for one-third octave bands. 

    Parameters
    ----------
    f0 : float, optional
        Central frequency of the lower band in Hz. Default is 25.

    Returns
    -------
    list
    Central frequencies of one-third octave bands.
    """

    if normalized:
        center = [25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 
                  500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 
                  5000, 6300, 8000, 10000, 12500, 16000, 20000]
        return center
    else:
        return [round(f0 * 2 ** (i/3), 1) for i in range(0, 29)]


def calc_parameter(param, imp, fs, filter='thirds', N=5):
    """
    Calculates the 'param' parameter for a signal in octave or one-third octave bands.

    Parameters
    ----------
    param : function
        Function that calculates the parameter. It should be of the form f(x, fs).
    imp : 1-D array
        Signal to process. Typically an impulse response.
    fs : int
        Sampling frequency of imp.
    filter : str, optional
        Type of filter to apply. If filter='octaves', calculates for octave bands,
        if it is 'thirds', calculates for one-third octave bands. Default is 'thirds'.

    Returns
    -------
    output : list
    List with the calculated parameters for each band.
    """
    
    if filter == 'thirds':
        output = []
        frecs = thirds()
        for i in range(len(frecs)):
            finf = 2 ** (-1/6) * frecs[i]
            fsup = 2 ** (1/6) * frecs[i]
            impfilt = bandpass_filter(imp, finf, fsup, fs, N)
            output.append(param(impfilt, fs))
        return output
    elif filter == 'octaves' :
        output = []
        frecs = octaves()
        for i in range(len(frecs)):
            finf = 2 ** (-1/2) * frecs[i]
            fsup = 2 ** (1/2) * frecs[i]
            impfilt = bandpass_filter(imp, finf, fsup, fs, N)
            output.append(param(impfilt, fs))
        return output

def maf_rcsv(x, M):
    """
    Implementation of a recursive moving average filter. Has rounding error when used with
    floats instead of ints.

    Parameters
    ----------
    x : array-like
        Signal to be filtered.
    M : int
        Window size.

    Returns
    -------
    y : 1-D array
    Filtered signal.

    """
    if type(M) != int:
        M = int(M)      # Window size
    if M >= len(x):
            raise IndexError('Window length greater than signal')
    L = len(x) - M + 1  # Amount of windows
    y = np.zeros(L)
    
    # Acc = np.sum(x[0:M-1])
    Acc = np.sum(x[0:M])
    y[0] = Acc/M
    
    for i in range(1, L):
        Acc = Acc + x[i+M-1] - x[i-1]
        y[i] = Acc/M
    y = np.hstack([y, y[L-M+1:]])
    return y


# =============================================================================
# CONVERSION TO dB    
# =============================================================================
def nivel_bandas(data, f_c, freqs, octave, dB=True):
    """
    Calculates the signal level for bands of the required width.

    Parameters
    ----------
    data : array-like
        Matrix with the signals to process in dB. axis=0 should be the
        frequency axis.
    f_c : array-like
        List of central frequencies of the bands of interest.
    freqs : array-like
        Frequency axis corresponding to 'data'.
    octave : int
        Bandwidth. octave = 1 for octave bands, octave = 3 for one-third
        octave bands, ...

    Returns
    -------
    numpy array
    Processed data with information for the specified bands in f_c.
    """

    bands = []
    for i in f_c:
        fsup = i * 2 ** (1/(2*octave))
        finf = i * 2 ** (-1/(2*octave))
        
        idxsup = np.argmin(abs(freqs- fsup))
        idxinf = np.argmin(abs(freqs- finf))
        
        if dB:
            aux = 10 ** (0.1 * data[idxinf:idxsup+1, :])
            bands.append(10*np.log10(aux.sum(axis=0)
                                      /len(data[idxinf:idxsup+1, 1])))
        else:
            nivel = 20*np.log10(sum(
                data[idxinf:idxsup+1]/len(data[idxinf:idxsup+1])))
            
            bands.append(nivel)
    return np.array(bands)

def min_nonzero(x):

    """
    Returns the minimum non-zero value in the array.

    Parameters
    ----------
    x : array-like
        Array from which to find the minimum non-zero value.

    Returns
    -------
    float or int
        Minimum non-zero value found in the array.
    """
    min = 1
    for i in range(x.size):
        if x[i] < min and x[i] != 0:
            min = x[i]
    return min

def min_nonzero2(x):
    """
    Replaces zero values in an array with the minimum non-zero value.

    Parameters
    ----------
    x : array-like
        Array in which to replace zero values.

    Returns
    -------
    numpy.ndarray
        Array with zero values replaced by the minimum non-zero value.
    """
    x = np.abs(x)
    zeros = np.argwhere(x == 0)
    nonzeros = np.nonzero(x)[0]
    # idx = 0
    if nonzeros.size == 0:
        return x
    else:
        for i in zeros:
            idx = np.argmin(np.abs(nonzeros - i))
            x[i] = x[nonzeros[idx]]
        return x

def a_dBFS(x):
    """
    Converts an array of signal values to dBFS (decibels relative to full scale).

    Parameters
    ----------
    x : array-like
        Array of signal values.

    Returns
    -------
    numpy.ndarray
        Array of signal values converted to dBFS.
    """
    x = np.abs(x)
    min = min_nonzero(x)
    for i in range(x.size):
        if x[i] == 0:
            x[i] = min
    return 20 * np.log10(x)


def a_dB(x):
    """
    Converts an array of signal values to dB (decibels) relative to the maximum value.

    Parameters
    ----------
    x : array-like
        Array of signal values.

    Returns
    -------
    numpy.ndarray
        Array of signal values converted to dB.
    """
    
    x = min_nonzero2(x)
    maximum = np.max(x)
    return 20 * np.log10(x / maximum)
# =============================================================================
# DESCRIPTORS
# =============================================================================

## RT
def calc_RT20(smoothed_IR, fs):
    """
    Calculates the reverberation time (RT20) of a room impulse response.

    Parameters
    ----------
    smoothed_IR : array-like
        Smoothed room impulse response.
    fs : int
        Sampling frequency of the room impulse response.

    Returns
    -------
    float
        Reverberation time (RT20) in seconds.
    """

    t = np.arange(0, len(smoothed_IR)/fs, 1/fs)
    maxval = np.max(smoothed_IR)
    i_start = int(np.argwhere(smoothed_IR >= maxval -5)[-1])
    i_end = int(np.argwhere(smoothed_IR >= maxval -25)[-1])
    p = least_squares(t[i_start:i_end], smoothed_IR[i_start:i_end])
    
    return round(-60 / p[0], 3)

def calc_RT30(smoothed_IR, fs):
    """
    Calculates the reverberation time (RT30) of a room impulse response.

    Parameters
    ----------
    smoothed_IR : array-like
        Smoothed room impulse response.
    fs : int
        Sampling frequency of the room impulse response.

    Returns
    -------
    float
        Reverberation time (RT30) in seconds
    """
    t = np.arange(0, len(smoothed_IR)/fs, 1/fs)
    maxval = np.max(smoothed_IR)
    i_start = int(np.argwhere(smoothed_IR >= maxval -5)[-1])
    i_end = int(np.argwhere(smoothed_IR >= maxval -35)[-1])
    p = least_squares(t[i_start:i_end], smoothed_IR[i_start:i_end])
    
    return round(-60 / p[0], 3)
    

def calc_EDT(smoothed_IR, fs):
    """
    Calculates the early decay time (EDT) of a room impulse response.

    Parameters
    ----------
    smoothed_IR : array-like
        Smoothed room impulse response.
    fs : int
        Sampling frequency of the room impulse response.

    Returns
    -------
    float
        Early decay time (EDT) in seconds
    """

    t = np.arange(0, len(smoothed_IR)/fs, 1/fs)
    maxval = np.max(smoothed_IR)
    i_start = int(np.argwhere(smoothed_IR >= maxval -1)[-1])
    i_end = int(np.argwhere(smoothed_IR >= maxval -10)[-1])
    p = least_squares(t[i_start:i_end], smoothed_IR[i_start:i_end])

    return round(-60 / p[0], 3)
    
    
## Clarity

def calc_C50(IR, fs):
    """
    Calculates the clarity index C50 of a room impulse response.

    Parameters
    ----------
    IR : array-like
        Room impulse response.
    fs : int
        Sampling frequency of the room impulse response.

    Returns
    -------
    float
        Clarity index C50 in decibels (dB).
    """

    t50 = int(0.05 * fs)
    IR = IR ** 2 # Raise the IR to the second power
    C50 = 10 * np.log10(np.cumsum(IR[:t50])  / np.cumsum(IR[t50:])) # Calculate the C50
        
    return round(C50, 3)

def calc_C80(IR, fs):
    """
    Calculates the clarity index C80 of a room impulse response.

    Parameters
    ----------
    IR : array-like
        Room impulse response.
    fs : int
        Sampling frequency of the room impulse response.

    Returns
    -------
    float
        Clarity index C80 in decibels (dB).
    """
    t80 = int(0.08 * fs)
    C80 = 10 * np.log10(np.cumsum(IR[:t80])  / np.cumsum(IR[t80:])) # Calculate the C80
    
    return round(C80, 3)

def c_parameters(filtered_IR, fs):
    
    N50 = int(.05 * fs)
    N80 = int(.08 * fs)
    C50 = 10 * np.log10(np.sum(filtered_IR[:N50]**2)  / np.sum(filtered_IR[N50:]**2))
    C80 = 10 * np.log10(np.sum(filtered_IR[:N80]**2)  / np.sum(filtered_IR[N80:]**2))
    return round(C50, 3), round(C80, 3)

## Tt & EDTt

def idx_Tt(filtered_IR):
    """
    Calculates the Transition Time index for a filtered impulse response.

    Parameters
    ----------
    filtered_IR : array-like
        Filtered impulse response.

    Returns
    -------
    int
        Transition Time index.
    """
    
    index = np.argmin(np.cumsum(filtered_IR**2) <= 0.99 * np.sum(filtered_IR**2))
    return index

def calc_EDTt(filtered_IR, smoothed_IR, fs):
    """
    Calculates the Early Decay Time (EDT) and Transition Time (Tt) for a filtered impulse response.

    Parameters
    ----------
    filtered_IR : array-like
        Filtered impulse response.
    smoothed_IR : array-like
        Smoothed impulse response.
    fs : int or float
        Sampling frequency of the impulse response.

    Returns
    -------
    EDTt : float
        Early Decay Time (EDT) in seconds.
    Tt : float
        Transition Time (Tt) in seconds.

    """
    
    index = idx_Tt(filtered_IR) # Transition Time index
    Tt = index / fs
    
    peak_idx = np.argmax(filtered_IR)
    
    N = len(smoothed_IR)
    
    t = np.arange(0, N/fs, 1/fs)
    
    p = least_squares(t[peak_idx : index], smoothed_IR[peak_idx : index])
    
    EDTt = -60 / p[0]
    
    return round(EDTt, 3), round(Tt, 3)

def calc_IACC_early(IR_L, IR_R, fs):

    """
    Calculates the early interaural cross-correlation (IACC) for a stereo impulse response.

    Parameters
    ----------
    IR_L : array-like
        Impulse response of the left channel.
    IR_R : array-like
        Impulse response of the right channel.
    fs : int or float
        Sampling frequency of the impulse response.

    Returns
    -------
    IACC_early : float
        Early interaural cross-correlation (IACC) value.

    """
    
    num = np.correlate(IR_L[0:int(0.8 * fs)], IR_R[0:int(0.08 * fs)])
    den = np.sqrt(np.sum(IR_L[0:int(0.8 * fs)] ** 2) * (np.sum(IR_R[0:int(0.8 * fs)] ** 2)))
    IACC_early = np.max(np.abs(num / den)) # Normalizad IACC_early
    
    return round(IACC_early, 3)