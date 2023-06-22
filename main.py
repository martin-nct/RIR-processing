# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 12:36:57 2023

@author: Fujitsu-A556
"""

import numpy as np
import scipy.signal as signal
import soundfile as sf
import funciones


class Room_IR():
    def __init__(self):
        
        self.fs = None
        self.IR = None
        self.is_binaural = False
        self.IR_L = None
        self.IR_R = None
        
    def load_sweep(self, file):
        
        self.sweep, self.fs = sf.read(file)
        
    def IR_from_ss(self, rec, f0, f1, T, fs):

        l = len(rec) # Lenght of the IR

        # Generate logarithmic sine sweep
        t = np.linspace(start=0, stop=T, num=int(T * fs), endpoint=False)   # Time vector
        sine_sweep = signal.chirp(t=t, f0=f0, f1=f1, t1=T, method="logarithmic") # Sine sweep
        self.sweep = sine_sweep
        self.fs = fs
        
        # Based on: "Simultaneous Measurement of Impulse Response and Distortion With a Swept-Sine Technique by A. Farina"
        M = 1/(2 * np.pi * np.exp(t * np.log(f1 / f0) / T)) # Modulation factor
        inv_filt = M * sine_sweep[::-1] # Inverse filter for sine sweep
        inv_filt /= max(abs(inv_filt))  # Normalize
        inv_filt = np.pad(array=inv_filt, pad_width=(0, int(l - T * fs)), mode="constant") # Zero padding
        
        # Processing in frequency domain
        signal_fft = np.fft.rfft(rec)
        inv_filt_fft = np.fft.rfft(inv_filt)
        
        IR_fft = signal_fft * inv_filt_fft  # Obtain the FFT of the impulse response
        IR = np.fft.irfft(IR_fft)           # Inverse FFT to recover the temporary IR
        return IR / np.max(np.abs(IR))
        
    def get_valid_bands(self, filter):
        
        self.freq, self.spectrum_sw = funciones.spectrum(self.sweep, self.fs, dB=False)
        
        if filter == 0:
            f_c = funciones.octaves()
            octave = 1
            
        elif filter == 1:
            f_c = funciones.thirds()
            octave = 3
        
        level_per_band = funciones.nivel_bandas(self.spectrum_sw, f_c, 
                                               self.freq, octave, False)
        
        fmax = np.argmax(level_per_band)
        self.valid_freqs = []
        for i in range(len(level_per_band)):
                  
        # If the sweep starts at 20 Hz and ends at 20 kHz, with a rate
        # of -10 dB/decade, it has a maximum difference of 30 dB between the extremes.
            
            if level_per_band[i] >= level_per_band[fmax] - 32:
                if f_c[i] >= fmax:
        # It drops 3 dB per octave or 1 dB per third.
                    if (octave == 1) and (level_per_band[i] >= (level_per_band[i-1] - 4)):
                        self.valid_freqs.append(f_c[i])
                    elif (octave == 3) and (level_per_band[i] >= (level_per_band[i-1] - 1.5)):
                        self.valid_freqs.append(f_c[i])
            
        while self.fstart > self.valid_freqs[0]:
            self.valid_freqs.pop(0)

    def get_inverse_filt(self):
        
        self.freq, self.spectrum_sw = funciones.spectrum(self.sweep, self.fs, dB=True)
        
        # The duration of the sweep without the final silence must be determined.
        # To achieve this, a window of 100 ms is chosen, and the RMS value is calculated for each window.
        # The silence is then identified as the derivative (abrupt change in level).

        time_window = 0.1
        RMS = funciones.rms_windows(self.sweep, self.fs, time_window)
        delta_RMS = RMS[:-1] - RMS[1:]
        
        T_inv = (1 + np.argmax(delta_RMS)) * time_window # Sweep duration
        N_inv = int(T_inv * self.fs)
        
        ifmax = np.argmax(self.spectrum_sw)
        ifstart = 0
        
        # Starting frequency index of the sweep.
        # Finds the frequency at which the amplitude is 12 dB below the maximum (0 dB).
        
        for i in range(self.spectrum_sw.size):
            
            if i < ifmax and self.spectrum_sw[i] <= (self.spectrum_sw[ifmax] - 12):
                
                ifstart = i
        
       
        # The spectrum can be simulated with a slope of 10 dB/decade starting from the initial frequency.
        spectrum_sim = - 10 * np.log10(self.freq[ifmax:] / self.freq[ifmax])
        
        delta = self.spectrum_sw[ifmax:] - spectrum_sim
        
    
        # Maximum frequency index of the sweep: the first sample at which the sweep is 6 dB lower than the simulated sweep.
        
        if np.argwhere(delta < -6).size > 0:
            ifend = int(np.argwhere(delta < - 6)[0])
        else:
            ifend = self.freq[-1] # If the sweep reached fs / 2 Hz.
        
        
        t = np.linspace(0, T_inv, N_inv, endpoint=False)
        self.fstart = self.freq[ifstart]
        self.fend = self.freq[ifend]
        
        # Amplitude modulation:
        M = np.exp(-t * np.log(self.fend/self.fstart) / T_inv)
        
        self.inverse_filter = M * self.sweep[N_inv-1::-1]
        self.inverse_filter /= max(self.inverse_filter)

    def load_recording(self, file):
        
        self.rec, self.fs_rec = sf.read(file)
        if self.rec.ndim > 1:
            self.is_binaural = True
            self.rec_L = self.rec[:, 0] # Left channel of the IR
            self.rec_R = self.rec[:, 1] # Right channel of the IR
            self.rec = self.rec_L
            
        
    def linear_convolve(self, rec):
        
        if self.fs == self.fs_rec:
            
            IR = signal.convolve(self.inverse_filter, rec, method='fft')
            
            IR /= max(abs(IR))
            
            # Convert to dB
            # self.IR_dB = funciones.a_dBFS(self.IR)
            
            return IR

    def IR_trim(self, IR, T_end=None):
        # The start of the impulse is found and the previous samples are discarded.
        if T_end is None:
            T_end = 5 # segs
        length = int(T_end * self.fs)
            
        N_start = np.argmax(np.abs(IR))
        N_correc = np.argwhere(np.abs(IR)>=0.1) # -20 dB 
        delta = N_start - N_correc[0]
        while delta > 200:          # The start is within 200 samples from the maximum.
            N_correc = N_correc[1:]
            delta = N_start - N_correc[0]
        N_correc = int(N_correc[0])
        IR = IR[N_correc:]
        if IR.size > length:
            IR = IR[:length]
        
        return IR

    def load_extIR(self, file):
        self.IR, self.fs = sf.read(file)
        
        if self.IR.ndim > 1:
            self.is_binaural = True
            self.IR_L = self.IR[:, 0] # Left channel of the IR
            self.IR_R = self.IR[:, 1] # Right channel of the IR
            self.IR = (self.IR_L + self.IR_R) / 2 # Combine both channels
            maxval = max(max(abs(self.IR_L)), max(abs(self.IR_R)))
            self.IR_L /= maxval
            self.IR_R /= maxval
        self.IR /= max(abs(self.IR)) 
        
        
    
    def smooth_energyc(self, IR, M=2400, mode=0):
        
        IR_filt = abs(signal.hilbert(IR))  # Hilbert Transform Module
        
        if M is None:   # If the window width is not specified.
            M = 2 * int(self.fs/self.fstart) # Window width = 2 * minimum frequency
        if M % 2 == 0: M+=1 # Odd window width
        
        if mode == 0:
            IR_filt = funciones.maf_rcsv(IR_filt, M)
        elif mode == 1:
            IR_filt = signal.savgol_filter(IR_filt, M, 1)
        else:
            raise RuntimeError('"mode" must be either 0 (moving average) or 1 (savgol).')
        
        
        return funciones.a_dB(IR_filt)

        
    def crosspoint_lundeby(self, Ec, impfilt, windows=10):
        '''
        Obtains the crossover point and the late decay slope.

        Parameters
        ----------
        Ec : 1darray
            Energy decay curve, in dB.
        windows : int, optional
            Number of windows per 10 dB drop for filtering Ec.
            The default is 10.

        Returns
        -------
        slope : float
            Late decay slope.
        i_c2 : int
        Index of the crossover point, such that tc = t[i_c2]

        '''
        
        # 1)Estimate noise in the last 10% of the signal
        
        noise = np.mean(Ec[int(0.9 * Ec.size):]) # 

        # 2) Linear regression #1 from 0 dB to noise + 5 dB
        t = np.arange(0, Ec.size/self.fs, 1/self.fs)
        i_end = int(np.argwhere(Ec > (noise + 10))[-1])
        p = funciones.least_squares(t[:i_end], Ec[:i_end])

        reg1 = np.polyval(p, t)
        
        # Preliminary crossing point is t[i_c]
        i_c = int(np.argwhere(reg1 >= noise)[-1])
        
        # 3) Moving average filter with 3 to 10 windows for every 10 dB of decay
        N = -10 * self.fs / p[0]  # Number of samples for a 10 dB decay
        N = int(N // windows) # windows size
        if N%2 == 0: N += 1
        if N <= 1:
            raise RuntimeError('Too small window')

        Ec2 = funciones.maf_rcsv(Ec, N)

        
        # 4) Noise estimation at a point after tc (5 - 10 dB), with 10% of the signal
        
        i_c2 = i_c
        delta = 1
        it = 0
        
        while delta > 0.01 and it < 10:
            
            noise2 = np.mean(Ec2[i_c2:]) # Post-crossover noise estimation
            
            i_end2 = int(np.argwhere(Ec2 >= (noise2 + 5))[-1])
            i_start2 = int(np.argwhere(Ec2 >= (noise2 + 25))[-1])
            
            # 5) Estimate slope from 25 to 5 dB above the noise floor.
            
            p2 = funciones.least_squares(t[i_start2:i_end2], Ec2[i_start2:i_end2])
            
            reg2 = np.polyval(p2, t)
         
            i_c3 = int(np.argwhere(reg2 >= noise2)[-1]) # New cross-point
            
            delta = abs(t[i_c3] - t[i_c2])
            i_c2 = i_c3
            it +=1
            
            
        C = max(impfilt) * 10 ** (p[1] / 10) * np.exp(p[0]/10/np.log10(np.exp(1))*i_c2) / (
            -p[0] / 10 / np.log10(np.exp(1)))
        # C = 0
        return C, i_c2

    def schroeder_int(self, IR, N_c, C):
        '''
        Obtains the energy decay curve using the Schroeder integral.

        Parameters
        ----------
        IR : 1d-array
            Room impulse response.
        N_c : int
            Index of the crossover point.
        C : float
            Lundeby noise compensation.

        Returns
        -------
        array
            Energy decay curve.

        '''
        IR = IR ** 2
        sch = np.cumsum(IR[N_c::-1] + C)
        sch /= (np.max(sch) + C)
        return 10 * np.log10(sch[::-1])
    
    def get_ETC(self, impfilt, fs, frec):
        """
        Calculates the Energy-Time Curve (ETC) for a given impulse response.

        Args:
        impfilt (array-like): The impulse response.
        fs (float): The sampling frequency in Hz.
        frec (float): The center frequency in Hz.

        Returns:
        tuple: A tuple containing the following:
            - ETC (array-like): The Energy-Time Curve.
            - Ec (array-like): The smoothed energy curve.
            - impfilt (array-like): The impulse response.
        """
        if frec <= 40: 
            M = int(self.fs / 20)
        elif frec >= 8000:
            M = int(self.fs / 4000)
        else:
            M = 2 * int(self.fs / frec) # Window equal to 2 periods of the center frequency
        
        Ec = self.smooth_energyc(impfilt, M)
        
        if self.method == 0:
            C, N_c = self.crosspoint_lundeby(Ec, impfilt)
            if not self.comp: C = 0
            return self.schroeder_int(impfilt, N_c, C), Ec, impfilt
        
        if self.method == 1:
            return Ec, impfilt
    
    def calcula_ETC(self, param, imp, fs, filter=0, N=5, reverse=0):
       
        if reverse == 1:
            imp = np.flip(imp)
        
        if filter == 1: # thirds
            salida = []
            frecs = funciones.thirds()
            for i in range(len(frecs)):
                finf = 2 ** (-1/6) * frecs[i]
                fsup = 2 ** (1/6) * frecs[i]
                if fsup >= fs:
                    break
                impfilt = funciones.bandpass_filter(imp, finf, fsup, fs, N)
                if reverse == 1:
                    impfilt = np.flip(impfilt)
                salida.append(param(impfilt, fs, frecs[i]))
            return salida
        elif filter == 0 :  # octaves
            salida = []
            frecs = funciones.octaves()
            for i in range(len(frecs)):
                finf = 2 ** (-1/2) * frecs[i]
                fsup = 2 ** (1/2) * frecs[i]
                if fsup >= fs/2:
                    break
                impfilt = funciones.bandpass_filter(imp, finf, fsup, fs, N)
                if reverse == 1:
                    impfilt = np.flip(impfilt)
                salida.append(param(impfilt, fs, frecs[i]))
            return salida
        
    def acoustical_parameters (self, smoothed_IR, filtered_IR, IR_R = None):

          # Dictionary to store the parameters
          d = {"RT20":"",
               "RT30":"",
               "EDT":"",
               "C50":"",
               "C80":"",
               "Tt":"",
                "EDTt":""
          }
          
          d["RT20"] = funciones.calc_RT20(smoothed_IR, self.fs)
          d["RT30"] = funciones.calc_RT30(smoothed_IR, self.fs)
          d["EDT"] = funciones.calc_EDT(smoothed_IR, self.fs) 
          d["C50"], d["C80"] = funciones.c_parameters(filtered_IR, self.fs)
          d["EDTt"], d['Tt'] = funciones.calc_EDTt(filtered_IR, smoothed_IR, self.fs)
          if self.is_binaural:
              d["IACCEARLY"] = funciones.calc_IACC_early(filtered_IR, IR_R, self.fs)
          return d
          
    def get_acparam(self, ETC):
        
        if self.is_binaural:
            ETC_L = ETC[0]
            ETC_R = ETC[1]
            
            results = []
            mmfilt = []
            
            if self.method == 0:
                schr = []
                for i in range(len(ETC_L)):
                    schr.append(ETC_L[i][0])
                    mmfilt.append(ETC_L[i][1])
                    results.append(self.acoustical_parameters(schr[i], ETC_L[i][2], ETC_R[i][2]))
                
            
            elif self.method == 1:
                for i in range(len(ETC_L)):
                    mmfilt.append(ETC_L[i][0])
                    results.append(self.acoustical_parameters(ETC_L[i][0], ETC_L[i][1], ETC_R[i][1]))
                schr = None
        else:
                
            results = []
            mmfilt = []
            
            if self.method == 0:
                schr = []
                for i in range(len(ETC)):
                    schr.append(ETC[i][0])
                    mmfilt.append(ETC[i][1])
                    results.append(self.acoustical_parameters(schr[i], ETC[i][2]))
                
            
            elif self.method == 1:
                for i in range(len(ETC)):
                    mmfilt.append(ETC[i][0])
                    results.append(self.acoustical_parameters(ETC[i][0], ETC[i][1]))
                schr = None
            
        return results, schr, mmfilt