# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 12:36:57 2023

@author: Fujitsu-A556
"""

import numpy as np
import scipy.signal as signal
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
import soundfile as sf
import funciones


class Room_IR():
    def __init__(self):
        
        self.fs = None
        self.IR = None
        
    def load_sweep(self, file):
        
        self.sweep, self.fs = sf.read(file)
        
    def IR_from_ss(self, f0, f1, T, fs):

        l = len(self.rec) # Lenght of the IR

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
        signal_fft = np.fft.rfft(self.IR)
        inv_filt_fft = np.fft.rfft(inv_filt)
        
        IR_fft = signal_fft * inv_filt_fft # Obtain the FFT of the impulse response
        IR = np.fft.ifft(IR_fft)           # Inverse FFT to recover the temporary IR
        self.IR = IR / np.max(np.abs(IR))
        
        # Pase a decibeles
        self.IR_dB = funciones.a_dBFS(self.IR)
        
        # Se encuentra el inicio del impulso y se descartan las muestras
        # anteriores
        Nstart = int(np.argwhere(self.IR_dB>=-30)[0])
        T_end = 5 # segundos
        Nend = int(T_end * self.fs)

        self.IR = self.IR[Nstart:Nstart+Nend]
        
        # Filtra la señal entre las frecuencias extremo del sweep
        self.IR = funciones.filtro_pasabanda(self.IR, f0, f1, self.fs)
        
        # return self.IR, self.fs
        
    def get_bandas_validez(self, filtro):
        
        self.freq, self.espectro_sw = funciones.espectro(self.sweep, self.fs, dB=False)
        
        if filtro == 0:
            f_c = funciones.octavas()
            octava = 1
            
        elif filtro == 1:
            f_c = funciones.tercios()
            octava = 3
        
        nivel_x_banda = funciones.nivel_bandas(self.espectro_sw, f_c, 
                                               self.freq, octava, False)
        
        fmax = np.argmax(nivel_x_banda)
        self.f_validas = []
        for i in range(len(nivel_x_banda)):
                  
        # Si el sweep inicia en 20 Hz y termina en 20 kHz, a razón
        # de -10 dB/decada tiene una diferencia máxima de 30 dB entre extremos
            
            if nivel_x_banda[i] >= nivel_x_banda[fmax] - 32:
                if f_c[i] >= fmax:
        # Cae 3 dB por octava o 1 dB por tercio
                    if (octava == 1) and (nivel_x_banda[i] >= (nivel_x_banda[i-1] - 4)):
                        self.f_validas.append(f_c[i])
                    elif (octava == 3) and (nivel_x_banda[i] >= (nivel_x_banda[i-1] - 1.5)):
                        self.f_validas.append(f_c[i])


    def get_inverse_filt(self):
        
        self.freq, self.espectro_sw = funciones.espectro(self.sweep, self.fs, dB=True)
        
        # Debe encontrarse la duración del sweep sin el silencio final. Para ello
        # se elige una ventana de 100 ms y se calcula el valor RMS en cada una
        # para luego encontrar el silencio como la derivada (cambio abrupto de nivel)
        tiempo_ventana = 0.1
        RMS = funciones.rms_ventanas(self.sweep, self.fs, tiempo_ventana)
        delta_RMS = RMS[:-1] - RMS[1:]
        
        T_inv = (1 + np.argmax(delta_RMS)) * tiempo_ventana # Duracion del sweep
        N_inv = int(T_inv * self.fs)
        
        ifmax = np.argmax(self.espectro_sw)
        ifstart = 0
        
        # Indice de la frecuencia de inicio del sweep. Encuentra la frecuencia
        # en la cual la amplitud está 12 dB debajo del máximo (0 dB)
        
        for i in range(self.espectro_sw.size):
            
            if i < ifmax and self.espectro_sw[i] <= (self.espectro_sw[ifmax] - 12):
                
                ifstart = i
        
        # El espectro se puede simular con caída de 10 dB/decada desde la 
        # frecuencia de inicio:
        
        espectro_sim = - 10 * np.log10(self.freq[ifmax:] / self.freq[ifmax])
        
        delta = self.espectro_sw[ifmax:] - espectro_sim
        
        # Índice de Frecuencia máxima del sweep: primera muestra en la cual
        # el sweep es 6 dB menor al simulado
        
        if np.argwhere(delta < -6).size > 0:
            ifend = int(np.argwhere(delta < - 6)[0])
        else:
            ifend = self.freq[-1] # Si el sweep llegara hasta fs / 2 Hz
        
        
        t = np.linspace(0, T_inv, N_inv, endpoint=False)
        self.fstart = self.freq[ifstart]
        self.fend = self.freq[ifend]
        
        # Modulación de amplitud:
        M = np.exp(-t * np.log(self.fend/self.fstart) / T_inv)
        
        self.inverse_filter = M * self.sweep[N_inv-1::-1]
        self.inverse_filter /= max(self.inverse_filter)

    def load_recording(self, file):
        
        self.rec, self.fs_rec = sf.read(file)
        
    def linear_convolve(self, T_end=None):
        
        if self.fs == self.fs_rec:
            
            self.IR = signal.convolve(self.inverse_filter, self.rec, method='fft')
            
            self.IR /= max(abs(self.IR))
            
            # Pase a decibeles
            self.IR_dB = funciones.a_dBFS(self.IR)
            
            # Se encuentra el inicio del impulso y se descartan las muestras
            # anteriores
            Nstart = int(np.argwhere(self.IR_dB>=-30)[0])
            if T_end is None:
                T_end = 5 # segundos
            Nend = int(T_end * self.fs)
            

            self.IR = self.IR[Nstart:Nstart+Nend]
            
            # Filtra la señal entre las frecuencias extremo del sweep
            self.IR = funciones.filtro_pasabanda(self.IR, self.fstart, 
                                                  self.fend, self.fs)
    
    def load_extIR(self, file):
        self.IR, self.fs = sf.read(file)
        
        self.IR /= max(abs(self.IR))
        
        # Pase a decibeles
        self.IR_dB = funciones.a_dBFS(self.IR)
        
        # Se encuentra el inicio del impulso y se descartan las muestras
        # anteriores
        Nstart = int(np.argwhere(self.IR_dB>=-30)[0])
        T_end = 5 # segundos
        Nend = int(T_end * self.fs)
        
        self.IR = self.IR[Nstart:Nstart+Nend]
        
    
    def smooth_energyc(self, IR, M=None, mode=1):
        
        IR_filt = abs(signal.hilbert(IR))  # Módulo de la transformada de Hilbert
        
        if M is None:   # Si no se especifica ancho de ventana
            M = 2 * int(self.fs/self.fstart) # Ancho de la ventana = 2 * minima frecuencia
        if M % 2 == 0: M+=1 # Ancho de ventana impar (no se si hace falta pero porlas)
        
        if mode == 0:
            IR_filt = median_filter(IR_filt, M, mode='reflect')
            # IR_filt = funciones.mediamovil_rcsv(IR_filt, M)
        elif mode == 1:
            IR_filt = signal.savgol_filter(IR_filt, M, 1)
        else:
            raise RuntimeError('"mode" must be either 0 (moving average) or 1 (savgol).')
        
        
        return funciones.a_dB(IR_filt)

        
    def crosspoint_lundeby(self, Ec, ventanas=10):
        '''
        Permite obtener el punto de cruce y la pendiente de decaimiento tardío.

        Parameters
        ----------
        Ec : 1darray
            Curva de decaimiento energético, en dB.
        ventanas : int, optional
            Cantidad de ventanas por cada 10 dB de caída para el filtrado de Ec.
            The default is 10.

        Returns
        -------
        slope : float
            Pendiente tardía de decaimiento energético.
        i_c2 : int
            Indice del punto de cruce, de forma tal que tc = t[i_c2].

        '''
        
        # 1) estimar ruido en el ultimo 10% de la señal
        # noise = 20 * np.log10(funciones.rms(self.IR[int(0.9*self.IR.size):]))
        noise = np.mean(Ec[int(0.9*Ec.size):]) # equivalente

        # 2) regresion lineal #1 desde 0 dB a noise + 5 dB
        t = np.arange(0, Ec.size/self.fs, 1/self.fs)
        i_end = int(np.argwhere(Ec > (noise + 10))[-1])
        p = funciones.cuad_min(t[:i_end], Ec[:i_end])

        reg1 = np.polyval(p, t)

        # Punto de cruce premilinar es t[i_c]
        i_c = int(np.argwhere(reg1 >= noise)[-1])

        # 3) Filtro de media movil con 3 a 10 ventanas cada 10 dB de caida
        N = -10 * self.fs / p[0]  # cantidad de muestras para 10 dB de caída
        N = int(N // ventanas) # tamaño de ventanas
        if N%2 == 0: N += 1
        # if N <= 1:
        #     raise RuntimeError('Ventana muy chica')
        # Ec2 = median_filter(Ec, N)
        # Ec2 = signal.savgol_filter(Ec, N, 1)
        Ec2 = funciones.mediamovil_rcsv(Ec, N)

        
        # 4) Estimación de noise en un punto posterior a tc (5 - 10 dB), con 10% de señal
        
        i_c2 = i_c
        delta = 1
        it = 0
        
        while delta > 0.01 and it < 10:
            
            noise2 = np.mean(Ec2[i_c2:]) # Ruido posterior al cruce
            
            
            i_end2 = int(np.argwhere(Ec2 >= (noise2 + 5))[-1])
            i_start2 = int(np.argwhere(Ec2 >= (noise2 + 25))[-1])
            
            # 5) Estimar pendiente desde 25 hasta 5 dB por sobre piso de ruido:
            
            p2 = funciones.cuad_min(t[i_start2:i_end2], Ec2[i_start2:i_end2])
            
            reg2 = np.polyval(p2, t)
            i_c3 = int(np.argwhere(reg2 >= noise2)[-1]) # Nuevo PUNTO DE CRUCE
            
            delta = abs(t[i_c3] - t[i_c2])
            i_c2 = i_c3
            slope = p2[0]
            it +=1
        return slope, i_c2

    def schroeder_int(self, IR, N_c):
        '''
        Obtiene la curva de decaimiento energético mediante la integral 
        de Schroeder.

        Parameters
        ----------
        N_c : int
            Índice del punto de cruce.

        Returns
        -------
        array
            Curva de decaimiento energético.

        '''
        IR = IR ** 2
        sch = np.cumsum(IR[N_c::-1])
        sch /= np.max(sch)
        return 10 * np.log10(sch[::-1])
    
    def get_ETC(self, impfilt, fs, method):
        
        Ec = self.smooth_energyc(impfilt)
        
        if method == 0:
            _, N_c = self.crosspoint_lundeby(Ec)
            
            return self.schroeder_int(impfilt, N_c), Ec
        
        if method == 1:
            return Ec
    
    def calcula_ETC(self, param, imp, fs, filtro=0, method=0, N=5, reverse=0):
        '''
        Calcula el parámetro 'param' para una señal por octavas o tercios
        de octava.

        Parameters
        ----------
        param : function
            función que calcula el parámetro. Debe ser del tipo f(x, fs).
        imp : 1darray
            señal a procesar. Típicamente una respuesta al impulso.
        fs : int
            Frecuencia de muestreo de imp.
        filtro : str, optional
            Tipo de filtro a aplicar. Si filtro='octavas', calcula por octavas, 
            si es 'tercios', por tercios. The default is 'tercios'.

        Returns
        -------
        salida : list
            lista con los parámetros calculados para cada banda.

        '''
        if reverse == 1:
            imp = np.flip(imp)
        
        if filtro == 1: # tercios
            salida = []
            frecs = funciones.tercios()
            for i in range(len(frecs)):
                finf = 2 ** (-1/6) * frecs[i]
                fsup = 2 ** (1/6) * frecs[i]
                impfilt = funciones.filtro_pasabanda(imp, finf, fsup, fs, N)
                if reverse == 1:
                    impfilt = np.flip(impfilt)
                salida.append(param(impfilt, fs, method))
            return salida
        elif filtro == 0 :  # octavas
            salida = []
            frecs = funciones.octavas()
            for i in range(len(frecs)):
                finf = 2 ** (-1/2) * frecs[i]
                fsup = 2 ** (1/2) * frecs[i]
                impfilt = funciones.filtro_pasabanda(imp, finf, fsup, fs, N)
                if reverse == 1:
                    impfilt = np.flip(impfilt)
                salida.append(param(impfilt, fs, method))
            return salida
        
    def acoustical_parameters (self, filtered_IR):

          # Dictionary to store the parameters
          d = {"RT20":"",
               "RT30":"",
               "EDT":"",
               # "IACCEARLY":"",
               "C50":"",
               "C80":"",
               "Tt":"",
               # "EDTt":""
          }
          
          d["RT20"] = funciones.calc_RT20(filtered_IR, self.fs)
          d["RT30"] = funciones.calc_RT30(filtered_IR, self.fs)
          d["EDT"] = funciones.calc_EDT(filtered_IR, self.fs)
          # d["IACCEARLY"] = # ¿Cómo procesamos este parámetro?
          d["C50"] = funciones.calc_C50(filtered_IR, self.fs) 
          d["C80"] = funciones.calc_C80(filtered_IR, self.fs)
          d["Tt"] = funciones.calc_Tt(filtered_IR, self.fs)
          # d["EDTt"] = ""
          return d
          
    def get_acparam(self, ETC, method=0):
        
        results = []
        
        if method == 0:
            schr = []
            mmfilt = []
            for i in range(len(ETC)):
                schr.append(ETC[i][0])
                mmfilt.append(ETC[i][1])
            mmfilt = np.array(mmfilt)
            
            for i in range(len(schr)):
                results.append(self.acoustical_parameters(schr[i]))  
        
        elif method == 1:
            for i in range(len(ETC)):
                results.append(self.acoustical_parameters(ETC[i]))
            schr = None
        
        return results, schr, mmfilt