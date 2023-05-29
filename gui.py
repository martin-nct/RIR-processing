# -*- coding: utf-8 -*-
"""
Created on Fri May 26 12:21:19 2023

@author: Fujitsu-A556
"""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
# from matplotlib import colors
import soundfile as sf
# import sounddevice as sd
from PyQt5.QtWidgets import (QApplication, QWidget, QFileDialog, QPushButton, QGridLayout,
                             QLabel, QColorDialog, QGroupBox, QLineEdit, QVBoxLayout, 
                             QRadioButton, QCheckBox, QTableWidget, QTableWidgetItem)

from main import Room_IR
import funciones

class SetUpWindow(QWidget, Room_IR):
    def __init__(self):
        # super().__init__()
        super().__init__(windowTitle = 'Settings')
        
        # Imports:
        importBox = QGroupBox('Import')
        
        loadSweep = QPushButton('Load Sweep')
        loadSweep.clicked.connect(self.load_sweep)
        loadRec = QPushButton('Load Record')
        loadRec.clicked.connect(self.load_recording)
        
        box0 = QVBoxLayout()
        box0.addWidget(loadSweep)
        box0.addWidget(loadRec)
        importBox.setLayout(box0)
        
        #Generate Sweep 
        generateBox = QGroupBox('Generate Sweep')
        
        labelfmin = QLabel('Sart Freq')
        labelfmax = QLabel('End Freq')
        labeltime = QLabel('Time')
        labelfs = QLabel('Sample Rate')
        
        fmin = QLineEdit()
        fmax = QLineEdit()
        time = QLineEdit()
        fs = QLineEdit()
        
        box1 = QGridLayout()
        box1.addWidget(labelfmin, 0, 0, 1, 1)
        box1.addWidget(fmin, 0, 1, 1, 1)
        box1.addWidget(labelfmax, 0, 2, 1, 1)
        box1.addWidget(fmax, 0, 3, 1, 1)
        box1.addWidget(labeltime, 1, 0, 1, 1)
        box1.addWidget(time, 1, 1, 1, 1)
        box1.addWidget(labelfs, 1, 2, 1, 1)
        box1.addWidget(fs, 1, 3, 1, 1)
        generateBox.setLayout(box1)
        
        # Filter Box
        filterBox = QGroupBox('Filter settings')
        
        self.rb_octave = QRadioButton('Octave Bands')
        self.rb_third = QRadioButton('1/3 Octave Bands')
        
        self.rb_octave.setChecked(True)
        
        box2 = QVBoxLayout()
        box2.addWidget(self.rb_octave)
        box2.addWidget(self.rb_third)
        filterBox.setLayout(box2)
        
        # Smoothing box
        smoothBox = QGroupBox('Smoothing Method')
        
        self.rb_sch = QRadioButton('Schroeder')
        self.rb_mavg = QRadioButton('Moving Average')
        
        self.rb_sch.setChecked(True)
        
        box3 = QVBoxLayout()
        box3.addWidget(self.rb_sch)
        box3.addWidget(self.rb_mavg)
        smoothBox.setLayout(box3)
        
        # Aditional Settings
        setBox = QGroupBox('Aditional Settings')
        
        noisecomp = QCheckBox('Noise Compensation')
        noisecomp.setChecked(False)
        self.reverse = QCheckBox('Reverse filtering')
        self.reverse.setChecked(False)
        
        box4 = QVBoxLayout()
        box4.addWidget(noisecomp)
        box4.addWidget(self.reverse)
        setBox.setLayout(box4)
        
        # Calculate
        self.process = QPushButton('Calculate Parameters')
        self.process.clicked.connect(self.open_main)

        
        # Main Layout
        mainBox = QGridLayout(self)
        mainBox.addWidget(importBox, 0, 0, 1, 1)
        mainBox.addWidget(generateBox, 0, 1, 1, 1)
        mainBox.addWidget(filterBox, 1, 0, 1, 1)
        mainBox.addWidget(smoothBox, 1, 1, 1, 1)
        mainBox.addWidget(setBox, 2, 0, 1, 1)
        mainBox.addWidget(self.process, 2, 1, 1, 1)
        
    # def reFormat(self):
    #     self.process.setText('Calculating...')
    #     self.process.setFlat(True)
    #     if self.process.text() == 'Calculating...':
    #         self.open_main()
    
    def open_main(self):

        if self.rb_octave.isChecked():
            filtro = 0
        elif self.rb_third.isChecked():
            filtro = 1
        
        if self.rb_sch.isChecked():
            method = 0
        elif self.rb_mavg.isChecked():
            method = 1
        
        # if self.reverse.isChecked():
        #     reverse = 1
        # else: reverse = 0
        
        # f_validas = self.get_bandas_validez(filtro)
        self.get_inverse_filt()
        self.linear_convolve()
        ETC = self.calcula_ETC(self.get_ETC, self.IR, self.fs, filtro, method)
        
        self.results, self.schroeder, self.mmfilt = self.get_acparam(ETC, method)
        
        self.ResultW = ResultWindow(self.results, self.schroeder, 
                                    self.mmfilt, self.fs, filtro)
        self.ResultW.show()
        self.hide()
        
        # self.ResultW.config_table(self.results, filtro)
        
        # # Grafico
        # t = np.arange(0, self.mmfilt[0].size/self.fs, 1/self.fs)
        # if self.schroeder is not None:
        #     self.ResultW.linea_ETC.set_xdata(self.schroeder[5].size)
        #     self.ResultW.linea_ETC.set_ydata(t[:self.schroeder[5].size])
        
        # self.ResultW.linea_EDC.set_xdata(t)
        # self.ResultW.linea_EDC.set_ydata(self.mmfilt[5])
        # self.ResultW.linea_EDC.axes.set_xlim(t[0], t[-1])
        # self.ResultW.linea_EDC.axes.set_ylim(min(self.mmfilt[5])-5, 1)
        
        # self.ResultW.linea_ETC.figure.canvas.draw()
        
    def load_sweep(self):
        filtro = 'WAV (*wav);;FLAC (*flac)'
        ruta = QFileDialog.getOpenFileName(filter=filtro)[0]
        if ruta != '':
            self.sweep, self.fs = sf.read(ruta)
    
    def load_recording(self):
        filtro = 'WAV (*wav);;FLAC (*flac)'
        ruta = QFileDialog.getOpenFileName(filter=filtro)[0]
        if ruta != '':
            self.rec, self.fs_rec = sf.read(ruta)

class ResultWindow(QWidget):
    def __init__(self, results, schroeder, mmfilt, fs, filtro):
        super().__init__(windowTitle = 'Room Impulse Response')
        
        self.results = results
        self.schroeder = schroeder
        self.mmfilt = mmfilt
        self.fs = fs
        self.filtro = filtro
        # Gráfico
        
        self.t = np.arange(0, self.mmfilt[0].size/self.fs, 1/self.fs)
        
        self.fig = Figure(dpi=100)
        ax = self.fig.subplots()
        self.linea_ETC = ax.plot([], [], '--m', label='Schroeder Integral')[0]
        self.linea_EDC = ax.plot([], [], '-k', label='Moving Average')[0]
        ax.grid()
        ax.set_title('Energy Time Curve')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Level [dB]')
        ax.set_ylim([-1, 1])
        canvas = FigureCanvas(self.fig)
        
        if self.schroeder is not None:
            self.linea_ETC.set_ydata(self.schroeder[5])
            self.linea_ETC.set_xdata(self.t[:self.schroeder[5].size])
        
        self.linea_EDC.set_xdata(self.t)
        self.linea_EDC.set_ydata(self.mmfilt[5])
        self.linea_EDC.axes.set_xlim(self.t[0], self.t[-1])
        self.linea_EDC.axes.set_ylim(min(self.mmfilt[5])-5, 1)
        
        self.linea_ETC.figure.canvas.draw()
        ax.legend()
        
        self.save_gr = QPushButton('Save Graph')
        self.save_gr.clicked.connect(self.save_graph)
        
        # Tabla
        self.tableWidget = QTableWidget(7, 11)
        self.tableWidget.cellClicked.connect(self.clicked_cell)
        self.config_table(self.results, self.filtro)
        
        self.save_tab = QPushButton('Save Data')
        self.save_tab.clicked.connect(self.save_data)
        

        layout = QGridLayout(self)
        layout.addWidget(canvas, 0, 0, 1, 4)
        layout.addWidget(self.tableWidget, 1, 0, 2, 4)
        layout.addWidget(self.save_gr, 0, 5, 1, 1)
        layout.addWidget(self.save_tab, 1, 5, 1, 1)
        
    def config_table(self, data, filtro):
        
        numcols = len(data)
        numrows = len(data[0])
        self.tableWidget.setColumnCount(numcols)
        self.tableWidget.setRowCount(numrows)
        self.tableWidget.setVerticalHeaderLabels((list(data[0].keys())))
        freqs = funciones.labels_bandas(filtro, Hz=True)
        self.tableWidget.setHorizontalHeaderLabels(freqs)
        for column in range(numcols):
            for row in range(numrows):
                item = str(list(data[column].values())[row])
                self.tableWidget.setItem(row, column, QTableWidgetItem(item))
        
    def clicked_cell(self, row, column):
        # print('clicked!', row, column)
        
        if self.schroeder is not None:
            self.linea_ETC.set_ydata(self.schroeder[column])
            self.linea_ETC.set_xdata(self.t[:self.schroeder[column].size])
        
        self.linea_EDC.set_ydata(self.mmfilt[column])
        self.linea_EDC.axes.set_ylim(min(self.mmfilt[5])-5, 1)
        
        self.linea_ETC.figure.canvas.draw()
    
    def save_graph(self):
        # print('y si')
        
        ruta = QFileDialog.getSaveFileName(directory='figure.png', 
                                           filter='Portable Network Graphics (*png)')[0]
        if ruta != '':
            self.fig.savefig(ruta, format='png')
    
    def save_data(self):
        # print('por supuesto')
        ruta = QFileDialog.getSaveFileName(directory='data.csv', 
                                           filter='*csv')[0]
        if ruta != '':
            data = ['Freq [Hz]']
            data.append(funciones.labels_bandas(self.filtro))
        
        
        
app = QApplication([])
ventana = SetUpWindow()
# ventana = ResultWindow()
ventana.show()
app.exec_()