# -*- coding: utf-8 -*-
"""
Created on Fri May 26 12:21:19 2023

@author: Fujitsu-A556
"""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvas
import soundfile as sf
from PyQt5.QtWidgets import (QApplication, QWidget, QFileDialog, QPushButton, 
                             QGridLayout, QLabel, QGroupBox, QLineEdit,
                             QVBoxLayout, QRadioButton, QCheckBox,
                             QTableWidget, QTableWidgetItem, QSizePolicy)
from PyQt5.QtGui import QColor, QCursor, QIcon 
from PyQt5.QtCore import QTimer, Qt 

from main import Room_IR
import funciones

class SetUpWindow(QWidget, Room_IR):
    def __init__(self):
        super().__init__(windowTitle = 'RIR Processor by G. Caminos & M. Nocito - Settings')
        
        # Imports:
        importBox = QGroupBox('Import')
    
        self.loadSweep = QPushButton('Load Sweep')
        self.loadSweep.clicked.connect(self.load_sweep)
        self.loadRec = QPushButton('Load Recording')
        self.loadRec.clicked.connect(self.load_recording)
        self.from_RIR = QCheckBox('Processed RIR')
        self.from_RIR.clicked.connect(self.processed_RIR)
        self.from_RIR.setChecked(False)
        
        box0 = QVBoxLayout()
        box0.addWidget(self.loadSweep)
        box0.addWidget(self.loadRec)
        box0.addWidget(self.from_RIR)
        importBox.setLayout(box0)
        
        #Generate Sweep 
        self.generateBox = QGroupBox('Generate Sweep')
        self.generateBox.setCheckable(True)
        self.generateBox.setChecked(False)
        self.generateBox.clicked.connect(self.reFormat_loadSweep)
        
        labelfmin = QLabel('Sart Frequency [Hz]')
        labelfmax = QLabel('End Frequency [Hz]')
        labeltime = QLabel('Time [s]')
        labelfs = QLabel('Sample Rate [Hz]')
        
        self.fmin = QLineEdit()
        self.fmax = QLineEdit()
        self.time = QLineEdit()
        self.fs = QLineEdit()
        
        box1 = QGridLayout()
        box1.addWidget(labelfmin, 0, 0, 1, 1)
        box1.addWidget(self.fmin, 0, 1, 1, 1)
        box1.addWidget(labelfmax, 0, 2, 1, 1)
        box1.addWidget(self.fmax, 0, 3, 1, 1)
        box1.addWidget(labeltime, 1, 0, 1, 1)
        box1.addWidget(self.time, 1, 1, 1, 1)
        box1.addWidget(labelfs, 1, 2, 1, 1)
        box1.addWidget(self.fs, 1, 3, 1, 1)
        self.generateBox.setLayout(box1)
        
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
        
        self.noisecomp = QCheckBox('Noise Compensation')
        self.noisecomp.setChecked(False)
        self.reverse = QCheckBox('Reverse filtering')
        self.reverse.setChecked(False)
        
        box4 = QVBoxLayout()
        box4.addWidget(self.noisecomp)
        box4.addWidget(self.reverse)
        setBox.setLayout(box4)
        
        # Calculate
        self.process = QPushButton('Calculate Parameters',
                                   sizePolicy=QSizePolicy(QSizePolicy.Expanding, 
                                                          QSizePolicy.Expanding))
        self.process.setMaximumHeight(100)
        self.process.clicked.connect(self.reFormatProcess)
        
        # Main Layout
        mainBox = QGridLayout(self)
        mainBox.addWidget(importBox, 0, 0, 1, 1)
        mainBox.addWidget(self.generateBox, 0, 1, 1, 1)
        mainBox.addWidget(filterBox, 1, 0, 1, 1)
        mainBox.addWidget(smoothBox, 1, 1, 1, 1)
        mainBox.addWidget(setBox, 2, 0, 1, 1)
        mainBox.addWidget(self.process, 2, 1, 1, 1)
        
        # Footer / Credits
        footer_label = QLabel('Developed by <a href="mailto:mnocito8@gmail.com" style="text-decoration: none; color: #FFF;">M. Nocito</a> & <a href="mailto:caminos47501@estudiantes.untref.edu.ar" style="text-decoration: none; color: #FFF;">G. Caminos</a>')
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setOpenExternalLinks(True)
        footer_label.setWordWrap(True)
        footer_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        footer_label.setCursor(QCursor(Qt.PointingHandCursor))
    
        # Footer Layout
        footer_layout = QVBoxLayout()
        footer_layout.addWidget(footer_label)
        
        # Footer Container
        footer_container = QWidget()
        footer_container.setLayout(footer_layout)
        footer_container.setMaximumHeight(50)
        
        # Add Footer to Main Layout
        mainBox.addWidget(footer_container, 3, 0, 1, 2)
        
        mainBox.setSpacing(10)
        
        self.resize(419, 400)
        
           
    def reFormatProcess(self):
        self.process.setText('Calculating, please wait...')
        self.process.setFlat(True)
        
        timer = QTimer()
        timer.singleShot(500, self.open_main)
    
    def reFormat_loadSweep(self):
        self.loadSweep.setFlat(self.generateBox.isChecked())
        
    def processed_RIR(self):
        if self.from_RIR.isChecked():
            self.loadSweep.setFlat(True)
            self.loadRec.setText('Load RIR')
        else: 
            self.loadSweep.setFlat(False)
            self.loadRec.setText('Load Recording')
        
    def open_main(self):

        if self.rb_octave.isChecked():
            filtro = 0
        elif self.rb_third.isChecked():
            filtro = 1
        
        if self.rb_sch.isChecked():
            self.method = 0 # Schroeder reverse time integral 
        elif self.rb_mavg.isChecked():
            self.method = 1 # Moving Average Filter
        
        # Arroja resultados insactifactorios:
        # if self.reverse.isChecked():
        #     reverse = 1
        # else: reverse = 0
        
        if self.noisecomp.isChecked():
            self.comp = 1
        else: self.comp = 0
        
        if not self.from_RIR.isChecked():

            if self.generateBox.isChecked():
                
                self.fstart = float(self.fmin.text())
                self.fend = float(self.fmax.text())
                self.time = float(self.time.text())
                self.fs = int(self.fs.text())
                self.IR_from_ss(self.fstart, self.fend, self.time, self.fs)
           
            else:
                self.get_inverse_filt()
                self.linear_convolve()
            
            self.IR_trim(T_end=10)
            self.get_bandas_validez(filtro)
        

            
        ETC = self.calcula_ETC(self.get_ETC, self.IR, self.fs, filtro)
        
        self.results, self.schroeder, self.mmfilt = self.get_acparam(ETC)
        
        self.ResultW = ResultWindow(self.results, self.schroeder, 
                                    self.mmfilt, self.fs, filtro, self.f_validas)
        self.ResultW.show()
        self.ResultW.resize(1100, 680)
        self.ResultW.move(123, 47)
        self.hide()
        
    def load_sweep(self):
        filtro = 'WAV (*wav);;FLAC (*flac)'
        ruta = QFileDialog.getOpenFileName(filter=filtro)[0]
        if ruta != '':
            self.sweep, self.fs = sf.read(ruta)
    
    def load_recording(self):
        filtro = 'WAV (*wav);;FLAC (*flac)'
        ruta = QFileDialog.getOpenFileName(filter=filtro)[0]
        if ruta != '':
            if self.from_RIR.isChecked():
                self.load_extIR(ruta)
            else:
                self.rec, self.fs_rec = sf.read(ruta)

#%%
class ResultWindow(QWidget):
    def __init__(self, results, schroeder, mmfilt, fs, filtro, f_validas):
        super().__init__(windowTitle = 'RIR Processor by G. Caminos & M. Nocito - Room Impulse Response')
        
        self.results = results
        self.schroeder = schroeder
        self.mmfilt = mmfilt
        self.fs = fs
        self.filtro = filtro
        self.f_validas = f_validas
        
        # Gráfico
        
        self.t = np.arange(0, self.mmfilt[0].size/self.fs, 1/self.fs)
        
        self.fig = Figure(dpi=100)
        ax = self.fig.subplots()
        
        # Config. style and colors of plots
        self.fig.patch.set_facecolor('#0a0a23')
        ax.set_facecolor('#0a0a23')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        
        if self.schroeder is not None:
            self.linea_ETC = ax.plot([], [], '--m', label='Schroeder Integral', zorder=5)[0]
        self.linea_EDC = ax.plot([], [], '-w', label='Moving Average')[0]
        ax.grid()
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
        
        label = funciones.labels_bandas(self.filtro, True)[5]
        self.linea_EDC.axes.set_title(f'Energy Time Curve ({label})', color="#FFF")
        
        self.linea_EDC.figure.canvas.draw()
        ax.legend()
        
        self.save_gr = QPushButton('Save Graph')
        self.save_gr.clicked.connect(self.save_graph)
        
        # Tabla
        self.tableWidget = QTableWidget(7, 11)
        self.tableWidget.setBaseSize(100, 100)
        self.tableWidget.cellClicked.connect(self.clicked_cell)
        self.config_table(self.results, self.filtro)
        
        self.save_tab = QPushButton('Save Data')
        self.save_tab.clicked.connect(self.save_data)
        
        layout = QGridLayout(self)
        layout.addWidget(canvas, 0, 0, 1, 1)
        layout.addWidget(self.tableWidget, 1, 0, 1, 1)
        layout.addWidget(self.save_gr, 0, 1, 1, 1)
        layout.addWidget(self.save_tab, 1, 1, 1, 1)
        
        # Footer / Credits
        footer_label = QLabel('Developed by <a href="mailto:mnocito8@gmail.com" style="text-decoration: none; color: #FFF;">M. Nocito</a> & <a href="mailto:caminos47501@estudiantes.untref.edu.ar" style="text-decoration: none; color: #FFF;">G. Caminos</a>')
        footer_label.setAlignment(Qt.AlignCenter)
        footer_label.setOpenExternalLinks(True)
        footer_label.setWordWrap(True)
        footer_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        footer_label.setCursor(QCursor(Qt.PointingHandCursor))
    
        # Footer Layout
        footer_layout = QVBoxLayout()
        footer_layout.addWidget(footer_label)
        
        # Footer Container
        footer_container = QWidget()
        footer_container.setLayout(footer_layout)
        footer_container.setMaximumHeight(50)
        
        # Add Footer to Main Layout
        layout.addWidget(footer_container, 3, 0, 1, 2)
        
        layout.setSpacing(10)
        
        
        
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
        
        #Colorea las columnas (bandas) en las cuales no es válido el resultado
        
        if filtro == 0:
            center = funciones.octavas()
        elif filtro == 1:
            center = funciones.tercios()
        
        for i, f in enumerate(center):
            
            if f not in self.f_validas:
                self.set_column_color(i, QColor(255, 0, 0, 127))
        
    def set_column_color(self, column, color):
        for j in range(self.tableWidget.rowCount()):
            self.tableWidget.item(j, column).setBackground(color)
        
    def clicked_cell(self, row, column):
        
        if self.schroeder is not None:
            self.linea_ETC.set_ydata(self.schroeder[column])
            self.linea_ETC.set_xdata(self.t[:self.schroeder[column].size])
        
        self.linea_EDC.set_ydata(self.mmfilt[column])
        self.linea_EDC.axes.set_ylim(min(self.mmfilt[5])-5, 1)
        
        label = funciones.labels_bandas(self.filtro, True)[column]
        self.linea_EDC.axes.set_title(f'Energy Time Curve ({label})', color="#FFF")
        
        self.linea_EDC.figure.canvas.draw()
    
    def save_graph(self):
        ruta = QFileDialog.getSaveFileName(directory='figure.png', 
                                           filter='Portable Network Graphics (*png)')[0]
        if ruta != '':
            self.fig.savefig(ruta, format='png')
    
    def save_data(self):
        ruta = QFileDialog.getSaveFileName(directory='data.csv', 
                                           filter='*csv')[0]
        if ruta != '':
            
            numcols = len(self.results)
            numrows = len(self.results[0])
    
            data = np.zeros((numrows+1, numcols+1), dtype='<U9')
    
            data[0, 0] = 'Freq [Hz]'
            data[0, 1:] = funciones.labels_bandas(self.filtro)
    
            for row in range(numrows):
                for column in range(numcols+1):
                    if column == 0:
                        data[row+1, column] = (list(self.results[0].keys())[row])
                    else:
                        data[row+1, column] = (list(self.results[column-1].values())[row])
    
            np.savetxt(ruta, data, fmt='%s', delimiter=',')


if __name__ == '__main__':
    app = QApplication([])
    
    # Styles
    with open('styles.css', 'r') as file:
        style_sheet = file.read()
    app.setStyleSheet(style_sheet)
    
    # Icon
    icon = QIcon('RIR.ico')

    ventana = SetUpWindow()
    ventana.setWindowIcon(icon)
    ventana.show()
    app.exec_()