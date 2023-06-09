# RIR-processing
Room Impulse Response Python processing software development.

**RIR-processing** is an open source software for obtaining Room Impulse Responses (RIRs) from measurments using logarithmic sine sweep excitation and calculate the system's acoustical parameters. 

Libraries required:

- `numpy`
- `scipy.signal`
- `matplotlib`
- `soundfile`
- `PyQt5`

To execute the application run `gui.py` file. The settings interface will show up. 

![Settings user interface](/images/SetupUI.png)

1. Load the `.wav`files to process choosing one of the three options:
   - For sweep files and sweep recordings, select "Load Sweep" and "Load Recording".
   - If the sweep information is provided, check "Generate Sweep" and fill the corresponding fields.
   - If the file is a processed RIR or an impulse response obtained by other method (such as bursts or claps), check "Processed RIR" and open the file in "Load RIR".
2. Choose the filter type.
3. Choose the smoothing method. "Schroeder" is the Scrhoeder integral, while "Moving Average" is a combined technique for obtaining the Decay Curve. For further information see the the [written report]().
4. Select the aditional settings. "Noise Compensation" is the Lundeby's noise compensation method for the Schroeder Integral. "Reverse Filtering" flips in time the impulse prior to applying the filter, and then reverses de filtered impulse to avoid modifying the decay with the filter impulse response. 
5. Press "Calculate Parameters". A window with the results will show up.
6. Select a cell in a frequency band column to display the corresponding curves. Save the image with "Save Graph".
7. The red columns (if any) displays the bands in which results may not be valid. Save the table to `.csv` with "Save Data". 

![Results display](/images/ResultUI.png)

[Written report](README.md) avaiable for further information. 
