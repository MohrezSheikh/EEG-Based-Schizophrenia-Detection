!pip install mne
!pip install pyedflib
!pip install pywt
!pip install keras
!pip install tensorflow
!pip install matplotlib
!pip install --upgrade tensorflow
!pip install scikeras

import os
import numpy as np
import pandas as pd
import mne
import pywt
import matplotlib
import matplotlib.pyplot as plt
# from mne.io import RawArray, read_raw_edf
from mne.io import concatenate_raws as  mne
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from google.colab import drive
from PIL import Image
from sklearn.model_selection import train_test_split


def DataPreProcess(raw):
    # 1. Read EDF
    raw = mne.io.read_raw_edf(edf_file, preload=True)

    #2. Remove noise and false differences (other denoising methods can be added as needed, such as ICA, etc.)
    raw.notch_filter([50, 60], fir_design='firwin')

    # 3. Create ICA object
    ica = mne.preprocessing.ICA(n_components=19, random_state=97, max_iter=800)
    ica.fit(raw)

    # Apply the components removed by ICA to the data
    ica.apply(raw)

    # 4. Select the channel of interest
    raw.pick_channels(channels)

    # 5. Filtering
    low_freq = 1  # Delta Waves和Theta Waves）
    high_freq = 30  #（Alpha Waves、Beta Waves和Gamma Waves）
    raw.filter(low_freq, high_freq)

    # 6. Re-reference to average reference
    raw.set_eeg_reference('average', projection=True)

    # # 7.（Baseline Correction）
    # raw.apply_baseline(baseline=(None, 0))

    # 8. Downsampling
    raw.resample(resample_rate)

def DivideData(raw,wavelet_time_freq_plots,window_size,sampling_rate):
    #Get filtered signal data
    data, times = raw[:]

    # Perform Continuous Wavelet Transform (CWT)
    wavelet_type = 'morl'  # Choose a wavelet for CWT
    scales = np.arange(1, 11)  # Define the scales for CWT
    cwt_coeffs, frequencies = pywt.cwt(data, scales, wavelet_type)

    # Compute wavelet coefficient energy
    energy = []
    for j in range(len(cwt_coeffs)):
        energy.append(np.sum(np.abs(cwt_coeffs[j])**2, axis=0))

    max_length = max(len(energy_level) for energy_level in energy)
    for j in range(len(energy)):
        padding = max_length - len(energy[j])
        energy[j] = np.pad(energy[j], (0, padding), 'constant')

    # normalized energy value
    scaler = MinMaxScaler()
    energy_normalized = scaler.fit_transform(np.vstack(energy).T).T

    # apply logarithmic scaling to energy values
    log_energy = np.log10(energy_normalized + 1)

    # Divide the energy value according to the size of the time window
    # sampling_rate = raw.info['sfreq']
    window_size_samples = int(window_size * sampling_rate)
    num_windows = (log_energy.shape[1] - window_size_samples) // window_size_samples + 1

    for i in range(num_windows):
        start_sample = i * window_size_samples
        end_sample = start_sample + window_size_samples

        # Extract the energy value of the current time window
        window_energy = log_energy[:, start_sample:end_sample]

        wavelet_time_freq_plots.append(window_energy)

    # Convert the list of wavelet time-frequency plots to numpy arrays
    wavelet_time_freq_plots = np.array(wavelet_time_freq_plots)
    return wavelet_time_freq_plots

def DrawAndSaveImages(wavelet_time_freq_plots,total_wavelet_plots,window_size,sampling_rate):
    total_file_wavelet_plots = 0
    # Get time axis and frequency axis
    time_axis = np.arange(wavelet_time_freq_plots.shape[1]) * window_size
    freq_axis = np.arange(wavelet_time_freq_plots.shape[2]) * (sampling_rate / (2 * wavelet_time_freq_plots.shape[2]))

    # Traverse the wavelet time-frequency graph of each time window and draw
    for i in range(wavelet_time_freq_plots.shape[0]):
        plt.figure(figsize=(10, 6))

        current_window_data = wavelet_time_freq_plots[i]

        time_axis_window = np.linspace(time_axis[0], time_axis[-1], current_window_data.shape[0])
        freq_axis_window = np.linspace(freq_axis[0], freq_axis[-1], current_window_data.shape[1])
        plt.imshow(current_window_data, aspect='auto', cmap='jet', extent=[time_axis_window[0], time_axis_window[-1], freq_axis_window[0], freq_axis_window[-1]])

        # Image extraction feature values for CNN do not need to display xy axis coordinates and colorbar
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency')
        plt.title(f'Wavelet Time-Frequency Plot - Window {i+1} - {edf_file}')
        plt.colorbar()
        # plt.axis('off')

        # Set the save path and image file
        save_path = ''
        prefix = os.path.basename(edf_file)[:1]
        print(prefix)
        if prefix == 'h':
            save_path = healthy_save_path
        if prefix == 's':
            save_path = schizophrenia_save_path
        file_name = f'{edf_file.split(".")[0]} - Window{i+1}.png'
        # plt.savefig(f'{save_path}/{file_name}')

        plt.show()

        plt.close()

        # Determine whether the edf_file belongs to healthy(0) or schizophrenia(1), and then add the corresponding label
        if edf_file.startswith('h'):
            y_labels.append(0)
        else:
            y_labels.append(1)
        total_file_wavelet_plots+=1
        total_wavelet_plots+=total_file_wavelet_plots


healthy_save_path = '/content/drive/MyDrive/eeg/healthy25-new'
schizophrenia_save_path = '/content/drive/MyDrive/eeg/schizophrenia25-new'


h_files=['h01.edf', 'h02.edf', 'h03.edf', 'h04.edf', 'h05.edf', 'h06.edf', 'h07.edf', 'h08.edf', 'h09.edf', 'h10.edf', 'h11.edf', 'h12.edf', 'h13.edf', 'h14.edf']
s_files = ['s01.edf', 's02.edf', 's03.edf', 's04.edf', 's05.edf', 's06.edf', 's07.edf', 's08.edf', 's09.edf', 's10.edf', 's11.edf', 's12.edf', 's13.edf', 's14.edf']
edf_files = s_files+h_files


# Define channel list
channels = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
"C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]


# Define time window size and overlap size in seconds
# Split the data into time windows of fixed length 25 seconds
window_size = 25

# No data overlap between adjacent time windows
overlap_size = 0

# Reduce the sample rate to 250 Hz
resample_rate = 250

# Store feature values and labels for all images
X_features = []
y_labels = []

total_wavelet_plots = 0
import mne

for edf_file in edf_files:
    # Store the wavelet time-frequency map of each time window
    wavelet_time_freq_plots = []
    raw = mne.io.read_raw_edf(edf_file, preload=True)
    sampling_rate = raw.info['sfreq']
    DataPreProcess(raw)
    wavelet_time_freq_plots = DivideData(raw,wavelet_time_freq_plots,window_size,sampling_rate)
    DrawAndSaveImages(wavelet_time_freq_plots,total_wavelet_plots,window_size,sampling_rate)
print(f"Total Wavelet Time-Frequency Plot count：{total_wavelet_plots}")