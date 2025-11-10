import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
from pathlib import Path

input_folder = Path('recordings')
output_folder = Path('spectrograms')

# wavfile.read returns the sample rate and the data
for wav_file in input_folder.glob('*.wav'):
    samplerate, data = wavfile.read('recordings/0_george_0.wav')
    # signal.spectrogram creates a spectrogram using Fourier transform
    # arguments: data = time series of measured values, fs = sample rate
    frequencies, times, spectrogram = signal.spectrogram(data,samplerate)

    plt.pcolormesh(times, frequencies, 10 * np.log10(spectrogram + 1e-9))
    plt.xticks(np.linspace(times[0],times[-1], 5)) 
    plt.yticks([0, 1000, 2000, 3000, 4000])
    plt.ylabel('Frekvencija [Hz]')
    plt.xlabel('Time [sec]')

    output_filename = output_folder / (wav_file.stem + '.png')

    plt.savefig(output_filename)
    print("sacuvan fajl ", output_filename)
    plt.close()

print("gotovo!")