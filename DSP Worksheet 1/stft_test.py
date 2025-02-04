import numpy as np
from scipy.io import wavfile
from scipy import signal
import scipy
import matplotlib.pyplot as plt

# read from storage
filename = "TwoNote_DPA_31.wav"
fs, data = wavfile.read(filename)

# window parameters in milliseconds
window_length_ms = 30
window_step_ms = 5

window_step = 512
window_length = 1024
window_count = int(np.floor((data.shape[0]-window_length)/window_step)+1)

# windowing function
windowing_fn = signal.windows.hann(window_length)

spectrogram_matrix = np.zeros([window_length,window_count],dtype=complex)
for window_ix in range(window_count):    
    data_window = np.multiply(windowing_fn,data[window_ix*window_step+np.arange(window_length)])
    spectrogram_matrix[:,window_ix] = scipy.fft.fft(data_window)
    
fft_length = int((window_length+1)/2)

# plt.pcolormesh(1024, 10 * np.log10(spectrogram_matrix))

t = np.arange(0.,len(data),1.)/fs
plt.figure(figsize=(12,8))
plt.subplot(311)
plt.plot(t,data)
plt.xlim([0,t[-1]])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Speech waveform')
plt.subplot(312)
plt.imshow(20*np.log10(0.2+np.abs(spectrogram_matrix[range(fft_length),:])),origin='lower',aspect='auto',extent=[0.,len(data)/fs,0.,fs/2000])
#plt.xticks(np.arange(0.,len(data)/fs,0.5));
plt.xlabel('Time (s)')
#plt.yticks(np.arange(0,fft_length,fft_length*10000/fs));
plt.ylabel('Frequency (kHz)');
plt.title('Speech spectrogram')
plt.subplot(313)
plt.imshow(20*np.log10(0.2+np.abs(spectrogram_matrix[range(fft_length//2),:])),origin='lower',aspect='auto',extent=[0.,len(data)/fs,0.,fs/4000])
plt.xlabel('Time (s)')
#plt.yticks(np.arange(0,fft_length,fft_length*10000/fs));
plt.ylabel('Frequency (kHz)');
plt.title('Speech spectrogram zoomed in to lower frequencies')
plt.tight_layout()
plt.show()