# 4 Short-Time Fourier Transform

## 4.1 Load Wave File
from scipy.io import wavfile

filepath = "TwoNote_DPA_31.wav"
samplerate, data = wavfile.read(filepath)

print(f"Properties of File - {filepath}")

# duration = num samples / sample rate, rounded to 3 places
print(f"Duration: {(len(data) / samplerate):.3f} seconds") 

# the number of channels is determined by the number of dimensions of the data
print(f"Number of channels: {len(data.shape)}")

print(f"Sample rate: {samplerate} Hz")

print(f"Signal length in samples: {len(data)}")

print(f"Bit depth: {data.dtype.itemsize * 8}")


## 4.2 Calculate the STFT
from scipy.fft import fft
from scipy.signal.windows import hann
import numpy as np

def stft(input_signal, window_function, window_size, hop_size):

    # pad the input signal with 0s at the end
    input_signal_pad_length = (window_size - (len(input_signal) % hop_size)) % hop_size
    input_signal_padded = np.pad(input_signal, (0, input_signal_pad_length))

    num_frames = int((len(input_signal_padded) - window_size) / hop_size + 1)

    # fill the stft matrix with 0s
    stft_matrix = np.zeros([num_frames, window_size], dtype=complex)

    for n in range(num_frames):
        
        # multiply the frame by the window function
        current_frame = np.multiply(window_function(window_size), input_signal_padded[n*hop_size : ((n*hop_size) + window_size)])
        
        # compute the fft of each frame and add it to the matrix
        stft_matrix[n, :] = fft(current_frame)

    return stft_matrix, num_frames, len(input_signal_padded)

# set the parameters of the stft function
window_size = 4096
hop_size = 2048
window_function = hann
ans, num_frames, padded_signal_length = stft(data, window_function, window_size, hop_size)

# ## 4.3 Plot and Interpret
import matplotlib.pyplot as plt

# we use np.abs ignore the phase
S = np.abs(ans[:, :window_size//2])

# x axis = time
times = np.linspace(0, (padded_signal_length / samplerate), num_frames)

# y axis = frequencies
frequencies = np.linspace(0, samplerate//2, window_size//2)

# Convert to dB scale
S_dB = 20 * np.log10(S) 
# Normalize so the maximum value is 0 dB
S_dB -= np.max(S_dB)  

plt.figure(figsize=(10,6))

# need to transpose the matrix because
# plt.colormesh takes the first argument as y
# and the second as x
plt.pcolormesh(times, frequencies, S_dB.T, shading='auto')
plt.title(f"Spectrogram of {filepath}")
plt.xlabel("time (seconds)")
plt.ylabel("frequency (Hz)")
plt.colorbar(label="Magnitude (dB)")
# plt.savefig("4_3.png", format="png", bbox_inches="tight", dpi=300)
plt.show()

### EXTRA
# generate waveform plot for the input signal
plt.figure(figsize=(10,6))
plt.title(f"Waveform plot of {filepath}")
plt.plot(np.linspace(0, len(data)/samplerate, len(data)), data)
plt.xlabel("time (seconds)")
plt.ylabel("amplitude")
# plt.savefig("4_3_extra.png", format="png", bbox_inches="tight", dpi=300)
plt.show()
