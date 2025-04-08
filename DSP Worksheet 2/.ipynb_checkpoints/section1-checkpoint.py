# 1 Filters with SciPy

# 1.1 Input Signal Analysis
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fft import fft, fftfreq, ifft

filepath = "input_1.wav"  # Replace with your file path
samplerate, data = wavfile.read(filepath)

data = data / np.max(np.abs(data))

signal_length = len(data)

print(f"Properties of File - {filepath}")

# duration = num samples / sample rate, rounded to 3 places
print(f"Duration: {(len(data) / samplerate):.3f} seconds") 

# the number of channels is determined by the number of dimensions of the data
print(f"Number of channels: {len(data.shape)}")

print(f"Sample rate: {samplerate} Hz")

print(f"Signal length in samples: {signal_length}")

print(f"Bit depth: {data.dtype.itemsize * 8}")

# x-axis in seconds
time_axis = np.arange(len(data)) / samplerate

plt.figure(figsize=(10, 6))
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.title(f"File {filepath} - Waveform")
plt.plot(time_axis, data)
plt.savefig("1_1_a.png", format="png", bbox_inches="tight", dpi=300)
plt.show()

signal_length_padded = 2 ** int(np.ceil(np.log2(signal_length)))

X = fft(data, signal_length_padded)
Xabs = np.abs(X[:signal_length_padded // 2])
XabsdB = 20 * np.log10(Xabs + 1e-10)

frequencies = fftfreq(signal_length_padded, 1/samplerate)[:signal_length_padded // 2]
plt.figure(figsize=(10, 6))
plt.xlabel("f/Hz")
plt.ylabel("|H(f)|")
plt.title(f"File {filepath} - Magnitude Spectrum")
plt.plot(frequencies, XabsdB)
plt.savefig("1_1_b.png", format="png", bbox_inches="tight", dpi=300)
plt.show()



# 1.2 Filter Design
