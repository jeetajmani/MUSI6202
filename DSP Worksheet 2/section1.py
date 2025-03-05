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
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title(f"File {filepath} - Magnitude Spectrum")
plt.plot(frequencies, XabsdB)
plt.savefig("1_1_b.png", format="png", bbox_inches="tight", dpi=300)
plt.show()

# principal component
principal_freq = np.argmax(XabsdB) * samplerate / signal_length_padded
print(f"Principal Frequency Component: {principal_freq}")

# zoomed-in spectrum
plt.figure(figsize=(10, 6))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title(f"File {filepath} - Magnitude Spectrum Peak: zoomed-in")
plt.plot(frequencies[1250:1300], XabsdB[1250:1300])
plt.savefig("1_1_c.png", format="png", bbox_inches="tight", dpi=300)
plt.show()

t = np.linspace(0, 4, int(samplerate*4), endpoint=False)
sine_234 = 0.8 * np.sin(2 * np.pi * 234.0087890625 * t)
X_sine = fft(sine_234, signal_length_padded)
Xabs_sine = np.abs(X_sine[:signal_length_padded // 2])
XabsdB_sine = 20 * np.log10(Xabs_sine + 1e-10)

plt.figure(figsize=(10, 6))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title(f"234Hz Sine Wave & {filepath} - Magnitude Spectrum")
plt.plot(frequencies, XabsdB, label=f"{filepath}")
plt.plot(frequencies, XabsdB_sine, label="Clean sine wave", alpha=0.8)
plt.legend() 
plt.savefig("1_1_d.png", format="png", bbox_inches="tight", dpi=300)
plt.show()

# wavfile.write("sine_234Hz.wav", samplerate, sine_234)

# 1.2 Filter Design
from scipy.signal import butter

cutoff_freq = 300  # Cutoff frequency in Hz
filter_order = 4

b, a = butter(filter_order, cutoff_freq / (samplerate / 2), btype='low')

print("Numerator coefficients (b):", b)
print("Denominator coefficients (a):", a)

# 1.3 Filter Analysis
from scipy.signal import freqz

w, h = freqz(b, a)
magnitude_dB = 20 * np.log10(np.abs(h))

# magnitude response
plt.figure(figsize=(10, 6))
plt.plot(w/np.pi, magnitude_dB)
plt.title("Magnitude Response (dB)")
plt.xlabel("Normalized Frequency")
plt.ylabel("Magnitude (dB)")
plt.savefig("1_3_a.png", format="png", bbox_inches="tight", dpi=300)
plt.show()

# phase response
plt.figure(figsize=(10, 6))
plt.plot(w/np.pi, np.unwrap(np.angle(h)))
plt.title("Phase Response")
plt.xlabel("Normalized Frequency")
plt.ylabel("Phase (radians)")
plt.savefig("1_3_b.png", format="png", bbox_inches="tight", dpi=300)
plt.show()

# 1.4
from scipy.signal import filtfilt

# applying filter to the original signal
filtered_signal = filtfilt(b, a, data)

# writing filtered signal to wav file
wavfile.write("filtered_signal.wav", samplerate, filtered_signal)

# compute magnitude spectrum
X_filtered = fft(filtered_signal, signal_length_padded)
Xabs_filtered = np.abs(X_filtered[:signal_length_padded // 2])
XabsdB_filtered = 20 * np.log10(Xabs_filtered + 1e-10)

# plot magnitude spectrum
plt.figure(figsize=(10, 6))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title(f"Signal y[n]: File {filepath} Filtered - Magnitude Spectrum")
plt.plot(frequencies, XabsdB_filtered)
plt.savefig("1_4_a.png", format="png", bbox_inches="tight", dpi=300)
plt.show()