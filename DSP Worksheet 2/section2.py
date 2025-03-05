# 2.3
from scipy.signal import freqz
import numpy as np
import matplotlib.pyplot as plt

# Filter coefficients
b = [1, 1]  # Numerator coefficients
T = 1 / 48000  # Sampling period (48,000 Hz sampling rate)
R = 1000       # Assumed resistance (1k Ohm)
C = 1e-6       # Assumed capacitance (1 microfarad)

# Denominator coefficients
a = [1 + (2 * R * C) / T, 1 - (2 * R * C) / T]

# Calculate frequency response
w, h = freqz(b, a, worN=1024)
frequencies = w * 48000 / (2 * np.pi)

# Plot frequency response
plt.figure(figsize=(12, 8))

# Magnitude Response
plt.subplot(2, 1, 1)
plt.plot(frequencies, 20 * np.log10(abs(h) + 1e-10))
plt.title('Frequency Response - Magnitude')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.grid()

# Phase Response
plt.subplot(2, 1, 2)
plt.plot(frequencies, np.angle(h))
plt.title('Frequency Response - Phase')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (radians)')
plt.grid()

plt.tight_layout()
plt.show()

#2.4
from scipy.signal import lfilter, dimpulse, dstep
import matplotlib.pyplot as plt

# Define filter coefficients
b = [1, 1]  # Numerator coefficients
T = 1 / 48000  # Sampling period (48,000 Hz sampling rate)
R = 1000       # Assumed resistance (1k Ohm)
C = 1e-6       # Assumed capacitance (1 microfarad)

# Denominator coefficients
a = [1 + (2 * R * C) / T, 1 - (2 * R * C) / T]

# Calculate impulse response
t_imp, h_imp = dimpulse((b, a, 1), n=50)

# Calculate step response
t_step, s_step = dstep((b, a, 1), n=50)

# Plot impulse response
plt.figure(figsize=(12, 6))
plt.stem(t_imp, h_imp[0].flatten(), basefmt=" ", use_line_collection=True)
plt.title('Impulse Response')
plt.xlabel('n (samples)')
plt.ylabel('Amplitude')
plt.grid()

# Plot step response
plt.figure(figsize=(12, 6))
plt.stem(t_step, s_step[0].flatten(), basefmt=" ", use_line_collection=True)
plt.title('Step Response')
plt.xlabel('n (samples)')
plt.ylabel('Amplitude')
plt.grid()

plt.show()

#2.5
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy.signal import spectrogram, lfilter

# Parameters
fs = 48000  # Sampling rate (Hz)
N = 4 * fs  # Total length: 4 * 48,000
R = 1000    # Assumed resistance (1k Ohm)
C = 1e-6    # Assumed capacitance (1 microfarad)
T = 1 / fs  # Sampling period

# Create a linear cutoff trajectory
fc = np.linspace(10, 20000, N)

# Generate white noise signal
np.random.seed(0)
x = np.random.normal(0, 1, N)

# Define filter coefficients
b = [1, 1]
y = np.zeros(N)

# Apply filtering
for i in range(1, N):
    a = [1 + (2 * R * C) / T, 1 - (2 * R * C) / T]
    y[i] = (b[0] * x[i] + b[1] * x[i - 1] - a[1] * y[i - 1]) / a[0]

# Save filtered signal
wav.write('filtered_sweep.wav', fs, y.astype(np.float32))
