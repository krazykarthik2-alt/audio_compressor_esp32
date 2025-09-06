import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from collections import deque

# Parameters
DURATION = 0.5
SR = 44100
CHUNK = int(DURATION * SR)
THRESHOLD_RATIO = 0.1

# Buffer to store incoming samples
buffer = deque(maxlen=CHUNK*2)

def audio_callback(indata, frames, time, status):
    buffer.extend(indata[:,0])

def process_chunk(samples):
    samples = np.array(samples)
    samples = samples.astype(np.float32)
    samples = samples / np.max(np.abs(samples))

    N = len(samples)
    freqs = np.fft.rfftfreq(N, 1/SR)
    fft_mag = np.abs(np.fft.rfft(samples))

    peaks, _ = find_peaks(fft_mag, height=np.max(fft_mag)*THRESHOLD_RATIO)
    if len(peaks) == 0:
        return None, freqs, fft_mag

    peak_freq = freqs[peaks[np.argmax(fft_mag[peaks])]]
    return peak_freq, freqs, fft_mag

# Live plotting setup
plt.ion()
fig, ax = plt.subplots(figsize=(10,4))
line_fft, = ax.plot([], [])
point, = ax.plot([], [], 'ro')
ax.set_xlim(0, 10000)
ax.set_ylim(0, 1)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude")
ax.set_title("Real-time Piezo Resonance")

# Start audio stream
with sd.InputStream(channels=1, samplerate=SR, callback=audio_callback):
    print("Listening for piezo signal... Press Ctrl+C to stop.")
    try:
        while True:
            if len(buffer) >= CHUNK:
                chunk = [buffer.popleft() for _ in range(CHUNK)]
                peak_freq, freqs, fft_mag = process_chunk(chunk)
                if peak_freq is not None:
                    print(f"Resonant frequency ~ {peak_freq:.1f} Hz")

                # Update plot safely in main thread
                line_fft.set_data(freqs, fft_mag/np.max(fft_mag))
                if peak_freq is not None:
                    point.set_data([peak_freq], [np.max(fft_mag)/np.max(fft_mag)])
                fig.canvas.draw()
                fig.canvas.flush_events()
    except KeyboardInterrupt:
        print("Stopped.")
