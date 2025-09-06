#!/usr/bin/env python3
"""
step1.py

- Input:  hello.mp3
- Output: intermediate_steps_mp3/step1_output.mp3
- Also saves waveform plots in imgs/step1_imgs/
- Processing:
    * Load mono, resample to 16 kHz
    * Remove silence based on RMS threshold
    * Normalize to RMS target
    * Apply small frequency shift (~200 Hz)
    * Save as 8-bit PCM MP3
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment, effects
from scipy.signal import hilbert

# ---------------------- Parameters ----------------------
RMS_THRESHOLD = 0.032      # threshold for silence removal
MIN_SILENCE_MS = 400       # minimum silence duration to remove
TARGET_SR = 16000          # resample rate
SHIFT_HZ = 200             # small frequency shift
# --------------------------------------------------------

# Ensure directories exist
os.makedirs("intermediate_steps_mp3", exist_ok=True)
os.makedirs("imgs/step1_imgs", exist_ok=True)

# --- Load audio ---
orig = AudioSegment.from_file("hello.mp3").set_channels(1).set_frame_rate(TARGET_SR)
samples = np.array(orig.get_array_of_samples()).astype(np.float32)

# Normalize to -1..1 based on sample width
norm_factor = float(2 ** (8 * orig.sample_width - 1))
samples = samples / norm_factor

# --- Silence removal using RMS ---
def remove_silence_rms(samples, sr, threshold=0.055, min_silence_ms=400):
    window_ms = 10
    window_size = max(1, int(sr * window_ms / 1000))
    rms = np.sqrt(np.convolve(samples**2, np.ones(window_size)/window_size, mode="same"))
    silent_mask = rms < threshold

    diff = np.diff(silent_mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if silent_mask[0]:
        starts = np.concatenate(([0], starts))
    if silent_mask[-1]:
        ends = np.concatenate((ends, [len(samples)]))

    min_silence_samples = int(sr * min_silence_ms / 1000)
    keep_mask = np.ones_like(samples, dtype=bool)
    for s, e in zip(starts, ends):
        if (e - s) >= min_silence_samples:
            keep_mask[s:e] = False

    return samples[keep_mask]

samples_working = remove_silence_rms(samples, TARGET_SR, RMS_THRESHOLD, MIN_SILENCE_MS)
print(f"Silence removed: {len(samples)} -> {len(samples_working)} samples")

# --- Normalize by RMS ---
rms = np.sqrt(np.mean(samples_working**2))
target_rms = 0.1
samples_working = samples_working * (target_rms / (rms + 1e-9))

# --- Apply small frequency shift ---
t = np.arange(len(samples_working)) / TARGET_SR
analytic = hilbert(samples_working)
shifted = np.real(analytic * np.exp(1j * 2 * np.pi * SHIFT_HZ * t))

# --- Convert to 8-bit PCM ---
shifted = shifted / np.max(np.abs(shifted)) * 127
shifted_int8 = shifted.astype(np.int8).tobytes()

processed_segment = AudioSegment(
    data=shifted_int8,
    sample_width=1,
    frame_rate=TARGET_SR,
    channels=1,
)
processed_segment = effects.normalize(processed_segment)

# Save MP3
output_path = "intermediate_steps_mp3/step1_output.mp3"
processed_segment.export(output_path, format="mp3", bitrate="64k")
print(f"Processed & shifted audio saved at {output_path}")

# --- Visualization ---
def to_numpy_for_plot(seg):
    arr = np.array(seg.set_channels(1).get_array_of_samples())
    sw = seg.sample_width
    denom = float(2 ** (8*sw - 1))
    return arr / denom, seg.frame_rate

orig_np, orig_sr = to_numpy_for_plot(orig)
proc_np, proc_sr = to_numpy_for_plot(processed_segment)

orig_t = np.linspace(0, len(orig_np)/orig_sr, num=len(orig_np))
proc_t = np.linspace(0, len(proc_np)/proc_sr, num=len(proc_np))

# Waveforms
plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.plot(orig_t, orig_np, linewidth=0.5)
plt.title("Original hello.mp3")
plt.xlabel("Time (s)"); plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(proc_t, proc_np, linewidth=0.5, color="green")
plt.title(f"Processed (Shifted by {SHIFT_HZ} Hz, Silence Removed, RMS Norm)")
plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("imgs/step1_imgs/waveforms.png")
plt.close()

# Overlapped
plt.figure(figsize=(14, 4))
plt.plot(orig_t, orig_np, alpha=0.5, label="Original", linewidth=0.5)
plt.plot(proc_t, proc_np, alpha=0.7, label="Processed", linewidth=0.5, color="green")
plt.title("Overlapped Waveforms")
plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
plt.legend()
plt.savefig("imgs/step1_imgs/overlap.png")
plt.close()

print("Graphs saved in imgs/step1_imgs/")
