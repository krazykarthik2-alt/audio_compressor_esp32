#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment, effects
from scipy.signal import butter, sosfilt

# ---------------------- Parameters ----------------------
TARGET_SR = 32000
RMS_THRESHOLD = 0.055
MIN_SILENCE_MS = 400# Divide 80–4000 Hz into 4 subbands
SUBBANDS = [
    (80,   640),   # center 360
    (640,  1200),  # center 920
    (1200, 1760),  # center 1480
    (1760, 2320),  # center 2040
    (2320, 2880),  # center 2600
    (2880, 3440),  # center 3160
    (3440, 4000)   # center 3720
]
CARRIERS = [1850, 1950, 2100, 2250, 2450, 2700, 2950]  # Hz

# --------------------------------------------------------

os.makedirs("intermediate_steps_mp3", exist_ok=True)
os.makedirs("imgs/step1_imgs", exist_ok=True)

# --- Load audio ---
orig = AudioSegment.from_file("femail.mp3").set_channels(1).set_frame_rate(TARGET_SR)
samples = np.array(orig.get_array_of_samples()).astype(np.float32)
samples /= np.max(np.abs(samples) + 1e-9)  # normalize to -1..1
frame_rate = TARGET_SR


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
    keep = np.ones(len(samples), dtype=bool)
    for s, e in zip(starts, ends):
        if (e - s) >= min_silence_samples:
            keep[s:e] = False
    return samples[keep]


samples_working = remove_silence_rms(samples, frame_rate, RMS_THRESHOLD, MIN_SILENCE_MS)
print(f"Silence removed: {len(samples)} -> {len(samples_working)} samples")


# --- Bandpass helper ---
def bandpass(data, fs, low, high, order=6):
    ny = 0.5 * fs
    sos = butter(order, [low/ny, high/ny], btype="band", output="sos")
    return sosfilt(sos, data)


# --- Multiband modulation ---
t = np.arange(len(samples_working)) / frame_rate
out = np.zeros_like(samples_working)

for (low, high), fc in zip(SUBBANDS, CARRIERS):
    band = bandpass(samples_working, frame_rate, low, high)
    shifted = band * np.cos(2 * np.pi * fc * t)
    out += shifted

# --- Normalize ---
out = out / (np.max(np.abs(out)) + 1e-9) * 0.9  # headroom

# --- Convert back to AudioSegment ---
out_int8 = (out * 127).astype(np.int8).tobytes()
processed_segment = AudioSegment(
    data=out_int8,
    sample_width=1,
    frame_rate=TARGET_SR,
    channels=1,
)
processed_segment = effects.normalize(processed_segment)

# --- Save to MP3 ---
output_path = "intermediate_steps_mp3/step1_output.mp3"
processed_segment.export(output_path, format="mp3", bitrate="64k")
print(f"Processed & spread audio saved at {output_path}")


# --- Visualization ---
def to_numpy_for_plot(seg):
    arr = np.array(seg.set_channels(1).get_array_of_samples())
    sw = seg.sample_width
    denom = float(2 ** (8 * sw - 1))
    return arr / denom, seg.frame_rate


orig_np, orig_sr = to_numpy_for_plot(orig)
proc_np, proc_sr = to_numpy_for_plot(processed_segment)
orig_t = np.linspace(0, len(orig_np)/orig_sr, num=len(orig_np))
proc_t = np.linspace(0, len(proc_np)/proc_sr, num=len(proc_np))

# Waveforms
plt.figure(figsize=(14,8))
plt.subplot(2,1,1)
plt.plot(orig_t, orig_np, linewidth=0.5)
plt.title("Original hello.mp3")
plt.xlabel("Time (s)"); plt.ylabel("Amplitude")

plt.subplot(2,1,2)
plt.plot(proc_t, proc_np, linewidth=0.5, color="green")
plt.title("Processed (Multiband Spread near 2–3 kHz)")
plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("imgs/step1_imgs/waveforms.png")
plt.close()

# Overlapped
plt.figure(figsize=(14,4))
plt.plot(orig_t, orig_np, alpha=0.5, label="Original", linewidth=0.5)
plt.plot(proc_t, proc_np, alpha=0.7, label="Processed", linewidth=0.5, color="green")
plt.title("Overlapped Waveforms")
plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
plt.legend()
plt.savefig("imgs/step1_imgs/overlap.png")
plt.close()

print("Graphs saved in imgs/step1_imgs/")
