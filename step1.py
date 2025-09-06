import os
from pydub import AudioSegment, effects
import numpy as np
import matplotlib.pyplot as plt

THRESHOLD = 1000
MIN_SILENCE = 400
# --- Ensure output directories exist ---
os.makedirs("intermediate_steps_mp3", exist_ok=True)
os.makedirs("imgs/step1_imgs", exist_ok=True)

# --- Load audio ---
orig = AudioSegment.from_file("hello.mp3")

# Force mono
orig = orig.set_channels(1)

# --- Band-pass filter (telephone band) ---
filtered = orig.low_pass_filter(3400).high_pass_filter(300)

# --- Silence detection & removal (>300 ms silences are deleted) ---
sample_width = filtered.sample_width
frame_rate = filtered.frame_rate
samples = np.array(filtered.get_array_of_samples())

# Amplitude threshold scaling
max_val = (2 ** (8 * sample_width - 1)) - 1
scaled_threshold = int(THRESHOLD * max_val / 32767)

silent_mask = np.abs(samples) < scaled_threshold
mask_int = silent_mask.astype(np.int8)
diff = np.diff(mask_int)
starts = np.where(diff == 1)[0] + 1
ends = np.where(diff == -1)[0] + 1
if mask_int[0] == 1:
    starts = np.concatenate(([0], starts))
if mask_int[-1] == 1:
    ends = np.concatenate((ends, [len(mask_int)]))

samples_working = samples.copy()
for s, e in zip(starts[::-1], ends[::-1]):  # reverse order deletion
    dur_ms = (e - s) / frame_rate * 1000.0
    if dur_ms >= MIN_SILENCE:  # delete silence longer than 300ms
        samples_working = np.delete(samples_working, np.s_[s:e])

raw_bytes = samples_working.tobytes()
processed_segment = AudioSegment(
    data=raw_bytes,
    sample_width=sample_width,
    frame_rate=frame_rate,
    channels=1
)

# --- Normalize & compress ---
processed_segment = effects.normalize(processed_segment)
processed_segment = effects.compress_dynamic_range(
    processed_segment,
    threshold=-20.0,
    ratio=4.0,
    attack=5.0,
    release=50.0
)

# --- Downsample: 8-bit PCM, 8kHz ---
final = processed_segment.set_sample_width(1).set_frame_rate(16000).set_channels(1)

# Export to MP3 (super low bitrate)
output_path = "intermediate_steps_mp3/step1_output.mp3"
final.export(output_path, format="mp3")
print(f"Processed audio saved at {output_path}")

# --- Visualization ---
def to_numpy_for_plot(seg):
    arr = np.array(seg.set_channels(1).get_array_of_samples())
    sw = seg.sample_width
    denom = float(2 ** (8 * sw - 1))
    return arr / denom, seg.frame_rate

orig_np, orig_sr = to_numpy_for_plot(orig)
proc_np, proc_sr = to_numpy_for_plot(final)
orig_t = np.linspace(0, len(orig_np) / orig_sr, num=len(orig_np))
proc_t = np.linspace(0, len(proc_np) / proc_sr, num=len(proc_np))

# Save original vs reduced waveform
plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.plot(orig_t, orig_np, linewidth=0.5)
plt.title("Original hello.mp3")
plt.xlabel("Time (s)"); plt.ylabel("Amplitude")

plt.subplot(2, 1, 2)
plt.plot(proc_t, proc_np, linewidth=0.5, color="green")
plt.title("Processed step1_output.mp3")
plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
plt.tight_layout()
plt.savefig("imgs/step1_imgs/waveforms.png")
plt.close()

# Save overlapped comparison
plt.figure(figsize=(14, 4))
plt.plot(orig_t, orig_np, alpha=0.5, label="Original", linewidth=0.5)
plt.plot(proc_t, proc_np, alpha=0.7, label="Processed", linewidth=0.5, color="green")
plt.title("Overlapped Waveforms")
plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
plt.legend()
plt.savefig("imgs/step1_imgs/overlap.png")
plt.close()

print("Graphs saved in imgs/step1_imgs/")
