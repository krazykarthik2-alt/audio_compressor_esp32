#!/usr/bin/env python3
"""
step2.py

- Input:  intermediate_steps_mp3/step1_output.mp3
- Output: this_sound.h (PROGMEM array + differential DAC playback for GPIO25/26)
- Behavior:
    * Resample to 16 kHz
    * Force mono, 8-bit PCM
    * Apply low-pass filter at 5000 Hz (to reduce HF images for a piezo resonant ~2-5 kHz)
    * Convert to unsigned 8-bit (0..255)
    * Save C header with array and DAC playback using dacWrite(DAC_PIN_A/ B)
"""

import os
import sys
import numpy as np
from pydub import AudioSegment

INPUT_FILE = "intermediate_steps_mp3/step1_output.mp3"
OUTPUT_FILE = "outputs/this_sound.h"

# Parameters
TARGET_SR = 16000           # sample rate we want in header (16 kHz)
LPF_CUTOFF_HZ = 5000        # low-pass cutoff to keep energy mostly under piezo resonance
HEADER_ARRAY_LINE_WIDTH = 20  # numbers per line in generated C array

def ensure_file_exists(path):
    if not os.path.isfile(path):
        print(f"ERROR: input file not found: {path}", file=sys.stderr)
        sys.exit(2)

def to_unsigned_u8(samples: np.ndarray) -> np.ndarray:
    """
    Convert numpy int array to unsigned 8-bit (0..255) suitable for dacWrite.
    pydub/get_array_of_samples may return signed values for 8-bit audio (-128..127).
    Handle both cases:
      - If min < 0 => assume signed 8-bit, convert by +128
      - Else if values already in 0..255, cast to uint8
      - For safety, clip into 0..255
    """
    samp = samples.astype(np.int32)  # avoid overflow for arithmetic
    if samp.min() < 0:
        # signed -> convert
        samp = samp + 128
    # clip to [0,255] then cast
    samp = np.clip(samp, 0, 255).astype(np.uint8)
    return samp

def main():
    ensure_file_exists(INPUT_FILE)

    print(f"Loading: {INPUT_FILE}")
    audio = AudioSegment.from_file(INPUT_FILE)

    # Force mono
    audio = audio.set_channels(1)

    # Resample first, then low-pass filter. pydub's filters operate on the segment's frame_rate.
    audio = audio.set_frame_rate(TARGET_SR)

    # Apply low-pass to reduce HF content above LPF_CUTOFF_HZ
    # pydub.low_pass_filter uses a simple filter implemented in ffmpeg; this is convenient.
    print(f"Applying low-pass filter at {LPF_CUTOFF_HZ} Hz")
    audio = audio.low_pass_filter(LPF_CUTOFF_HZ)

    # Force to 8-bit sample width (1 byte / sample)
    audio = audio.set_sample_width(1)

    # Convert to numpy samples
    samples = np.array(audio.get_array_of_samples())

    # Convert to unsigned 8-bit 0..255 samples for the DAC
    u8 = to_unsigned_u8(samples)
    sound_len = len(u8)
    print(f"Prepared samples: {sound_len} samples (mono, {TARGET_SR} Hz, 8-bit)")

    # Create header file
    print(f"Writing header to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w") as f:
        f.write("#ifndef THIS_SOUND_H\n")
        f.write("#define THIS_SOUND_H\n\n")
        f.write("// Differential DAC playback for ESP32 using DAC pins (GPIO 25 / 26)\n")
        f.write("#define DAC_PIN_A 25\n")
        f.write("#define DAC_PIN_B 26\n\n")
        f.write("#include <Arduino.h>\n")
        f.write("#include <pgmspace.h>\n\n")
        f.write("// Audio metadata\n")
        f.write(f"const unsigned int this_sound_len = {sound_len}; // number of samples\n")
        f.write(f"// sample rate (Hz) used to generate the data: {TARGET_SR}\n\n")

        f.write("const unsigned char this_sound[] PROGMEM = {\n")
        # write numbers in lines
        for i in range(0, sound_len, HEADER_ARRAY_LINE_WIDTH):
            chunk = u8[i:i+HEADER_ARRAY_LINE_WIDTH]
            line = ", ".join(str(int(x)) for x in chunk)
            # add comma at end except for last line (safe to leave trailing comma in C)
            f.write("  " + line + ",\n")
        f.write("};\n\n")

        f.write("// Differential DAC playback function (blocking)\n")
        f.write("void play_this_sound() {\n")
        f.write("    // Notes:\n")
        f.write("    //  - This is a blocking playback routine intended for ESP32.\n")
        f.write("    //  - It writes sample to DAC_PIN_A and inverted sample to DAC_PIN_B\n")
        f.write("    //  - Keep small series resistors (220..470 ohm) and a reconstruction low-pass\n")
        f.write("    //    (e.g. R=1k, C=33nF) in hardware to reduce HF stepping noise and damping.\n\n")
        f.write("    const unsigned long SAMPLE_RATE = " + str(TARGET_SR) + "UL;\n")
        f.write("    const unsigned int SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE; // microseconds per sample\n\n")
        f.write("    // Play buffer (blocking)\n")
        f.write("    for (unsigned int i = 0; i < this_sound_len; ++i) {\n")
        f.write("        uint8_t sample = pgm_read_byte_near(this_sound + i);\n")
        f.write("        // write differential outputs: A = sample, B = inverted\n")
        f.write("        dacWrite(DAC_PIN_A, sample);\n")
        f.write("        dacWrite(DAC_PIN_B, 255 - sample);\n")
        f.write("        // wait until next sample\n")
        f.write("        ets_delay_us(SAMPLE_PERIOD_US);\n")
        f.write("    }\n\n")
        f.write("    // restore mid-level to avoid DC bias on the piezo\n")
        f.write("    dacWrite(DAC_PIN_A, 128);\n")
        f.write("    dacWrite(DAC_PIN_B, 128);\n")
        f.write("}\n\n")
        f.write("#endif // THIS_SOUND_H\n")

if __name__ == "__main__":
    main()
