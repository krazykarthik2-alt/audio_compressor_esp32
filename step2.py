#!/usr/bin/env python3
"""
step2.py

- Input:  intermediate_steps_mp3/step1_output.mp3
- Output: outputs/this_sound.h
- Behavior:
    * Resample to 16 kHz
    * Force mono, 8-bit PCM
    * Apply low-pass filter at 5 kHz
    * Convert into two branches:
        - cos-modulated (goes to DAC pins 25/26)
        - sin-modulated (goes to PWM pins 33/32)
    * Each is unsigned 8-bit (0..255)
    * Save as PROGMEM arrays in this_sound.h
"""

import os
import sys
import numpy as np
from pydub import AudioSegment

INPUT_FILE = "intermediate_steps_mp3/step1_output.mp3"
OUTPUT_FILE = "outputs/this_sound.h"

TARGET_SR = 32000
LPF_CUTOFF_HZ = 5000
HEADER_ARRAY_LINE_WIDTH = 20
CARRIER_HZ = 2000   # carrier frequency for modulation (center ~2 kHz)

DAC_PIN_A, DAC_PIN_B = 25, 26
PWM_PIN_A, PWM_PIN_B = 33, 32
PWM_FREQ_HZ, PWM_RES_BITS = 20000, 8

def ensure_file_exists(path):
    if not os.path.isfile(path):
        print(f"ERROR: input file not found: {path}", file=sys.stderr)
        sys.exit(2)

def normalize_to_u8(arr: np.ndarray) -> np.ndarray:
    """Normalize to unsigned 8-bit (0..255)."""
    arr = arr / np.max(np.abs(arr))  # scale -1..1
    arr = (arr * 127.5 + 127.5).clip(0, 255)
    return arr.astype(np.uint8)

def main():
    ensure_file_exists(INPUT_FILE)

    print(f"Loading: {INPUT_FILE}")
    audio = AudioSegment.from_file(INPUT_FILE)
    audio = audio.set_channels(1).set_frame_rate(TARGET_SR)
    audio = audio.low_pass_filter(LPF_CUTOFF_HZ)
    audio = audio.set_sample_width(2)  # process as 16-bit internally

    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= np.max(np.abs(samples))  # normalize to -1..1

    t = np.arange(len(samples)) / TARGET_SR
    cos_branch = samples * np.cos(2 * np.pi * CARRIER_HZ * t)
    sin_branch = samples * np.sin(2 * np.pi * CARRIER_HZ * t)

    u8_cos = normalize_to_u8(cos_branch)
    u8_sin = normalize_to_u8(sin_branch)
    sound_len = len(u8_cos)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    print(f"Writing header to: {OUTPUT_FILE}")

    with open(OUTPUT_FILE, "w") as f:
        f.write("#ifndef THIS_SOUND_H\n")
        f.write("#define THIS_SOUND_H\n\n")
        f.write("// Dual-branch (cos/sin) playback for ESP32\n")
        f.write(f"#define DAC_PIN_A {DAC_PIN_A}\n")
        f.write(f"#define DAC_PIN_B {DAC_PIN_B}\n")
        f.write(f"#define PWM_PIN_A {PWM_PIN_A}\n")
        f.write(f"#define PWM_PIN_B {PWM_PIN_B}\n\n")
        f.write("#include <Arduino.h>\n")
        f.write("#include <pgmspace.h>\n\n")
        f.write(f"const unsigned int this_sound_len = {sound_len};\n")
        f.write(f"// Sample rate: {TARGET_SR} Hz, Carrier: {CARRIER_HZ} Hz\n\n")

        # cos array
        f.write("const unsigned char this_sound_cos[] PROGMEM = {\n")
        for i in range(0, sound_len, HEADER_ARRAY_LINE_WIDTH):
            chunk = u8_cos[i:i+HEADER_ARRAY_LINE_WIDTH]
            f.write("  " + ", ".join(str(int(x)) for x in chunk) + ",\n")
        f.write("};\n\n")

        # sin array
        f.write("const unsigned char this_sound_sin[] PROGMEM = {\n")
        for i in range(0, sound_len, HEADER_ARRAY_LINE_WIDTH):
            chunk = u8_sin[i:i+HEADER_ARRAY_LINE_WIDTH]
            f.write("  " + ", ".join(str(int(x)) for x in chunk) + ",\n")
        f.write("};\n\n")

        # playback function
        f.write("void play_this_sound() {\n")
        f.write("    const int CHANNEL_A = 0;\n")
        f.write("    const int CHANNEL_B = 1;\n")
        f.write(f"    const int PWM_FREQ_HZ = {PWM_FREQ_HZ};\n")
        f.write(f"    const int PWM_RES_BITS = {PWM_RES_BITS};\n")
        f.write(f"    const unsigned long SAMPLE_RATE = {TARGET_SR}UL;\n")
        f.write("    const unsigned int SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE;\n\n")
        f.write("    static bool initialized = false;\n")
        f.write("    if (!initialized) {\n")
        f.write("        ledcSetup(CHANNEL_A, PWM_FREQ_HZ, PWM_RES_BITS);\n")
        f.write("        ledcSetup(CHANNEL_B, PWM_FREQ_HZ, PWM_RES_BITS);\n")
        f.write("        ledcAttachPin(PWM_PIN_A, CHANNEL_A);\n")
        f.write("        ledcAttachPin(PWM_PIN_B, CHANNEL_B);\n")
        f.write("        ledcWrite(CHANNEL_A, 128);\n")
        f.write("        ledcWrite(CHANNEL_B, 128);\n")
        f.write("        initialized = true;\n")
        f.write("    }\n\n")
        f.write("    for (unsigned int i = 0; i < this_sound_len; ++i) {\n")
        f.write("        uint8_t c = pgm_read_byte_near(this_sound_cos + i);\n")
        f.write("        uint8_t s = pgm_read_byte_near(this_sound_sin + i);\n")
        f.write("        dacWrite(DAC_PIN_A, c);\n")
        f.write("        dacWrite(DAC_PIN_B, 255 - c);\n")
        f.write("        ledcWrite(CHANNEL_A, s);\n")
        f.write("        ledcWrite(CHANNEL_B, 255 - s);\n")
        f.write("        ets_delay_us(SAMPLE_PERIOD_US);\n")
        f.write("    }\n\n")
        f.write("    dacWrite(DAC_PIN_A, 128);\n")
        f.write("    dacWrite(DAC_PIN_B, 128);\n")
        f.write("    ledcWrite(CHANNEL_A, 128);\n")
        f.write("    ledcWrite(CHANNEL_B, 128);\n")
        f.write("}\n\n")
        f.write("#endif // THIS_SOUND_H\n")

if __name__ == "__main__":
    main()
