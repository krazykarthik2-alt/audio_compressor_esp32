#!/usr/bin/env python3
"""
step2_dac_pwm_dual_complementary_sine_touch.py

- Output: outputs/this_sound.h
- Behavior:
    * Pure sine wave at target frequency
    * 32 kHz sample rate
    * 8-bit PCM
    * DAC25/26 = complementary push-pull
    * PWM32/33 = complementary push-pull
    * Maximize output for piezos
    * Supports using DAC piezo for touch & PWM piezo for vibration sensing
"""

import os
import numpy as np

OUTPUT_FILE = "outputs/this_sound.h"

TARGET_SR = 32000        # 32 kHz
FREQ_HZ = 1000           # 1 kHz sine wave
DURATION_S = 2           # 2 seconds
HEADER_ARRAY_LINE_WIDTH = 20

DAC_PIN_A, DAC_PIN_B = 25, 26
PWM_PIN_A, PWM_PIN_B = 32, 33
PWM_FREQ_HZ, PWM_RES_BITS = 40000, 8   # 40 kHz PWM

# --- Generate waveform ---
t = np.arange(0, DURATION_S, 1 / TARGET_SR)
sine_wave = np.sin(2 * np.pi * FREQ_HZ * t)

# --- Normalize to full 8-bit range 0..255 for max push-pull output ---
sine_u8 = ((sine_wave * 127) + 128).clip(0, 255).astype(np.uint8)
sound_len = len(sine_u8)

# --- Prepare output folder ---
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
print(f"Writing header to: {OUTPUT_FILE}")

with open(OUTPUT_FILE, "w") as f:
    f.write("#ifndef THIS_SOUND_H\n")
    f.write("#define THIS_SOUND_H\n\n")
    f.write("// Dual DAC + Dual PWM complementary sine wave playback for ESP32\n")
    f.write(f"#define DAC_PIN_A {DAC_PIN_A}\n")
    f.write(f"#define DAC_PIN_B {DAC_PIN_B}\n")
    f.write(f"#define PWM_PIN_A {PWM_PIN_A}\n")
    f.write(f"#define PWM_PIN_B {PWM_PIN_B}\n\n")
    f.write("#include <Arduino.h>\n")
    f.write("#include <pgmspace.h>\n\n")
    f.write(f"const unsigned int this_sound_len = {sound_len};\n")
    f.write(f"// Sample rate: {TARGET_SR} Hz\n\n")

    # data array
    f.write("const unsigned char this_sound[] PROGMEM = {\n")
    for i in range(0, sound_len, HEADER_ARRAY_LINE_WIDTH):
        chunk = sine_u8[i:i+HEADER_ARRAY_LINE_WIDTH]
        f.write("  " + ", ".join(str(int(x)) for x in chunk) + ",\n")
    f.write("};\n\n")

    # playback function with touch + vibration sensing comments
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
    f.write("        uint8_t v = pgm_read_byte_near(this_sound + i);\n")
    f.write("        // DAC push-pull\n")
    f.write("        dacWrite(DAC_PIN_A, v);\n")
    f.write("        dacWrite(DAC_PIN_B, 255 - v);\n")
    f.write("        // PWM push-pull\n")
    f.write("        ledcWrite(CHANNEL_A, v);\n")
    f.write("        ledcWrite(CHANNEL_B, 255 - v);\n")
    f.write("        ets_delay_us(SAMPLE_PERIOD_US);\n")
    f.write("    }\n\n")

    f.write("    // reset outputs\n")
    f.write("    dacWrite(DAC_PIN_A, 128);\n")
    f.write("    dacWrite(DAC_PIN_B, 128);\n")
    f.write("    ledcWrite(CHANNEL_A, 128);\n")
    f.write("    ledcWrite(CHANNEL_B, 128);\n\n")

    f.write("    // --- Touch & vibration sensing ---\n")
    f.write("    // Use DAC piezo for touchRead(pin)\n")
    f.write("    // Use PWM piezo for analogRead(pin) or ADC to detect vibration\n")
    f.write("}\n\n")
    f.write("#endif // THIS_SOUND_H\n")

print(f"Sine wave header generated: {OUTPUT_FILE}")
