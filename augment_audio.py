import librosa
import numpy as np
import soundfile as sf  # type: ignore
import os
import random
from enum import auto, Enum


class SilenceMethods(Enum):
    BEFORE = auto()
    AFTER = auto()
    RANDOM = auto()


def load_audio(file_path, sr=22050):
    y, _ = librosa.load(file_path, sr=sr)
    return y


def save_audio(y, sr, output_path):
    sf.write(output_path, y, sr)


def pitch_shift(y, sr, n_steps):
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)


def time_stretch(y, rate):
    return librosa.effects.time_stretch(y, rate=rate)


def time_shift(y, shift_max=0.5):
    shift = int(np.random.uniform(-shift_max, shift_max) * len(y))
    return np.roll(y, shift)


def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    return y + noise_factor * noise


def augment_and_save(file_path, output_dir, variant_count=10, sr=22050):
    y = load_audio(file_path, sr)
    base_name = os.path.basename(file_path).replace(".wav", "")

    # Generate and save multiple augmented versions
    for variant in range(variant_count):
        try:
            rand_pitch = random.uniform(-2, 2)
            rand_stretch = random.uniform(0.9, 1.1)

            y_pitch = pitch_shift(y, sr=sr, n_steps=rand_pitch)
            y_stretch = time_stretch(y, rate=rand_stretch)
            y_shift = time_shift(y)
            y_noise = add_noise(y)

            save_audio(y_pitch, sr, f"{output_dir}/{base_name}_{variant}_pitch.wav")
            save_audio(y_stretch, sr, f"{output_dir}/{base_name}_{variant}_stretch.wav")
            save_audio(y_shift, sr, f"{output_dir}/{base_name}_{variant}_shift.wav")
            save_audio(y_noise, sr, f"{output_dir}/{base_name}_{variant}_noise.wav")
        except Exception as e:
            print(f"Error processing {file_path} (variant {variant}): {e}")


def get_longest_sound_duration(directory_path):
    longest_duration = 0
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) and filename.endswith(".wav"):
            try:
                duration = librosa.get_duration(filename=file_path)
                if duration > longest_duration:
                    longest_duration = duration
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return longest_duration


def pad_silence(target_duration, file_path, method=SilenceMethods.RANDOM, sr=22050):
    duration = librosa.get_duration(filename=file_path)
    if duration >= target_duration:
        print(f"No padding needed for {file_path}.")
        return

    y = load_audio(file_path, sr)
    silence_duration = target_duration - duration
    silence_samples = int(silence_duration * sr)

    left_silence, right_silence = np.zeros(0), np.zeros(0)
    if method == SilenceMethods.BEFORE:
        left_silence = np.zeros(silence_samples)
    elif method == SilenceMethods.AFTER:
        right_silence = np.zeros(silence_samples)
    elif method == SilenceMethods.RANDOM:
        split = random.randint(0, silence_samples)
        left_silence = np.zeros(split)
        right_silence = np.zeros(silence_samples - split)

    padded_audio = np.concatenate([left_silence, y, right_silence])
    save_audio(padded_audio, sr, file_path)
    print(f"Padded {file_path} with {silence_duration:.2f} seconds of silence.")


if __name__ == "__main__":
    input_dir = "./sneezes_split/"
    output_dir = "./augmented_sneeze_audio/"
    os.makedirs(output_dir, exist_ok=True)

    # Augment and save all input files
    for file in os.listdir(input_dir):
        if file.endswith(".wav"):
            augment_and_save(os.path.join(input_dir, file), output_dir)

    # Pad input files to match the longest duration
    longest_input_duration = get_longest_sound_duration(input_dir)
    for file in os.listdir(input_dir):
        if file.endswith(".wav"):
            pad_silence(longest_input_duration, os.path.join(input_dir, file))

    # Pad output files to match the longest output duration
    longest_output_duration = get_longest_sound_duration(output_dir)
    for file in os.listdir(output_dir):
        if file.endswith(".wav"):
            pad_silence(longest_output_duration, os.path.join(output_dir, file))
