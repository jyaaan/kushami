"""Reconstructs original audio from its spectrogram.
Or it tries to, anyway.

Resources:
https://www.reddit.com/r/synthrecipes/comments/kx2fdb/reverse_engineering_sound_from_its_spectrogram/
 - General how-to
 - Why grayscale is important

https://stackoverflow.com/questions/61132574/can-i-convert-spectrograms-generated-with-librosa-back-to-audio
 - As spectrograms represent the magnitude of the signal, but not the, we will attempt to estimate the lost
   phase data iteratively using the Griffin-Lim algorithm.


Still to be resolved:
 - The audio data is quantized to the bit depth of the image -- 16-bit for PNG -- which introduces
   quantization noise.
 - Image resolution will drop some details of the original spectrogram, which may be irreversible.
 - Higher frequencies perform worse, why?
 - Amplitude is reduced. Is this consistent and can we correct for it?
"""

import numpy as np
import os
import librosa
import soundfile as sf  # type: ignore
from PIL import Image

from .const import (
    CONST_16_BIT,
    hop_length,
    n_fft,
)

input_dir = "/Users/johnyamashiro/projects/kushami/reconstruct_test/spectrograms"
output_dir = "/Users/johnyamashiro/projects/kushami/reconstruct_test/reconstructed"
os.makedirs(output_dir, exist_ok=True)


def load_image(file_path):
    """load a 16-bit png image and convert it back to a normalized spectrogram"""
    img = Image.open(file_path)
    img_array = np.array(img).astype(np.float32)

    img_array /= CONST_16_BIT

    return img_array


def convert_image_to_db(img_array, original_min_db):
    """convert normalized image back to mel spectrogram in dB scale"""
    # undo the scaling
    mel_spec_db = img_array * (original_max_db - original_min_db) + original_min_db
    return mel_spec_db


def convert_db_to_power(mel_spec_db):
    """convert mel spectrogram in dB back to power."""
    return librosa.db_to_power(mel_spec_db, ref=1.0)


def apply_griffin_lim(mel_spec_power, n_iter=32):
    """reconstruct audio from mel power spectrogram using Griffin-Lim algo"""
    # convert mel spectrogram back to linear-frequency spectrogram
    linear_spec = librosa.feature.inverse.mel_to_stft(
        mel_spec_power,
        sr=sr,
        n_fft=n_fft,
    )

    # GLA
    reconstructed_audio = librosa.griffinlim(
        linear_spec, n_iter=n_iter, n_fft=n_fft, hop_length=hop_length
    )
    print(f"Reconstructed Audio Length: {len(reconstructed_audio) / sr} seconds")

    return reconstructed_audio


def save_audio(file_path, audio):
    sf.write(file_path, audio, sr)


if __name__ == "__main__":
    for file in os.listdir(input_dir):
        if not file.endswith(".png"):
            continue

        audio_filename = os.path.splitext(file)[0] + ".wav"
        img_path = os.path.join(input_dir, file)
        audio_path = os.path.join(output_dir, audio_filename)

        print(f"Processing: {img_path}")
        print(f"Output Audio Path: {audio_path}")

        try:
            # load sample rate and original dB range per file
            sr = np.load(img_path.replace(".png", "_sr.npy"))
            print(f"Loaded sample rate: {sr}")

            db_range = np.load(img_path.replace(".png", "_db_range.npy"))
            original_min_db, original_max_db = db_range

            img_array = load_image(img_path)

            mel_spec_db = convert_image_to_db(img_array, original_min_db)
            mel_spec_power = convert_db_to_power(mel_spec_db)
            reconstructed_audio = apply_griffin_lim(mel_spec_power)

            save_audio(audio_path, reconstructed_audio)
            print(f"Saved: {audio_path}")
        except Exception as e:
            print(f"Failed to process {file}: {e}")
