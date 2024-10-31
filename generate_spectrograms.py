import librosa
import numpy as np
import numpy.typing as npt
import os
from PIL import Image

from .const import (
    CONST_16_BIT,
    hop_length,
    n_fft,
)


def normalize_spectrogram(mel_spec_db: npt.NDArray) -> npt.NDArray:
    # normalize the spectrogram to the range [0, 65535]
    # but return min/max for reconstruction
    min_db = mel_spec_db.min()
    max_db = mel_spec_db.max()
    mel_spec_db -= min_db
    mel_spec_db /= max_db
    mel_spec_db *= CONST_16_BIT
    normalized_mel_spec_db = mel_spec_db.astype(np.uint16)

    return normalized_mel_spec_db, min_db, max_db


def save_spectrogram(file_path, output_dir, label):
    """generate mel spectrogram as a normalized 16-bit png."""
    y, sr = librosa.load(file_path, sr=None)
    print(f"Audio Length: {len(y)=}, Sample Rate: {sr=}")
    print(f"Original Audio Length: {len(y) / sr} seconds")

    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, center=False
    )

    # convert to db array
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    normalized_mel_spec_db, original_min_db, original_max_db = normalize_spectrogram(
        mel_spec_db
    )

    # save the normalized spectrogram as a 16-bit png image
    output_path = os.path.join(
        output_dir, f"{label}_{os.path.basename(file_path).replace('.wav', '.png')}"
    )
    Image.fromarray(normalized_mel_spec_db).save(output_path, format="PNG")
    print(f"Saved spectrogram image to {output_path}")

    # save min/max dB and sample rate for reconstruction
    np.save(
        output_path.replace(".png", "_db_range.npy"), [original_min_db, original_max_db]
    )
    np.save(output_path.replace(".png", "_sr.npy"), sr)


if __name__ == "__main__":
    for label in ["reconstruct_test"]:
        input_dir = f"./{label}"
        output_dir = f"./reconstruct_test/spectrograms/{label}"
        os.makedirs(output_dir, exist_ok=True)
        for file in os.listdir(input_dir):
            if file.endswith(".wav"):
                save_spectrogram(os.path.join(input_dir, file), output_dir, label)
