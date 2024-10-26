import librosa
import numpy as np
import soundfile as sf  # type: ignore
import os

from combine_audio import combine_wavs

INPUT_DIR = "./background"
COMBINED_FILE_NAME = "./sneeze/combined_backgrounds.wav"
OUTPUT_DIR = "./backgrounds_split"


def split_into_fixed_intervals(input_file, output_dir, chunk_duration=0.25):
    """Split audio into fixed-length chunks of specified duration.
    My sneezes are, on average, 0.25 seconds long with some variation we should address
    Mean (sec): 0.263658839880038
    Median (sec): 0.25541950113378686
    Standard Deviation (sec): 0.12008344343581413
    """
    y, sr = librosa.load(input_file, sr=None)

    # Calculate the number of samples per chunk
    chunk_samples = int(chunk_duration * sr)

    os.makedirs(output_dir, exist_ok=True)

    total_chunks = len(y) // chunk_samples
    print(f"Total {total_chunks} chunks of {chunk_duration:.2f} seconds each.")

    # Loop through and save each chunk
    for i in range(total_chunks):
        start = i * chunk_samples
        end = start + chunk_samples
        chunk = y[start:end]

        output_path = os.path.join(output_dir, f"chunk_{i + 1}.wav")
        sf.write(output_path, chunk, sr)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    combine_wavs(INPUT_DIR, COMBINED_FILE_NAME)
    split_into_fixed_intervals(COMBINED_FILE_NAME, OUTPUT_DIR)
