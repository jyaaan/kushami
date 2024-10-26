import librosa
import numpy as np
import soundfile as sf  # type: ignore
import os

from combine_audio import combine_wavs

INPUT_DIR = "./sneeze"
COMBINED_FILE_NAME = "./sneeze/combined_sneezes.wav"
OUTPUT_DIR = "./sneezes_split"


def split_audio(input_file, output_dir, top_db=20, min_duration=0.15):
    """Split the audio by non-silent sections and save each segment."""
    y, sr = librosa.load(input_file, sr=None)

    # Detect non-silent intervals
    intervals = librosa.effects.split(y, top_db=top_db)
    print(f"Detected {len(intervals)} intervals")

    os.makedirs(output_dir, exist_ok=True)

    saved_count = 0
    durations = []
    for i, (start, end) in enumerate(intervals):
        duration = (end - start) / sr
        print(f"Interval {i}: Start={start}, End={end}, Duration={duration:.2f}s")

        if duration >= min_duration:
            durations.append(duration)
            output_path = os.path.join(output_dir, f"sneeze_{saved_count + 1}.wav")
            sf.write(output_path, y[start:end], sr)
            print(f"Saved: {output_path}")
            saved_count += 1

    if saved_count == 0:
        print(
            "No valid segments were saved. Check the top_db or min_duration settings."
        )
        return

    mean_value = np.mean(durations)
    median_value = np.median(durations)
    std_deviation = np.std(durations)

    print(f"Mean (sec): {mean_value}")
    print(f"Median (sec): {median_value}")
    print(f"Standard Deviation (sec): {std_deviation}")


if __name__ == "__main__":
    combine_wavs(INPUT_DIR, COMBINED_FILE_NAME)
    split_audio(COMBINED_FILE_NAME, OUTPUT_DIR)
