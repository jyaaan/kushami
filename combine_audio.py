import librosa
import numpy as np
import soundfile as sf  # type: ignore
import os


def combine_wavs(input_dir, output_file, target_sr=22050):
    """Combine all WAV files in the directory into one."""
    combined_audio = []

    if os.path.exists(output_file):
        print(f"Removing existing file: {output_file}")
        os.remove(output_file)

    for file in sorted(os.listdir(input_dir)):
        if file.endswith(".wav"):
            y, sr = librosa.load(os.path.join(input_dir, file), sr=None)

            # Resample if the sample rate is different from the target sample rate
            if sr != target_sr:
                print(f"Resampling {file} from {sr} Hz to {target_sr} Hz")
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

            combined_audio.append(y)

    # Concatenate all audio arrays
    combined_audio = np.concatenate(combined_audio)

    # Save the combined audio to a single file
    sf.write(output_file, combined_audio, target_sr)
    print(f"Combined audio saved to {output_file}")
