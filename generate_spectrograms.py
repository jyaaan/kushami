import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os


def save_spectrogram(file_path, output_dir, label):
    """Saves image of mel spectrogram from a .WAV file"""
    # y: waveform
    # sr: sampling rate
    y, sr = librosa.load(file_path)
    print(f"Audio Length: {len(y)=}, Sample Rate: {sr=}")
    # convert to mel spectrogram represented by list of list of floats
    # represents intensity of different frequencies over time on mel scale
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)

    # convert from power values to log scale in dB
    # ref=np.max is a hard limiter to 0 dB
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # 6 x 3 in plot window, no axes
    plt.figure(figsize=(6, 3))
    # has to go after figure()
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel")
    plt.axis("off")

    # save!
    output_path = os.path.join(
        output_dir, f"{label}_{os.path.basename(file_path).replace('.wav', '.png')}"
    )
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)

    plt.close()


if __name__ == "__main__":
    for label in ["sneezes_split", "backgrounds_split", "augmented_sneeze_audio"]:
        """ Create spectrogram images for files inf sneeze and background directories """
        input_dir = f"./{label}"
        output_dir = f"./spectrograms/{label}"
        os.makedirs(output_dir, exist_ok=True)
        for file in os.listdir(input_dir):
            if file.endswith(".wav"):
                save_spectrogram(os.path.join(input_dir, file), output_dir, label)
