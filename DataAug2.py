import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import random
from scipy.io import wavfile

# Directory Paths
data_dir = 'C:/Desktop/AI Project/Noise_Clips'  # Directory where your original audio files are stored
output_dir = 'C:/Desktop/AI Project/NoiseSpectro'  # Directory where augmented spectrograms will be saved
os.makedirs(output_dir, exist_ok=True)

# Data Augmentation Parameters
TIME_SHIFT_MAX = 0.2  # Maximum percentage of total time to shift
NOISE_FACTOR = 0.05  # Increased noise factor for Gaussian noise addition

# Function to add Gaussian Noise
def add_noise(data, noise_factor=NOISE_FACTOR):
    noise = np.random.normal(0, noise_factor, data.shape)
    return data + noise

# Function to time-shift an audio signal
def time_shift(data, shift_max=TIME_SHIFT_MAX):
    shift = int(shift_max * len(data) * random.uniform(-1, 1))
    return np.roll(data, shift)

# Function to apply pitch shift
def pitch_shift(data, sampling_rate, n_steps):
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=n_steps)


# Function to save log-mel spectrogram as an image without axis measurements
def save_spectrogram(y, sr, output_path):
    plt.figure(figsize=(10, 4))
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis=None, y_axis=None, cmap='viridis')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

# Loop through the files in your data directory
for file_name in os.listdir(data_dir):
    if file_name.endswith('.wav'):
        file_path = os.path.join(data_dir, file_name)
        y, sr = librosa.load(file_path, sr=None)

        # Original Spectrogram
        save_spectrogram(y, sr, os.path.join(output_dir, f'{file_name}_original.png'))

        # # Augmentation 1: Add Gaussian Noise
        # y_noisy = add_noise(y)
        # save_spectrogram(y_noisy, sr, os.path.join(output_dir, f'{file_name}_noisy.png'))
        #
        # # Augmentation 2: Time Shift
        # y_shifted = time_shift(y)
        # save_spectrogram(y_shifted, sr, os.path.join(output_dir, f'{file_name}_shifted.png'))
        #
        # # Augmentation 3: Pitch Shift
        # n_steps = random.choice([-2, -1, 1, 2])  # Randomly choose to pitch shift up or down by 1 or 2 semitones
        # y_pitch_shifted = pitch_shift(y, sr, n_steps)
        # save_spectrogram(y_pitch_shifted, sr, os.path.join(output_dir, f'{file_name}_pitch_shifted.png'))

        print(f'Augmented images for {file_name} saved.')
