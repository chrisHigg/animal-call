from pydub import AudioSegment
import os

def normalize_audio_rms(audio_path, target_dBFS=-20.0):
    """
    Normalize the audio file to a target dBFS.

    Parameters:
    audio_path (str): Path to the input audio file.
    target_dBFS (float): Desired target loudness in decibels relative to full scale.

    Returns:
    AudioSegment: Normalized audio segment.
    """
    audio = AudioSegment.from_file(audio_path)
    change_in_dBFS = target_dBFS - audio.dBFS
    normalized_audio = audio.apply_gain(change_in_dBFS)
    return normalized_audio

# Define paths to your audio files
input_directory = r"C:\Desktop\AI Project"
output_directory = r"C:\Desktop\AI Project"

# Create output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Loop through the audio files and normalize each one
for audio_file in os.listdir(input_directory):
    if audio_file.endswith(".wav"):
        input_path = os.path.join(input_directory, audio_file)
        output_path = os.path.join(output_directory, f"normalized_{audio_file}")

        try:
            # Normalize the audio
            normalized_audio = normalize_audio_rms(input_path)

            # Export normalized audio to output directory
            normalized_audio.export(output_path, format="wav")

            print(f"Normalized and saved: {output_path}")
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
