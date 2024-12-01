import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from pydub import AudioSegment

# Load the trained model
model = load_model('C:/Desktop/AI Project/Attempt#3/modelNoise.h5')  # Replace with the path to your saved model

# Parameters for Spectrogram
img_height, img_width = 224, 224  # Match the input dimensions of your model
color_map = 'viridis'  # You can change this to any other color map you prefer


# Function to save a log-mel spectrogram with specific format
def save_spectrogram(y, sr, output_path):
    plt.figure(figsize=(10, 4))
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis=None, y_axis=None, cmap=color_map)
    plt.axis('off')  # Remove axes for a cleaner image
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

def classify_with_percentages(wav_file, threshold=0.6):
    # Convert .wav file to log-mel spectrogram
    spectrogram_path = "temp_spectrogram.png"

    # Load the .wav file
    y, sr = librosa.load(wav_file, sr=None)
    save_spectrogram(y, sr, spectrogram_path)

    # Load the spectrogram image for classification
    img = image.load_img(spectrogram_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale to match the training preprocessing

    # Predict the class
    predictions = model.predict(img_array)
    max_prob = np.max(predictions)  # Get the highest probability
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Define the mapping from class index to label
    class_labels = {0: 'Atlantic Spotted Dolphin', 1: 'Beluga', 2: 'Noise', 3: 'NorthernRightWhale'}  # Replace with your actual class labels

    # Print prediction percentages for each class
    print("Prediction Percentages for Each Class:")
    for idx, label in class_labels.items():
        print(f"{label}: {predictions[0][idx] * 100:.2f}%")

    # Check if the maximum probability is above the threshold
    if max_prob < threshold:
        print("Predicted Class: Unknown (Low Confidence)")
    else:
        print(f"Predicted Class: {class_labels[predicted_class]} (Confidence: {max_prob:.2f})")

    # Clean up temporary files
    #os.remove(spectrogram_path)

# Example usage
classify_with_percentages('C:/Desktop/AI Project/Attempt#3/RealTimeTest/81013008.wav')  # Replace with the path to your .wav file

