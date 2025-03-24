import os
import pickle
import tensorflow_hub as hub
import numpy as np
import librosa

def extract_vggish_features(audio_file, sr_vggish = 16000):
    # Load audio and resample
    y, sr = librosa.load(audio_file, sr=sr_vggish, mono=True)

    num_samples = sr  # 1 second of audio
    num_segments = len(y) // num_samples
    
    features = []
    
    for i in range(num_segments):
        segment = y[i * num_samples: (i + 1) * num_samples]
        if len(segment) < num_samples:
            continue  # Skip short segments
        
        # Convert to the expected VGGish input shape
        # segment = segment.reshape(-1)  # Reshape for model input
        embedding = vggish_model(segment)  # Extract features
        features.append(embedding.numpy())
    
    return np.mean(features, axis=0)  # Average over segments
    
    # embeddings = np.array(vggish_model(y))
    # print(embeddings.shape)
    # return embeddings

# Load VGGish model from TensorFlow Hub
vggish_model = hub.load('https://tfhub.dev/google/vggish/1')

DATASET_PATH = "dataset"
EMOTIONS = ["happy", "sad", "angry", "neutral", "disgust", "fearful"]

features = []
labels = []

for emotion in EMOTIONS:
    emotion_folder = os.path.join(DATASET_PATH, emotion)
    for file in os.listdir(emotion_folder):
        file_path = os.path.join(emotion_folder, file)
        try:
            feature_vector = extract_vggish_features(file_path)
            features.append(feature_vector.reshape(-1))
            labels.append(EMOTIONS.index(emotion))  # Convert label to number
        except Exception as e:
            print(f"Skipping {file}: {e}")

# Save extracted features
with open("features.pkl", "wb") as f:
    pickle.dump((features, labels), f)
