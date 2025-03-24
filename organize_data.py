import os
import shutil

# Define paths
DATASET_PATH = "1/AudioWAV/"  # Change this to your dataset root folder
DEST_PATH = "dataset/"

# Emotion Mapping
EMOTION_MAP = {
    "NEU": "neutral",
    "HAP": "happy",
    "SAD": "sad",
    "ANG": "angry",
    "FEA": "fearful",
    "DIS": "disgust"
}

# Ensure destination directories exist
for emotion in EMOTION_MAP.values():
    os.makedirs(os.path.join(DEST_PATH, emotion), exist_ok=True)

# Process files
for file in os.listdir(DATASET_PATH):
    if file.endswith(".wav"):
        parts = file.split("_")
        
        if len(parts) == 4:
            emotion_code = parts[2]   # Emotion (01 to 08)
            emotion_folder = EMOTION_MAP.get(emotion_code, "unknown")
            src = os.path.join(DATASET_PATH, file)
            dest = os.path.join(DEST_PATH, emotion_folder, file)

            shutil.move(src, dest)
            print(f"Moved {file} to {emotion_folder}/")

