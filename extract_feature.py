import os
import pandas as pd
import numpy as np
import librosa
import pickle

def get_mfccs(path):
    X, sample_rate = librosa.load(path
                                  , res_type='kaiser_fast'
                                  ,duration=2.5
                                  ,sr=44100
                                  ,offset=0.5
                                 )
    sample_rate = np.array(sample_rate)
    
    # mean as the feature. Could do min and max etc as well. 
    mfccs = np.mean(librosa.feature.mfcc(y=X, 
                                        sr=sample_rate, 
                                        n_mfcc=13),
                    axis=0)
    return mfccs

def get_features():
    DATASET_PATH = "dataset"
    EMOTIONS = ["happy", "sad", "angry", "neutral", "disgust", "fearful"]

    df = pd.DataFrame(columns=['file', 'label', 'feature'])
    counter = 0

    for emotion in EMOTIONS:
        emotion_folder = os.path.join(DATASET_PATH, emotion)
        for file in os.listdir(emotion_folder):
            file_path = os.path.join(emotion_folder, file)
            try:
                feature_vector = get_mfccs(file_path)
                df.loc[counter] = [file_path, emotion, feature_vector]
                counter=counter+1  
            except Exception as e:
                print(f"Skipping {file}: {e}")
    df = pd.concat([df,pd.DataFrame(df['feature'].values.tolist())],axis=1)
    df = df.drop('feature', axis=1)
    df = df.fillna(0)
    return df