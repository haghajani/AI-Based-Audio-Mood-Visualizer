from extract_feature import get_mfccs
import pandas as pd
from model import get_mode_to_predit
import pickle
import numpy as np
import librosa

def predict_emotion(path, json, weight, learning_rate):
    signal, sr = librosa.load(path
                              ,res_type='kaiser_fast'
                              ,sr=44100
                              ,offset=0.5
                             )
    
    total_duration = librosa.get_duration(y=signal, sr=sr)
    segment_duration = 2.5

    model = get_mode_to_predit(json, weight, learning_rate)

    # Load label binarizer
    with open('labels', 'rb') as infile:
        lb = pickle.load(infile)
    
    predictions = []

    for i in range(0, int(total_duration), int(segment_duration)):
        start_sample = int(i * sr)
        end_sample = int((i + segment_duration) * sr)

        # Handle edge cases where audio is shorter than segment
        if end_sample > len(signal):
            break

        segment = signal[start_sample:end_sample]
        sr = np.array(sr)

        # Extract MFCCs for the segment
        mfccs = np.mean(librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13), axis=0)
        mfccs_df = pd.DataFrame(data=mfccs).T
        mfccs_df = np.expand_dims(mfccs_df, axis=2)

        # Predict emotion
        pred = model.predict(mfccs_df, batch_size=16, verbose=1)
        final = pred.argmax(axis=1)
        final = final.astype(int).flatten()
        emotion = lb.inverse_transform(final)[0]
        print(emotion)
        predictions.append(emotion)

    return predictions