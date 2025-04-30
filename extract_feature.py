import os
import pandas as pd
import numpy as np
import librosa
import pickle
from organize_data import get_RAV

def get_mfccs(path):
    X, sample_rate = librosa.load(path
                                  , res_type='kaiser_fast'
                                  ,duration=2.5
                                  ,sr=44100
                                  ,offset=0.5
                                 )
    sample_rate = np.array(sample_rate)
    
    mfccs = np.mean(librosa.feature.mfcc(y=X, 
                                        sr=sample_rate, 
                                        n_mfcc=13),
                    axis=0)
    return mfccs

def get_features():

    RAV_df = get_RAV()

    df = pd.DataFrame(columns=['feature'])

    # loop feature extraction over the entire dataset
    counter=0
    for index,path in enumerate(RAV_df.path):
        mfccs = get_mfccs(path)
        df.loc[counter] = [mfccs]
        counter=counter+1   

    df = pd.concat([RAV_df,pd.DataFrame(df['feature'].values.tolist())],axis=1)
    df=df.fillna(0)
    return(df)