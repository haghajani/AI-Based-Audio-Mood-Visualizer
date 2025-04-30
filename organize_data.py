import os
import pandas as pd

def get_RAV():
    RAV = "RAVDESS/audio_song_actors_01-24/"

    dir_list = os.listdir(RAV)
    dir_list.sort()

    emotion = []
    path = []
    for i in dir_list:
        fname = os.listdir(RAV + i)
        for f in fname:
            part = f.split('.')[0].split('-')
            emotion.append(int(part[2]))
            path.append(RAV + i + '/' + f)
        
    RAV_df = pd.DataFrame(emotion)
    RAV_df = RAV_df.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'})
    RAV_df.columns = ['label']
    RAV_df = pd.concat([RAV_df,pd.DataFrame(path, columns = ['path'])],axis=1)
    return RAV_df