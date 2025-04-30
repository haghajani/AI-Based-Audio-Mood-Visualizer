from sklearn.model_selection import train_test_split
from extract_feature import get_features
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.utils import np_utils
import pickle

def get_training_data():
    df = get_features()

    # Split between train and test 
    X_train, X_test, y_train, y_test = train_test_split(df.drop(['path','label'],axis=1)
                                                        , df.label
                                                        , test_size=0.25
                                                        , shuffle=True
                                                        , random_state=42
                                                    )

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    X_train = (X_train - mean)/std
    X_test = (X_test - mean)/std

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # one hot encode the target 
    lb = LabelEncoder()
    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))

    filename = 'labels'
    outfile = open(filename,'wb')
    pickle.dump(lb,outfile)
    outfile.close()

    print("done preparing data")
    return X_train, y_train, X_test, y_test
