
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import librosa
#import matplotlib.pyplot as plt
import pandas as pd
import os
import librosa
import numpy as np
from tqdm import tqdm
import pickle

scnt = 0
ccnt = 0

def features_extractor(file):
    audio, sample_rate = librosa.load(file)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    return mfccs_scaled_features
extracted_features = []

cry_dir = "./Crying baby/"

for file in os.scandir(cry_dir):
    final_class_label = 1
    data = features_extractor(file)
    extracted_features.append([data, final_class_label])
    print(ccnt)
    ccnt += 1

silence_dir = "./Silence/"

for file in os.scandir(silence_dir):
    final_class_label = 0
    data = features_extractor(file)
    extracted_features.append([data, final_class_label])
    print(scnt)
    scnt += 1

extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])

X=np.array(extracted_features_df['feature'].tolist())
Y=np.array(extracted_features_df['class'].tolist())

X_train,X_test,y_train,y_test=train_test_split(X, Y,test_size=0.2,random_state=0)

model = KNeighborsClassifier()

model.fit(X_train, y_train)
file = open("AudioModel.pkl","wb")
pickle.dump(model, file)
print("Success")

prediction = model.predict(X_test)
print(prediction)
print(y_test)
filename="New.wav"
audio, sample_rate = librosa.load(filename, res_type='kaiser_fast')
print(sample_rate)
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

print(mfccs_scaled_features)
mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
print(mfccs_scaled_features)
print(mfccs_scaled_features.shape)

pdc = model.predict(mfccs_scaled_features)
print(pdc)