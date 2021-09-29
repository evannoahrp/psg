import joblib
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# load dataset
df = pd.read_csv("Data PSG 2019.csv")
dataset = df.values
xtest = dataset[:,0:5]
ytest = dataset[:,5:8]


# load scaler
scaler_filename = "scaler.save"
scaler = joblib.load(scaler_filename)
testpreprocess = scaler.transform(xtest)
testdf = pd.DataFrame(testpreprocess, columns = df.columns[0:5])
inputpreprocess = testdf.values
inputxa = inputpreprocess[:, [0, 1, 4]]
inputxb = inputpreprocess


#kelasifikasi
kelas = 2
if(kelas == 0):
    inputx = inputxa
    best_model = 'best_model_BBU.h5'
else:
    inputx = inputxb
    if(kelas == 1):
        best_model = 'best_model_TBU.h5'
    else:
        best_model = 'best_model_BBTB.h5'


#encode output
ytrue = ytest[:,kelas]
encoder = LabelEncoder()
encoder.fit(ytrue)
encoded = encoder.transform(ytrue)
ytrue = np_utils.to_categorical(encoded)
ytrue = np.argmax(ytrue, axis = 1)


#load the saved model & predict classes
saved_model = load_model(best_model)
ypred = saved_model.predict_classes(inputx)


#accuracy of the predicted values
acc = accuracy_score(ytrue, ypred) * 100
print(acc)