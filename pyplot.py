# mlp overfit on the moons dataset
import pandas
import numpy as np
from keras.layers import Dense
from keras.models import load_model, Sequential
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn import preprocessing
from keras.utils.vis_utils import plot_model
from keras.utils.vis_utils import model_to_dot
# load dataset
dataframe = pandas.read_csv("Data PSG 2017.csv")
dataset = dataframe.values
#X = dataset[:,0:4].astype(float)
X = dataset[:,0:5]
Y = dataset[:,5:8]

x_scaler = preprocessing.MinMaxScaler().fit(X)
x_scaled = x_scaler.transform(X)
newdf = pandas.DataFrame(x_scaled, columns = dataframe.columns[0:5])
inputpreprocess = newdf.values
x = inputpreprocess[:, [0, 1, 4]]
# x = inputpreprocess

#set output
outputY = Y[:,0]

encoder = LabelEncoder()
encoder.fit(outputY)
encoded = encoder.transform(outputY)
y = np_utils.to_categorical(encoded)
y_true = np.argmax(y, axis = 1)

kfold = KFold(n_splits = 5)
es = EarlyStopping(monitor = 'loss', mode = 'min', verbose = 1, patience = 50)
mc = ModelCheckpoint("best_model_BBUv2.h5", monitor = 'accuracy', mode = 'max', verbose = 1, save_best_only=True)

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim = len(x[0]), activation = 'tanh'))
    model.add(Dense(15, activation = 'tanh'))
    model.add(Dense(len(y[0]), activation = 'softmax'))
    # Compile model 
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model           

def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements

accu = []
avg_loss = []
for train, val in kfold.split(x):
    trainX, testX = x[train, :], x[val, :]
    trainy, testy = y[train], y[val]
    model = baseline_model()

    # fit model
    history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=1000, verbose=0, callbacks=[es, mc])
    
    # evaluate the model
    test_loss, test_acc = model.evaluate(testX, testy, verbose=0)
    print('Acc: %.3f, Loss: %.3f' % (test_acc, test_loss))
    y_true = np.argmax(testy, axis = 1)    
    y_pred = model.predict(testX)
    y_pred = np.argmax(y_pred, axis = 1)

    # confusion matrix & accuracy of the predicted values
    conf_mat = confusion_matrix(y_true, y_pred)
    print(conf_mat)

    acc = accuracy(conf_mat) * 100
    print(acc)
    accu.append(acc)

    losss = test_loss
    print(losss)
    avg_loss.append(losss)

    # plot loss training history
    pyplot.subplot(211)
    pyplot.title('Cross-Entropy Loss', pad=-40)
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    # plot accuracy training history
    pyplot.subplot(212)
    pyplot.title('Accuracy', pad=-40)
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.show()
