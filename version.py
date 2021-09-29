import joblib
import numpy as np
import os
import pandas as pd
import sys
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense          
from keras.models import load_model, Sequential
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import pyplot
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *   
from PyQt5.QtWidgets import *
import sklearn
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import inspect
from PyQt5 import Qt

# vers = ['%s = %s' % (k,v) for k,v in vars(Qt).items() if k.lower().find('version') >= 0 and not inspect.isbuiltin(v)]
# print('\n'.join(sorted(vers)))

sklearn.show_versions()