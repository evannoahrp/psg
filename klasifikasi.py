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
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

class TableModel(QtCore.QAbstractTableModel):

    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == QtCore.Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]
    
    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return str(self._data.columns[section])

            if orientation == QtCore.Qt.Vertical:
                return str(self._data.index[section])

class Ui_MainWindow(QtWidgets.QWidget):
 
    #fungsi untuk import file csv
    def import_raw_data(self):
        try:
            #mengambil path dari file csv
            filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', r"C:\\Users\\Sperbia\\Documents", "CSV Files (*.csv)")
            path = filename[0]
            #print path di textfield
            self.TextEditRD.setText(path)
            #membaca file csv berdasarkan path dari user kedalam dataframe
            self.df = pd.read_csv(f"{path}")
            #memasukan isi dari dataframe kedalam tabel
            modelRD = TableModel(self.df)
            self.tableRD.setModel(modelRD)
        
        except:
            pass
    
    #fungsi untuk proses preprocessing
    def preprocessing_raw_data(self):
        try:
            #memisahkan data input dan data output dari dataframe
            dataset = self.df.values
            x = dataset[:,0:5]
            self.y = dataset[:,5:8]
            
            #melakukan preprocessing kepada data input dengan normalisasi minmax
            x_scaler = preprocessing.MinMaxScaler().fit(x)
            scaler_filename = "scaler.save"
            joblib.dump(x_scaler, scaler_filename)
            x_scaled = x_scaler.transform(x)
            
            #hasil preprocessing input dimasukan kedalam dataframe
            newdf = pd.DataFrame(x_scaled, columns = self.df.columns[0:5])
            
            #memasukan dataframe kedalam tabel
            modelPreprocessing = TableModel(newdf)
            self.tablePreprocessing.setModel(modelPreprocessing)
            
            #penyesuaian ukuran column dari tabel setelah memasukan dataframe
            header = self.tablePreprocessing.horizontalHeader()
            header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
            header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
            header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
            
            #membagi input berdasarkan kelas
            inputpreprocess = newdf.values
            self.inputA = inputpreprocess[:, [0, 1, 4]]
            self.inputB = inputpreprocess
        
        except:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('Belum Import Data')
            msg.setWindowTitle("Error")
            msg.exec_()

    def modeljst(self):
        try:
            kelas = self.comboBoxKlasifikasi.currentIndex()
            if(kelas == 0):
                input = self.inputA
                best_model = 'best_model_BBU1.h5'
            else:
                input = self.inputB
                if(kelas == 1):
                    best_model = 'best_model_TBU1.h5'
                else:
                    best_model = 'best_model_BBTB1.h5'

            #mengambil output berdasarkan kelas
            outputY = self.y[:,kelas]
            #memanggil class encoder
            encoder = LabelEncoder()
            #transformasi output kedalam bentuk kategorial
            encoder.fit(outputY)
            encoded = encoder.transform(outputY)
            output = np_utils.to_categorical(encoded)

            #memanggil parameter pelatihan JST dari user
            hLayer = self.comboBoxHL.currentIndex()
            neuron1 = int(self.comboBoxNeuron1.currentText())
            aktivasi1 = str(self.comboBoxA1.currentText())
            neuron2 = int(self.comboBoxNeuron2.currentText())
            aktivasi2 = str(self.comboBoxA2.currentText())
            training = str(self.comboBoxTrain.currentText())

            #membangun model
            def baseline_model():
                # create model
                model = Sequential()
                model.add(Dense(neuron1, input_dim = len(input[0]), activation = aktivasi1))
                if(hLayer == 1):
                    model.add(Dense(neuron2, activation = aktivasi2))
                model.add(Dense(len(output[0]), activation = 'softmax'))
                
                # Compile model
                model.compile(loss = 'categorical_crossentropy', optimizer = training, metrics = ['accuracy'])
                return model
            
            def accuracy(confusion_matrix):
                diagonal_sum = confusion_matrix.trace()
                sum_of_all_elements = confusion_matrix.sum()
                return diagonal_sum / sum_of_all_elements

            kfold = KFold(n_splits = 5)
            es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 10)
            mc = ModelCheckpoint(best_model, monitor = 'val_accuracy', mode = 'max', verbose = 1, save_best_only=True)
            accu = []
            avg_loss = []

            for train, val in kfold.split(input):
                trainX, testX = input[train, :], input[val, :]
                trainy, testy = output[train], output[val]
                model = baseline_model()

                # fit model
                model.fit(trainX, trainy, validation_data=(testX, testy), epochs=1000, verbose=0, callbacks=[es, mc])
                
                # evaluate the model
                test_loss, test_acc = model.evaluate(testX, testy, verbose=0)
                print('Acc: %.3f, Loss: %.3f' % (test_acc, test_loss))
                y_true = np.argmax(testy, axis = 1)    
                y_pred = model.predict(testX)
                y_pred = np.argmax(y_pred, axis = 1)
                
                # confusion matrix & accuracy of the predicted values
                conf_mat = confusion_matrix(y_true, y_pred)
                acc = accuracy(conf_mat) * 100
                print(conf_mat)
                print(acc)
                
                accu.append(acc)
                avg_loss.append(test_loss)

            acc = sum(accu)/kfold.n_splits
            self.textEditAkurasi.setText('%.3f' % (acc))
            print(sum(avg_loss)/kfold.n_splits)
        
        except:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('Belum Ada Data Preprocessing')
            msg.setWindowTitle("Error")
            msg.exec_()

    def reset_model(self):
        self.comboBoxHL.setCurrentIndex(0)
        self.comboBoxNeuron1.setCurrentIndex(0)
        self.comboBoxA1.setCurrentIndex(0)
        self.comboBoxNeuron2.setCurrentIndex(0)
        self.comboBoxA2.setCurrentIndex(0)
        self.comboBoxTrain.setCurrentIndex(0)
        self.textEditAkurasi.setText("")

    def Message_UDT(self):
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("Uji Data Tunggal")
        msg.setText("Pilih Klasifikasi")
        msg.setIcon(QtWidgets.QMessageBox.Question)
        msg.setStandardButtons(QtWidgets.QMessageBox.Cancel)
        msg.addButton("BB/U", QtWidgets.QMessageBox.NoRole)
        msg.addButton("TB/U", QtWidgets.QMessageBox.NoRole)
        msg.addButton("BB/TB", QtWidgets.QMessageBox.NoRole)
        msg.buttonClicked.connect(self.uji_data_tunggal)
        msg.exec_()

    def uji_data_tunggal(self, i):
        try:
            kelas = i.text()
            if(kelas != "Cancel"):
                jKelamin = int(self.comboBoxJK.currentIndex()) + 1
                bBadan = self.textEditBB.toPlainText()
                PB_TB = self.textEditPBTB.toPlainText()
                pDiukur = int(self.comboBoxPD.currentIndex()) + 3
                umur = self.textEditUmur.toPlainText()

                scaler_filename = "scaler.save"
                scaler = joblib.load(scaler_filename)
                manual_input = pd.DataFrame({'Js.L/P':[jKelamin], 'Berat B.':[bBadan], 'PB / TB':[PB_TB], 'Posisi diukur':[pDiukur], 'Umur':[umur]})
                manualpreprocess = scaler.transform(manual_input)
                print(manualpreprocess)

                if(kelas == "BB/U"):
                    xNew = manualpreprocess[:, [0, 1, 4]].astype(float)
                    best_model = 'best_model_BBU.h5'
                else:
                    xNew = manualpreprocess.astype(float)
                    if(kelas == "TB/U"):
                        best_model = 'best_model_TBU.h5'
                    else:
                        best_model = 'best_model_BBTB.h5'

                #load the saved model
                saved_model = load_model(best_model)
                #predict inputan data baru
                yNew = saved_model.predict_classes(xNew)

                if(kelas == "BB/U"):
                    if(yNew == 0):
                        klf = "Baik"
                    elif(yNew == 1):
                        klf = "Buruk"
                    elif(yNew == 2):
                        klf = "Kurang"
                    else:
                        klf = "Lebih"
                elif(kelas == "TB/U"):
                    if(yNew == 0):
                        klf = "Normal"
                    elif(yNew == 1):
                        klf = "Pendek"
                    elif(yNew == 2):
                        klf = "Sangat Pendek"
                    else:
                        klf = "Tinggi"
                else:
                    if(yNew == 0):
                        klf = "Gemuk"
                    elif(yNew == 1):
                        klf = "Kurus"
                    elif(yNew == 2):
                        klf = "Normal"
                    else:
                        klf = "Sangat Kurus"

                self.textEditSG.setText(klf)
            else:
                pass
            
        except:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText('input tidak valid')
            msg.setWindowTitle("Error")
            msg.exec_()
        

    def resetUDT(self):
        self.comboBoxJK.setCurrentIndex(0)
        self.textEditBB.setText("")
        self.textEditPBTB.setText("")
        self.comboBoxPD.setCurrentIndex(0)
        self.textEditUmur.setText("")
        self.textEditSG.setText("")

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1000, 710)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")


        #Label judul
        self.labelJudul1 = QtWidgets.QLabel(self.centralwidget)
        self.labelJudul1.setGeometry(QtCore.QRect(230, 15, 540, 20))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.labelJudul1.setFont(font)
        self.labelJudul1.setObjectName("labelJudul1")
        self.labelJudul2 = QtWidgets.QLabel(self.centralwidget)
        self.labelJudul2.setGeometry(QtCore.QRect(245, 45, 510, 20))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.labelJudul2.setFont(font)
        self.labelJudul2.setObjectName("labelJudul2")


        #Line pemisah judul dan group box
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(0, 69, 1001, 31))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")


        #Group Box raw data
        self.groupBoxRD = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBoxRD.setGeometry(QtCore.QRect(20, 100, 610, 310))
        self.groupBoxRD.setObjectName("groupBoxRD")
        #Tabel raw data
        self.tableRD = QtWidgets.QTableView(self.groupBoxRD)
        self.tableRD.setGeometry(QtCore.QRect(20, 20, 570, 225))
        self.tableRD.setObjectName("tableRD")
        #Text Edit raw data
        self.TextEditRD = QtWidgets.QTextEdit(self.groupBoxRD)
        self.TextEditRD.setGeometry(QtCore.QRect(20, 265, 350, 25))
        self.TextEditRD.setObjectName("TextEditRD")
        self.TextEditRD.setReadOnly(True)
        #Button import raw data
        self.buttonImportRD = QtWidgets.QPushButton(self.groupBoxRD)
        self.buttonImportRD.setGeometry(QtCore.QRect(390, 265, 90, 25))
        self.buttonImportRD.setObjectName("buttonImportRD")
        self.buttonImportRD.clicked.connect(self.import_raw_data)
        #Button preprocessing raw data
        self.buttonPreprocessingRD = QtWidgets.QPushButton(self.groupBoxRD)
        self.buttonPreprocessingRD.setGeometry(QtCore.QRect(500, 265, 90, 25))
        self.buttonPreprocessingRD.setObjectName("buttonPreprocessingRD")
        self.buttonPreprocessingRD.clicked.connect(self.preprocessing_raw_data)


        #Group Box preprocessing
        self.groupBoxPreprocessing = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBoxPreprocessing.setGeometry(QtCore.QRect(20, 420, 610, 270))
        self.groupBoxPreprocessing.setObjectName("groupBoxPreprocessing")
        #Tabel preprocessing
        self.tablePreprocessing = QtWidgets.QTableView(self.groupBoxPreprocessing)
        self.tablePreprocessing.setGeometry(QtCore.QRect(20, 20, 570, 225))
        self.tablePreprocessing.setObjectName("tablePreprocessing")


        #Group Box pelatihan
        self.groupBoxPelatihan = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBoxPelatihan.setGeometry(QtCore.QRect(650, 100, 330, 270))
        self.groupBoxPelatihan.setObjectName("groupBoxPelatihan")
        #Label hidden layer
        self.labelHL = QtWidgets.QLabel(self.groupBoxPelatihan)
        self.labelHL.setGeometry(QtCore.QRect(20, 20, 70, 25))
        self.labelHL.setObjectName("labelHL")
        #Combo Box hidden layer
        self.comboBoxHL = QtWidgets.QComboBox(self.groupBoxPelatihan)
        self.comboBoxHL.setGeometry(QtCore.QRect(100, 20, 40, 25))
        self.comboBoxHL.setObjectName("comboBoxHL")
        self.comboBoxHL.addItem("")
        self.comboBoxHL.addItem("")
        #Label fungsi aktivasi hidden layer pertama
        self.labelA1 = QtWidgets.QLabel(self.groupBoxPelatihan)
        self.labelA1.setGeometry(QtCore.QRect(180, 60, 70, 25))
        self.labelA1.setObjectName("labelA1")
        #Combo Box fungsi aktivasi hidden layer pertama
        self.comboBoxA1 = QtWidgets.QComboBox(self.groupBoxPelatihan)
        self.comboBoxA1.setGeometry(QtCore.QRect(240, 60, 70, 25))
        self.comboBoxA1.setObjectName("comboBoxA1")
        self.comboBoxA1.addItem("")
        self.comboBoxA1.addItem("")
        #Label fungsi aktivasi hidden layer kedua
        self.labelA2 = QtWidgets.QLabel(self.groupBoxPelatihan)
        self.labelA2.setGeometry(QtCore.QRect(180, 100, 70, 25))
        self.labelA2.setObjectName("labelA2")
        #Combo Box fungsi aktivasi hidden layer kedua
        self.comboBoxA2 = QtWidgets.QComboBox(self.groupBoxPelatihan)
        self.comboBoxA2.setGeometry(QtCore.QRect(240, 100, 70, 25))
        self.comboBoxA2.setObjectName("comboBoxA2")
        self.comboBoxA2.addItem("")
        self.comboBoxA2.addItem("")
        #Label fungsi training
        self.labelTrain = QtWidgets.QLabel(self.groupBoxPelatihan)
        self.labelTrain.setGeometry(QtCore.QRect(180, 20, 70, 25))
        self.labelTrain.setObjectName("labelTrain")
        #Combo Box fungsi training
        self.comboBoxTrain = QtWidgets.QComboBox(self.groupBoxPelatihan)
        self.comboBoxTrain.setGeometry(QtCore.QRect(240, 20, 70, 25))
        self.comboBoxTrain.setObjectName("comboBoxTrain")
        self.comboBoxTrain.addItem("")
        self.comboBoxTrain.addItem("")
        self.comboBoxTrain.addItem("")
        #Label neuron hidden layer pertama
        self.labelNeuron1 = QtWidgets.QLabel(self.groupBoxPelatihan)
        self.labelNeuron1.setGeometry(QtCore.QRect(20, 60, 50, 25))
        self.labelNeuron1.setObjectName("labelNeuron1")
        #Combo Box neuron hidden layer pertama
        self.comboBoxNeuron1 = QtWidgets.QComboBox(self.groupBoxPelatihan)
        self.comboBoxNeuron1.setGeometry(QtCore.QRect(100, 60, 40, 25))
        self.comboBoxNeuron1.setObjectName("comboBoxNeuron1")
        self.comboBoxNeuron1.addItem("")
        self.comboBoxNeuron1.addItem("")
        self.comboBoxNeuron1.addItem("")
        self.comboBoxNeuron1.addItem("")
        self.comboBoxNeuron1.addItem("")
        self.comboBoxNeuron1.addItem("")
        self.comboBoxNeuron1.addItem("")
        self.comboBoxNeuron1.addItem("")
        self.comboBoxNeuron1.addItem("")
        self.comboBoxNeuron1.addItem("")
        #Label neuron hidden layer kedua
        self.labelNeuron2 = QtWidgets.QLabel(self.groupBoxPelatihan)
        self.labelNeuron2.setGeometry(QtCore.QRect(20, 100, 50, 30))
        self.labelNeuron2.setObjectName("labelNeuron2")
        #Combo Box neuron hidden layer kedua
        self.comboBoxNeuron2 = QtWidgets.QComboBox(self.groupBoxPelatihan)
        self.comboBoxNeuron2.setGeometry(QtCore.QRect(100, 100, 40, 25))
        self.comboBoxNeuron2.setObjectName("comboBoxNeuron2")
        self.comboBoxNeuron2.addItem("")
        self.comboBoxNeuron2.addItem("")
        self.comboBoxNeuron2.addItem("")
        self.comboBoxNeuron2.addItem("")
        self.comboBoxNeuron2.addItem("")
        self.comboBoxNeuron2.addItem("")
        self.comboBoxNeuron2.addItem("")
        self.comboBoxNeuron2.addItem("")
        self.comboBoxNeuron2.addItem("")
        self.comboBoxNeuron2.addItem("")
        #Label klasifikasi
        self.labelKlasifikasi = QtWidgets.QLabel(self.groupBoxPelatihan)
        self.labelKlasifikasi.setGeometry(QtCore.QRect(20, 140, 50, 30))
        self.labelKlasifikasi.setObjectName("labelKlasifikasi")
        #Combo Box klasifikasi
        self.comboBoxKlasifikasi = QtWidgets.QComboBox(self.groupBoxPelatihan)
        self.comboBoxKlasifikasi.setGeometry(QtCore.QRect(100, 140, 55, 25))
        self.comboBoxKlasifikasi.setObjectName("comboBoxKlasifikasi")
        self.comboBoxKlasifikasi.addItem("")
        self.comboBoxKlasifikasi.addItem("")
        self.comboBoxKlasifikasi.addItem("")
        #Button proses pelatihan
        self.buttonProsesPelatihan = QtWidgets.QPushButton(self.groupBoxPelatihan)
        self.buttonProsesPelatihan.setGeometry(QtCore.QRect(100, 180, 90, 25))
        self.buttonProsesPelatihan.setObjectName("buttonProsesPelatihan")
        self.buttonProsesPelatihan.clicked.connect(self.modeljst)
        #Button reset pelatihan
        self.buttonResetPelatihan = QtWidgets.QPushButton(self.groupBoxPelatihan)
        self.buttonResetPelatihan.setGeometry(QtCore.QRect(200, 180, 90, 25))
        self.buttonResetPelatihan.setObjectName("buttonResetPelatihan")
        self.buttonResetPelatihan.clicked.connect(self.reset_model)
        #Label akurasi
        self.labelAkurasi1 = QtWidgets.QLabel(self.groupBoxPelatihan)
        self.labelAkurasi1.setGeometry(QtCore.QRect(20, 220, 70, 25))
        self.labelAkurasi1.setObjectName("labelAkurasi1")
        #Text Edit akurasi
        self.textEditAkurasi = QtWidgets.QTextEdit(self.groupBoxPelatihan)
        self.textEditAkurasi.setGeometry(QtCore.QRect(100, 220, 190, 25))
        self.textEditAkurasi.setObjectName("textEditAkurasi")
        self.textEditAkurasi.setReadOnly(True)
        #Label simbol "%" akurasi
        self.labelAkurasi2 = QtWidgets.QLabel(self.groupBoxPelatihan)
        self.labelAkurasi2.setGeometry(QtCore.QRect(295, 220, 30, 25))
        self.labelAkurasi2.setObjectName("labelAkurasi2")


        #Group Box uji data tunggal
        self.groupBoxUDT = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBoxUDT.setGeometry(QtCore.QRect(650, 380, 330, 310))
        self.groupBoxUDT.setObjectName("groupBoxUDT")
        #Label jenis kelamin
        self.labelJK = QtWidgets.QLabel(self.groupBoxUDT)
        self.labelJK.setGeometry(QtCore.QRect(20, 20, 80, 25))
        self.labelJK.setAutoFillBackground(True)
        self.labelJK.setObjectName("labelJK")
        #Combo Box jenis kelamin
        self.comboBoxJK = QtWidgets.QComboBox(self.groupBoxUDT)
        self.comboBoxJK.setGeometry(QtCore.QRect(140, 20, 170, 25))
        self.comboBoxJK.setObjectName("comboBoxJK")
        self.comboBoxJK.addItem("")
        self.comboBoxJK.addItem("")
        #Label berat badan
        self.labelBB = QtWidgets.QLabel(self.groupBoxUDT)
        self.labelBB.setGeometry(QtCore.QRect(20, 60, 80, 25))
        self.labelBB.setAutoFillBackground(True)
        self.labelBB.setObjectName("labelBB")
        #Text Edit berat badan
        self.textEditBB = QtWidgets.QTextEdit(self.groupBoxUDT)
        self.textEditBB.setGeometry(QtCore.QRect(140, 60, 170, 25))
        self.textEditBB.setObjectName("textEditBB")
        #Label PB/TB
        self.labelPBTB = QtWidgets.QLabel(self.groupBoxUDT)
        self.labelPBTB.setGeometry(QtCore.QRect(20, 100, 80, 25))
        self.labelPBTB.setAutoFillBackground(True)
        self.labelPBTB.setObjectName("labelPBTB")
        #Text Edit PB/TB
        self.textEditPBTB = QtWidgets.QTextEdit(self.groupBoxUDT)
        self.textEditPBTB.setGeometry(QtCore.QRect(140, 100, 170, 25))
        self.textEditPBTB.setObjectName("textEditPBTB")
        #Label posisi diukur
        self.labelPD = QtWidgets.QLabel(self.groupBoxUDT)
        self.labelPD.setGeometry(QtCore.QRect(20, 140, 80, 25))
        self.labelPD.setAutoFillBackground(True)
        self.labelPD.setObjectName("labelPD")
        #Combo Box posisi diukur
        self.comboBoxPD = QtWidgets.QComboBox(self.groupBoxUDT)
        self.comboBoxPD.setGeometry(QtCore.QRect(140, 140, 170, 25))
        self.comboBoxPD.setObjectName("comboBoxPD")
        self.comboBoxPD.addItem("")
        self.comboBoxPD.addItem("")
        #Label umur
        self.labelUmur = QtWidgets.QLabel(self.groupBoxUDT)
        self.labelUmur.setGeometry(QtCore.QRect(20, 180, 80, 25))
        self.labelUmur.setAutoFillBackground(True)
        self.labelUmur.setObjectName("labelUmur")
        #Text Edit umur
        self.textEditUmur = QtWidgets.QTextEdit(self.groupBoxUDT)
        self.textEditUmur.setGeometry(QtCore.QRect(140, 180, 170, 25))
        self.textEditUmur.setObjectName("textEditUmur")
        #Button proses uji data tunggal
        self.buttonProsesUDT = QtWidgets.QPushButton(self.groupBoxUDT)
        self.buttonProsesUDT.setGeometry(QtCore.QRect(140, 220, 80, 25))
        self.buttonProsesUDT.setObjectName("buttonProsesUDT")
        self.buttonProsesUDT.clicked.connect(self.Message_UDT)
        #Button reset uji data tunggal
        self.buttonResetUDT = QtWidgets.QPushButton(self.groupBoxUDT)
        self.buttonResetUDT.setGeometry(QtCore.QRect(230, 220, 80, 25))
        self.buttonResetUDT.setObjectName("buttonResetUDT")
        self.buttonResetUDT.clicked.connect(self.resetUDT)
        #Label status gizi
        self.labelSG = QtWidgets.QLabel(self.groupBoxUDT)
        self.labelSG.setGeometry(QtCore.QRect(20, 260, 80, 25))
        self.labelSG.setAutoFillBackground(True)
        self.labelSG.setObjectName("labelSG")
        #Text Edit status gizi
        self.textEditSG = QtWidgets.QTextEdit(self.groupBoxUDT)
        self.textEditSG.setGeometry(QtCore.QRect(140, 260, 170, 25))
        self.textEditSG.setObjectName("textEditSG")
        self.textEditSG.setReadOnly(True)


        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Klasifikasi Status Gizi Balita"))
        self.labelJudul1.setText(_translate("MainWindow", "KLASIFIKASI STATUS GIZI BALITA MENGGUNAKAN"))
        self.labelJudul2.setText(_translate("MainWindow", "JARINGAN SYARAF TIRUAN BACKPROPAGATION"))
        self.groupBoxRD.setTitle(_translate("MainWindow", "Raw Data"))
        self.buttonPreprocessingRD.setText(_translate("MainWindow", "Preprocessing"))
        self.buttonImportRD.setText(_translate("MainWindow", "Import Data"))
        self.groupBoxUDT.setTitle(_translate("MainWindow", "Uji Data Tunggal"))
        self.labelJK.setText(_translate("MainWindow", "Js.L/P"))
        self.comboBoxJK.setItemText(0, _translate("MainWindow", "Laki-laki"))
        self.comboBoxJK.setItemText(1, _translate("MainWindow", "Perempuan"))
        self.buttonProsesUDT.setText(_translate("MainWindow", "Proses"))
        self.buttonResetUDT.setText(_translate("MainWindow", "Reset"))
        self.labelSG.setText(_translate("MainWindow", "Status Gizi"))
        self.labelUmur.setText(_translate("MainWindow", "Umur"))
        self.labelPD.setText(_translate("MainWindow", "Posisi Diukur"))
        self.labelPBTB.setText(_translate("MainWindow", "PB / TB"))
        self.labelBB.setText(_translate("MainWindow", "Berat B."))
        self.comboBoxPD.setItemText(0, _translate("MainWindow", "Terlentang"))
        self.comboBoxPD.setItemText(1, _translate("MainWindow", "Berdiri"))
        self.groupBoxPreprocessing.setTitle(_translate("MainWindow", "Hasil Preprocessing"))
        self.groupBoxPelatihan.setTitle(_translate("MainWindow", "Pelatihan"))
        self.labelHL.setText(_translate("MainWindow", "Hidden Layer"))
        self.labelA1.setText(_translate("MainWindow", "Aktivasi 1"))
        self.comboBoxHL.setItemText(0, _translate("MainWindow", "1"))
        self.comboBoxHL.setItemText(1, _translate("MainWindow", "2"))
        self.comboBoxA1.setItemText(0, _translate("MainWindow", "relu"))
        self.comboBoxA1.setItemText(1, _translate("MainWindow", "tanh"))
        self.comboBoxA2.setItemText(0, _translate("MainWindow", "relu"))
        self.comboBoxA2.setItemText(1, _translate("MainWindow", "tanh"))
        self.labelA2.setText(_translate("MainWindow", "Aktivasi 2"))
        self.labelAkurasi1.setText(_translate("MainWindow", "Akurasi"))
        self.labelAkurasi2.setText(_translate("MainWindow", "%"))

        self.labelTrain.setText(_translate("MainWindow", "Optimizer"))
        self.comboBoxTrain.setItemText(0, _translate("MainWindow", "AdaGrad"))
        self.comboBoxTrain.setItemText(1, _translate("MainWindow", "Adam"))
        self.comboBoxTrain.setItemText(2, _translate("MainWindow", "RMSProp"))

        self.labelNeuron1.setText(_translate("MainWindow", "Neuron 1"))
        self.comboBoxNeuron1.setItemText(0, _translate("MainWindow", "5"))
        self.comboBoxNeuron1.setItemText(1, _translate("MainWindow", "10"))
        self.comboBoxNeuron1.setItemText(2, _translate("MainWindow", "15"))
        self.comboBoxNeuron1.setItemText(3, _translate("MainWindow", "20"))
        self.comboBoxNeuron1.setItemText(4, _translate("MainWindow", "25"))
        self.comboBoxNeuron1.setItemText(5, _translate("MainWindow", "30"))
        self.comboBoxNeuron1.setItemText(6, _translate("MainWindow", "35"))
        self.comboBoxNeuron1.setItemText(7, _translate("MainWindow", "40"))
        self.comboBoxNeuron1.setItemText(8, _translate("MainWindow", "45"))
        self.comboBoxNeuron1.setItemText(9, _translate("MainWindow", "50"))

        self.labelNeuron2.setText(_translate("MainWindow", "Neuron 2"))
        self.comboBoxNeuron2.setItemText(0, _translate("MainWindow", "5"))
        self.comboBoxNeuron2.setItemText(1, _translate("MainWindow", "10"))
        self.comboBoxNeuron2.setItemText(2, _translate("MainWindow", "15"))
        self.comboBoxNeuron2.setItemText(3, _translate("MainWindow", "20"))
        self.comboBoxNeuron2.setItemText(4, _translate("MainWindow", "25"))
        self.comboBoxNeuron2.setItemText(5, _translate("MainWindow", "30"))
        self.comboBoxNeuron2.setItemText(6, _translate("MainWindow", "35"))
        self.comboBoxNeuron2.setItemText(7, _translate("MainWindow", "40"))
        self.comboBoxNeuron2.setItemText(8, _translate("MainWindow", "45"))
        self.comboBoxNeuron2.setItemText(9, _translate("MainWindow", "50"))

        self.labelKlasifikasi.setText(_translate("MainWindow", "Klasifikasi"))
        self.comboBoxKlasifikasi.setItemText(0, _translate("MainWindow", "BB/U"))
        self.comboBoxKlasifikasi.setItemText(1, _translate("MainWindow", "TB/U"))
        self.comboBoxKlasifikasi.setItemText(2, _translate("MainWindow", "BB/TB"))

        self.buttonProsesPelatihan.setText(_translate("MainWindow", "Proses"))

        self.buttonResetPelatihan.setText(_translate("MainWindow", "Reset"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)

    MainWindow.show()
    sys.exit(app.exec_())
