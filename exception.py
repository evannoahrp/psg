try:
  print(x)
except:
  print("An exception occurred")

  , fit_params = {'callbacks': [self.es]}

  
        #modelDT = Sequential()
        #modelDT.add(Dense(neuron1, input_dim = self.inputdim, activation = aktivasi1))
        #if(hLayer == 1):
        #    modelDT.add(Dense(neuron2, activation = aktivasi2))
        #modelDT.add(Dense(4, activation = 'softmax'))
        # Compile model
        #modelDT.compile(loss = 'binary_crossentropy', optimizer = train, metrics = ['accuracy'])
        #modelDT.fit(self.input, self.output, epochs = 2000, verbose = 1, callbacks = [self.es])
        #yNew = modelDT.predict_classes(xNew)


        
            #mengencode output dari klasifikasi BBU kedalam bentuk kategorial
            encoder.fit(outputBBU)
            encoded_BBU = encoder.transform(outputBBU)
            self.dummy_BBU = np_utils.to_categorical(encoded_BBU)
            #Baik   = 1000
            #Buruk  = 0100
            #Kurang = 0010
            #Lebih  = 0001
            #mengencode output dari klasifikasi TBU kedalam bentuk kategorial
            encoder.fit(outputTBU)
            encoded_TBU = encoder.transform(outputTBU)
            self.dummy_TBU = np_utils.to_categorical(encoded_TBU)
            #Normal         = 1000
            #Pendek         = 0100
            #Sangat Pendek  = 0010
            #Tinggi         = 0001
            #mengencode output dari klasifikasi BBTB kedalam bentuk kategorial
            encoder.fit(outputBBTB)
            encoded_BBTB = encoder.transform(outputBBTB)
            self.dummy_BBTB = np_utils.to_categorical(encoded_BBTB)
            #Gemuk          = 1000
            #Kurus          = 0100
            #Normal         = 0010
            #Sangat Kurus   = 0001
            #Split data masing masing klasifikasi menjadi training dan testing