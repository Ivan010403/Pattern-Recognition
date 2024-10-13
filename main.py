import os
import string 
import time
import numpy as np
import pandas as pd
from keras.models import Sequential 
from keras.layers import Dense, LSTM, Embedding, RepeatVector, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint 
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model 
from keras import optimizers 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

pd.set_option('display.max_colwidth', 100)

#------test set---------
test_set = pd.read_parquet("test.parquet")
test_set = test_set.drop(['id', 'file'], axis=1)

test_set['la'] = [s.translate(str.maketrans('', '', string.punctuation)) for s in test_set['la']] 
test_set['en'] = [s.translate(str.maketrans('', '', string.punctuation)) for s in test_set['en']] 

#------train set--------
train_set = pd.read_parquet("test.parquet") #TODO: change that!!!
train_set = train_set.drop(['id', 'file'], axis=1)

train_set['la'] = [s.translate(str.maketrans('', '', string.punctuation)) for s in train_set['la']] 
train_set['en'] = [s.translate(str.maketrans('', '', string.punctuation)) for s in train_set['en']] 

#-------------
# Convert text to lowercase 
# for i in range(len(train_set)): 
#     deu_eng[i,0] = deu_eng[i,0].lower() 
#     deu_eng[i,1] = deu_eng[i,1].lower()


#TODO: разобраться с токенизированием строк
# Prepare La tokenizer
la_tokenizer = Tokenizer()
la_tokenizer.fit_on_texts(train_set['la'])
la_vocab_size = len(la_tokenizer.word_index) + 1 
la_length = 30

# Prepare En tokenizer 
en_tokenizer = Tokenizer()
en_tokenizer.fit_on_texts(train_set['en'])
en_vocab_size = len(en_tokenizer.word_index) + 1 
en_length = 30


#проблема с тем, что у нас не нормированные предложения
# Encode and pad sequences 
def encode_sequences(tokenizer, length, lines):          
    # integer encode sequences
    seq = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    seq = pad_sequences(seq, maxlen=length, padding='post')

    return seq


trainX = encode_sequences(la_tokenizer, la_length, train_set['la'])
trainY = encode_sequences(en_tokenizer, en_length, train_set['en'])

# Prepare validation data 
# testX = encode_sequences(la_tokenizer, la_length, test_set['la'])
# testY = encode_sequences(en_tokenizer, en_length, test_set['en'])

def make_model(in_vocab, out_vocab, in_timesteps, out_timesteps, n):
    model = Sequential()
    model.add(Embedding(in_vocab, n, input_length=in_timesteps, mask_zero=True))
    model.add(LSTM(n))
    model.add(Dropout(0.3))
    model.add(RepeatVector(out_timesteps))
    model.add(LSTM(n, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(Dense(out_vocab, activation='softmax'))
    model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='sparse_categorical_crossentropy')
    return model

print("la_vocab_size:", la_vocab_size, la_length)
print("en_vocab_size:", en_vocab_size, en_length)

# model = make_model(la_vocab_size, en_vocab_size, la_length, en_length, 512)

# num_epochs = 40
# history = model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1), epochs=num_epochs, batch_size=512, validation_split=0.2, callbacks=None, verbose=1)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.legend(['train','validation'])
# plt.show()
# model.save('la-en-model.h5')


model = load_model('la-en-model.h5')
def get_word(n, tokenizer):
    if n == 0:
        return ""
    for word, index in tokenizer.word_index.items():
        if index == n:
            return word
    return ""


print(train_set[:1])

phrs_enc = encode_sequences(la_tokenizer, la_length, ["Tibi autem qui sapis quam potest"])
print("phrs_enc:", phrs_enc)
print("-----------------")

predict_x=model.predict(phrs_enc)

classes_x = []

for i in range(0, len(predict_x[0])):
    classes_x.append(np.argmax(predict_x[0][i]))  # Get the index of the highest probability class

print(classes_x)

print(get_word(classes_x[0], en_tokenizer))