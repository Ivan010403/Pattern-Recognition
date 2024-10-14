import collections
import string
import time
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential 
from keras.layers import Dense, LSTM, Embedding, RepeatVector, Dropout
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
# from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.models import load_model 
from keras.losses import sparse_categorical_crossentropy

def read_text(filename):
    with open(filename, mode='rt', encoding='utf-8') as file:
        text = file.read()
        sents = text.strip().split('\n')
        return [i.split('\t') for i in sents]

data = read_text("deutch.txt")
deu_eng = np.array(data)

deu_eng = deu_eng[:30000,:]
print("Dictionary size:", deu_eng.shape)

# Remove punctuation 
deu_eng[:,0] = [s.translate(str.maketrans('', '', string.punctuation)) for s in deu_eng[:,0]] 
deu_eng[:,1] = [s.translate(str.maketrans('', '', string.punctuation)) for s in deu_eng[:,1]] 

# convert text to lowercase 
for i in range(len(deu_eng)): 
    deu_eng[i,0] = deu_eng[i,0].lower() 
    deu_eng[i,1] = deu_eng[i,1].lower()
# print('{} la words.'.format(len([word for sentence in la_words_counter for word in sentence.split()])))
# print('{} unique la words.'.format(len(la_words_counter)))
# print('10 Most common words in the la dataset:')
# print('"' + '" "'.join(list(zip(*la_words_counter.most_common(10)))[0]) + '"')
# print()
# print('{} en words.'.format(len([word for sentence in en_words_counter for word in sentence.split()])))
# print('{} unique en words.'.format(len(en_words_counter)))
# print('10 Most common words in the en dataset:')
# print('"' + '" "'.join(list(zip(*en_words_counter.most_common(10)))[0]) + '"')


def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    # TODO: Implement
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    return tokenizer.texts_to_sequences(x), tokenizer


def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    if length is None:
        length = max([len(sentence) for sentence in x])
    return pad_sequences(x, maxlen=length, padding='post', truncating='post')

# Pad Tokenized output
# test_pad = pad(text_tokenized)
# for sample_i, (token_sent, pad_sent) in enumerate(zip(text_tokenized, test_pad)):
#     print('Sequence {} in x'.format(sample_i + 1))
#     print('  Input:  {}'.format(np.array(token_sent)))
#     print('  Output: {}'.format(pad_sent))


def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)
    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)
# Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)
    return preprocess_x, preprocess_y, x_tk, y_tk

# preproc_la_sentences, preproc_en_sentences, la_tokenizer, en_tokenizer = preprocess(deu_eng[:,0], deu_eng[:,1])
preproc_la_sentences, preproc_en_sentences, la_tokenizer, en_tokenizer = preprocess(["The buzzer sounded", "The clock is wrong", "The alarm went off"], deu_eng[:,1])

print(deu_eng[:1,0])
print(deu_eng[:1,1])


max_la_sequence_length = preproc_la_sentences.shape[1]
max_en_sequence_length = preproc_en_sentences.shape[1]
la_vocab_size = len(la_tokenizer.word_index)
en_vocab_size = len(en_tokenizer.word_index)
print('Data Preprocessed')
print("Max la sentence length:", max_la_sequence_length)
print("Max en sentence length:", max_en_sequence_length)
print("Latin vocabulary size:", la_vocab_size)
print("English vocabulary size:", en_vocab_size)

def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'
    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

def simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    """
    Build and train a basic RNN on x and y
    :param input_shape: Tuple of input shape
    :param output_sequence_length: Length of output sequence
    :param english_vocab_size: Number of unique English words in the dataset
    :param french_vocab_size: Number of unique French words in the dataset
    :return: Keras model built, but not trained
    """
    # TODO: Build the layers
    learning_rate = 1e-3
    model = Sequential()
    model.add(GRU(128, input_shape=input_shape[1:], return_sequences=True))
    model.add(Dropout(0.5))
    model.add(GRU(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(256, activation='relu')))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax'))) 
    
    
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model

# Reshaping the input to work with a basic RNN


tmp_x = pad(preproc_la_sentences, max_la_sequence_length)
print(tmp_x)
tmp_x = tmp_x.reshape((-1, preproc_en_sentences.shape[-2], 1))

print(tmp_x[:1])

# # Train the neural network
# simple_rnn_model = simple_model(
#     tmp_x.shape,
#     max_en_sequence_length,
#     la_vocab_size + 1,
#     en_vocab_size + 1)
# simple_rnn_model.fit(tmp_x, preproc_en_sentences, batch_size=300, epochs=10, validation_split=0.2)
# # Print prediction(s)

# simple_rnn_model.save('model.h5')

model = load_model('model.h5')

print(logits_to_text(model.predict(tmp_x[:1])[0], en_tokenizer))
