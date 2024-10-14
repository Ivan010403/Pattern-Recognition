import collections
import string
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional, Embedding, LSTM
# from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
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

for sample_i in range(2):
    print('small_vocab_en Line {}:  {}'.format(sample_i + 1, deu_eng[sample_i,0]))
    print('small_vocab_fr Line {}:  {}'.format(sample_i + 1, deu_eng[sample_i,1]))

english_words_counter = collections.Counter([word for sentence in deu_eng[:,0] for word in sentence.split()])
deutsh_words_counter = collections.Counter([word for sentence in deu_eng[:,1] for word in sentence.split()])
print('{} English words.'.format(len([word for sentence in deu_eng[:,0] for word in sentence.split()])))
print('{} unique English words.'.format(len(english_words_counter)))
print('10 Most common words in the English dataset:')
print('"' + '" "'.join(list(zip(*english_words_counter.most_common(10)))[0]) + '"')
print()
print('{} Deutsh words.'.format(len([word for sentence in deu_eng[:,1] for word in sentence.split()])))
print('{} unique Deutsh words.'.format(len(deutsh_words_counter)))
print('10 Most common words in the Deutsh dataset:')
print('"' + '" "'.join(list(zip(*deutsh_words_counter.most_common(10)))[0]) + '"')

def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    # TODO: Implement
    x_tk = Tokenizer()
    x_tk.fit_on_texts(x)
    return x_tk.texts_to_sequences(x), x_tk

def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    # TODO: Implement
    if length is None:
        length = max([len(sentence) for sentence in x])
    return pad_sequences(x, maxlen=length, padding='post', truncating='post')

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

preproc_english_sentences, preproc_deutsh_sentences, english_tokenizer, deutsh_tokenizer = preprocess(deu_eng[:,0], deu_eng[:,1])
    
max_english_sequence_length = preproc_english_sentences.shape[1]
max_deutsh_sequence_length = preproc_deutsh_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
deutsh_vocab_size = len(deutsh_tokenizer.word_index)


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

def model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    learning_rate = 0.001
    inputs = Input(shape=input_shape[1:])
    emb = Embedding(english_vocab_size, 100)(inputs)
    gru = Bidirectional(GRU(128, dropout=0.5))(emb)
    final_enc = Dense(256, activation='relu')(gru)
    
    dec1 = RepeatVector(output_sequence_length)(final_enc)
    decgru = Bidirectional(LSTM(512, dropout=0.2, return_sequences=True))(dec1)
    layer = TimeDistributed(Dense(french_vocab_size, activation='softmax'))
    final = layer(decgru)
    
    
    model = Model(inputs=inputs, outputs=final)
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    return model
    
 
model = model_final(preproc_english_sentences.shape, preproc_deutsh_sentences.shape[1],len(english_tokenizer.word_index)+1, len(deutsh_tokenizer.word_index)+1)
model.fit(preproc_english_sentences, preproc_deutsh_sentences, batch_size=300, epochs=30, validation_split=0.2)

model.save('model.h5')

y_id_to_word = {value: key for key, value in deutsh_tokenizer.word_index.items()}

y_id_to_word[0] = '<PAD>'
sentence = 'how old are you'
sentence = [english_tokenizer.word_index[word] for word in sentence.split()]

sentence = pad_sequences([sentence], maxlen=preproc_english_sentences.shape[-1], padding='post')
sentences = np.array([sentence[0], preproc_english_sentences[0]])
predictions = model.predict(sentences, len(sentences))

print('Sample 1:')
print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))
print('wie alt sind sie')
print('Sample 2:')
print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[1]]))
print(' '.join([y_id_to_word[np.max(x)] for x in preproc_deutsh_sentences[0]]))