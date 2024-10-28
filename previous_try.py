import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import pathlib
import random
import string
import re, time
import numpy as np
import tensorflow as tf
import pandas as pd
import keras_nlp as nlp
import keras
from keras import layers
from keras import ops

#----------------------------------downloading data------------------------------------------
# downloading dataset from standart datasets of keras
text_file = keras.utils.get_file(
    fname="spa-eng.zip",
    origin="http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip",
    extract=True,
)
text_file = pathlib.Path(text_file).parent / "spa-eng" / "spa.txt"

with open(text_file, encoding='utf-8') as f:
    lines = f.read().split("\n")[:-1]
text_pairs = []

# separate eng and spa sentences from each other
for line in lines:
    eng, spa = line.split("\t")
    spa = "[start] " + spa + " [end]"
    text_pairs.append((eng, spa))

# just check our data (5 random rows)
for _ in range(5):
    print(random.choice(text_pairs))

# random shuffling of data and making 3 sets for training, validation and testing
random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples :]

print(f"{len(text_pairs)} total pairs")
print(f"{len(train_pairs)} training pairs")
print(f"{len(val_pairs)} validation pairs")
print(f"{len(test_pairs)} test pairs")

# 99343 total pairs
# 69541 training pairs
# 14901 validation pairs
# 14901 test pairs

#------------------------------bringing data to the desired form-----------------------------
strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

vocab_size = 15000
sequence_length = 20
batch_size = 64

# bringing data to lowercase and delete all punctuation
# TODO: add to report the possibility of adding layer for punctuation
def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string) 
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")

# creation of two layer for vectorization data
eng_vectorization = keras.layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)
spa_vectorization = keras.layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1, #because in epochs we will have [0, N) <- not N itself
    standardize=custom_standardization,
)

# distinguish eng from spa and vectorize that. creation of vocabulary
train_eng_texts = [pair[0] for pair in train_pairs]
train_spa_texts = [pair[1] for pair in train_pairs]
eng_vectorization.adapt(train_eng_texts)
spa_vectorization.adapt(train_spa_texts)

# formating of our dataset
def format_dataset(eng, spa):
    eng = eng_vectorization(eng)
    spa = spa_vectorization(spa)
    return (
        {
            "encoder_inputs": eng,
            "decoder_inputs": spa[:, :-1],
        },
        spa[:, 1:],
    )

def spa_vectorisation(name):
    name = name.split()
    for i in range (0, len(name)):
        rand = random.randint(0,100)
        if rand <= 37:
            name[i] = ""
    
    name = " ".join(name)
    return nlp.func_name(name, 4)

# creation and vectorizing of datasets
def make_dataset(pairs):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)

    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    #(<tf.Tensor: shape=(), dtype=string, numpy=b"I'll call back at four o'clock.">, <tf.Tensor: shape=(), dtype=string, numpy=b'[start] Llamar\xc3\xa9 de nuevo a las cuatro. [end]'>)
    dataset = dataset.batch(batch_size)
    # batch_size = 64, that;s why i will get array of (64, 2)
    dataset = dataset.map(format_dataset)
    # shape (64, 20) and data -> nuumbers metamorphosis
    return dataset.cache().shuffle(2048).prefetch(16)

train_ds = make_dataset(train_pairs)
# for element in train_ds:  
#     print(element)
#     time.sleep(10)

val_ds = make_dataset(val_pairs)


# checking the dimension of our datasets
for inputs, targets in train_ds.take(1):
    print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
    print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
    print(f"targets.shape: {targets.shape}")



#------------------------creation of 3 user layers--------------------------------------
# trensformer emcoder layer
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = keras.ops.cast(mask[:, None, :], dtype="int32")
        else:
            padding_mask = None

        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "dense_dim": self.dense_dim,
                "num_heads": self.num_heads,
            }
        )
        return config

# PositionalEmbedding layer учёт порядка слов!
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = keras.ops.shape(inputs)[-1]
        positions = keras.ops.arange(0, length, 1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        else:
            return keras.ops.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config

# PositionaDecoder layer
class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(latent_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = keras.ops.cast(mask[:, None, :], dtype="int32")
            padding_mask = keras.ops.minimum(padding_mask, causal_mask)
        else:
            padding_mask = None

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = keras.ops.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = keras.ops.arange(sequence_length)[:, None]
        j = keras.ops.arange(sequence_length)
        mask = tf.keras.ops.cast(i >= j, dtype="int32")
        mask = keras.ops.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = keras.ops.concatenate(
            [keras.ops.expand_dims(batch_size, -1), keras.ops.convert_to_tensor([1, 1])],
            axis=0,
        )
        return keras.ops.tile(mask, mult)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "latent_dim": self.latent_dim,
                "num_heads": self.num_heads,
            }
        )
        return config
    

#---------------------------creation of model---------------------------------------------
embed_dim = 256
latent_dim = 2048
num_heads = 8 # attention heads!


# (encoder inputs [eng 0], decoder inputs [spa 0]) (target [spa +1])
encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)
encoder = keras.Model(encoder_inputs, encoder_outputs)

decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, encoded_seq_inputs)
x = layers.Dropout(0.5)(x)
decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

decoder_outputs = decoder([decoder_inputs, encoder_outputs])
transformer = keras.Model(
    [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
)

epochs = 45  # This should be at least 30 for convergence

# transformer.summary()
# transformer.compile(
#     "rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
# )
# transformer.fit(train_ds, epochs=epochs, validation_data=val_ds)
# transformer.save('model_new_v2.h5')

transformer = keras.models.load_model('model_new_v2.h5', custom_objects = {'PositionalEmbedding': PositionalEmbedding, 'TransformerEncoder': TransformerEncoder, 'TransformerDecoder': TransformerDecoder})

spa_vocab = spa_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 20

def decode_sequence(input_sentence):
    tokenized_input_sentence = eng_vectorization([input_sentence])
    dec_sentence = spa_vectorisation(input_sentence)
    dec_sentences = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = spa_vectorization([dec_sentences])[:, :-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])

        # print("predictions = ", len(predictions))
        # print("pred[0] = ", len(predictions[0]))
        # print("pred[0][0] = ", len(predictions[0][0]))
        # print(len(spa_index_lookup))
        

        sampled_token_index = ops.convert_to_numpy(
            ops.argmax(predictions[0, i, :])
        ).item(0)
        sampled_token = spa_index_lookup[sampled_token_index]
        dec_sentences += " " + sampled_token

        if sampled_token == "[end]":
            break
    return dec_sentence






test_eng_texts = ['English texts for beginners to practice reading and comprehension online and for free.', 'The neighbor’s cat even walks on the street in winter.']
for i in range(len(test_eng_texts)):
    input_sentence = test_eng_texts[i]
    translated = decode_sequence(input_sentence)

    print(input_sentence)
    print(translated)
    print()