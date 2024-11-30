import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
from tokenizers import BertWordPieceTokenizer
import matplotlib.pyplot as plt
import time

# Define parameters
MAX_LENGTH = 40
BATCH_SIZE = 64
EPOCHS = 10
VOCAB_SIZE = 8000
EMBEDDING_DIM = 512
NUM_HEADS = 8
FF_DIM = 2048
NUM_LAYERS = 4
LATENT_DIM = 100


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model=EMBEDDING_DIM)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)



tokenizer = BertWordPieceTokenizer("vocab.txt", lowercase=True)

def encode_sentence(sentence):
    tokens = tokenizer.encode(sentence)
    return tokens.ids

def decode_sentence(token_ids):
    return tokenizer.decode(token_ids)

def preprocess_data(data, max_length):
    tokenized_data = [encode_sentence(sentence) for sentence in data]
    padded_data = tf.keras.preprocessing.sequence.pad_sequences(tokenized_data, maxlen=max_length, padding='post')
    return padded_data

# Example sentences for demonstration
sentences = ["This is an example.", "Transformers are powerful.", "Natural language processing is fascinating."]
encoded_sentences = preprocess_data(sentences, max_length=MAX_LENGTH)



class PositionalEncoding(layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        pos_encoding = np.concatenate([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]
        
        
def scaled_dot_product_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights
    
    
    
class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads