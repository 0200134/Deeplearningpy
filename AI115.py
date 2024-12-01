import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, MultiHeadAttention, LayerNormalization, Dropout
from tensorflow.keras.models import Model

# Positional Encoding
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(sequence_length, d_model)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'sequence_length': self.sequence_length,
            'd_model': self.d_model,
        })
        return config

    def positional_encoding(self, sequence_length, d_model):
        angles = self.get_angles(np.arange(sequence_length)[:, np.newaxis],
                                 np.arange(d_model)[np.newaxis, :], d_model)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        pos_encoding = angles[np.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

# Transformer Encoder Layer
def transformer_encoder_layer(units, d_model, num_heads, dropout):
    inputs = Input(shape=(None, d_model))
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
    attention = Dropout(dropout)(attention)
    attention = LayerNormalization(epsilon=1e-6)(attention + inputs)

    outputs = Dense(units=units, activation='relu')(attention)
    outputs = Dropout(dropout)(outputs)
    outputs = Dense(units=d_model)(outputs)
    outputs = LayerNormalization(epsilon=1e-6)(outputs + attention)
    return Model(inputs=inputs, outputs=outputs)

# Transformer Encoder
def transformer_encoder(sequence_length, vocab_size, d_model, num_heads, num_layers, units, dropout):
    inputs = Input(shape=(None,))
    embedding = Embedding(vocab_size, d_model)(inputs)
    embedding *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embedding = PositionalEncoding(sequence_length, d_model)(embedding)

    x = embedding
    for _ in range(num_layers):
        x = transformer_encoder_layer(units, d_model, num_heads, dropout)(x)

    outputs = Dense(units=vocab_size)(x)
    return Model(inputs=inputs, outputs=outputs)

# Hyperparameters
vocab_size = 10000  # Vocabulary size
sequence_length = 50  # Sequence length
d_model = 128  # Dimension of model
num_heads = 8  # Number of attention heads
num_layers = 4  # Number of encoder layers
units = 512  # Number of units in feed forward network
dropout = 0.1  # Dropout rate

# Build and compile the Transformer model
transformer = transformer_encoder(sequence_length, vocab_size, d_model, num_heads, num_layers, units, dropout)
transformer.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
transformer.summary()
