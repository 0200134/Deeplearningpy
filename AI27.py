class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model=512)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

from tokenizers import BertWordPieceTokenizer

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

# Example usage
sentences = ["This is an example.", "Transformers are powerful."]
encoded_sentences = preprocess_data(sentences, max_length=20)


import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

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
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, _ = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output

def point_wise_feed_forward_network(d_model, dff):
    return models.Sequential([
        layers.Dense(dff, activation='relu'),
        layers.Dense(d_model)
    ])

class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.dropout3 = layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)
        return out3, attn_weights_block1, attn_weights_block2

class Encoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(input_vocab_size, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding(x)
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x

class Decoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(target_vocab_size, d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask,[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/chenjianbo0705/more-learning/tree/8192da5c509537826da8d916d6a6afb4cf6caca3/Tensorflow2.0%20Transformer%E6%A8%A1%E5%9E%8B%E4%B8%AD%E8%8B%B1%E7%BF%BB%E8%AF%91%2FTransformer.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "1")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/TobiasNorlund/language-model/tree/e8b6d19e6cc56ca5d23b442b561ffae8c5d16742/main.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "2")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/lingyixia/sample_demos/tree/9de84a19ba4beb18dbe35213f1dcc64d813bc038/Transformer.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "3")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/NNDEV1/NMTWithTransformers/tree/9bb7529d9589de1ddda1434268b53e94b9d65379/model%2Fmodel_utils.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "4")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/eagle705/bert/tree/62267e48a7652f58742be22fffb88757f826e51f/model%2Fembedding%2Fpositional_encoding.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "5")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/hxstarklin/STSAN/tree/7c186ae4f1e7421eda5b37d7ae1d0f3bcb20e25c/models.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "6")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/nanqiaobei/Road_tfboy/tree/03aa4aa3026ca461705a871eb377b47ab54714b7/test01.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "7")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/SamuelLAN/Chain_Dream_Construction_of_COVID-19_Knowledge_Graph/tree/f880ca99297c4efc785a4241b9d06ceab426fac5/lib%2Ftf_models%2Ftransformer.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "8")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/xiaobai19950529/SAN_1/tree/96a8c87ef79556ad15f97b75d20998e1557aca0c/model.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "9")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/baichii/tf2/tree/521d70a5b6f538b15adff6341d73befb367193d4/transformer%2Fattention.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "10")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/phaphuang/triq/tree/22c34f96ac966b7530cc20a3e50c6f6def884074/models%2Ftransformer_iqa.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "11")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/laurent-twd/seq2spell/tree/fa31d1de8ffe88e7a928719c97b7babd521649ef/Transformer%2Fmodel.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "12")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/kawasaki-kento/Transformer/tree/98716f561f76bc0f018298709f79ad6f88a2aadb/model.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "13")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/YangHan-Morningstar/Transformer-PyTorch/tree/13b59d30ab68db54c8892edd18c0d1959e90e8b3/Transformer%2FModules.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "14")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/codesteller/time-series-with-transformer/tree/ff2ee3a3f181ed9571c69a8134d7bf23a59a2faf/transformers_multigpu_tf.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "15")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/shige0409/automatic-headline-generation/tree/e9759c62b6f1884c51e2c4b70ff8fc5a6bbb875a/web%2Fmodels%2Ftransformer.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "16")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/young31/Deep-Learning/tree/07410be0618b16a40f87b7a59072234d77482800/transformer%2Flayers.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "17")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/fastestimator/RUA/tree/6930d74bfff2417ccf13ee261ec377041bf3567c/rua_gridsearch%2Fvit_tiny_imagenet.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "18")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/vbvg2008/benchmarks/tree/5fddd333aa9cc5a5d1f305a6b192b8eb702a2307/transformer%2Ftransformer_tf.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "19")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/GyanendroKh/Transliteration/tree/ffef97c42fb2caa0241be6941ca12524734fc9d8/transformer%2Flayers.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "20")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/Karlydiamond214/readerbenchpy/tree/1a070ae678f58ccd6f358c0802bdf0b3b3dde9d3/rb%2Fprocessings%2Fgec%2Ftransformer%2Fdecoder_layer.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "21")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/Yegor9151/Introduction_to_Natural_Language_Processing/tree/259d2c8c3a136caef6bd02d9a5f98aade7932235/course_project%2Fmy_layers%2Fencoder_decoder.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "22")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/201528007010032/kaikeba_project2/tree/316f0fe79c6ee73b96543e28e0b46f2046426f69/model%2Ftransformer%2Flayers.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "23")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/StephenZou/QA_Summary/tree/b3bab1968f7460a4e5e419d20864afd1fa876488/transformer_pgn%2Fdecoders%2Fself_attention_decoder.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "24")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/zhaorui-tan/transformer_tf2/tree/3bee2b7ad7fadde3d98716a62ff8bb324faa6ee4/Transformer.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "25")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/SunYanCN/BAND/tree/afba041b239d1a3766fa8418ddf399bbbd718ef1/band%2Flayers.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "26")