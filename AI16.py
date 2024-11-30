import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, LayerNormalization, Dropout, Input
from tensorflow.keras.models import Model
import numpy as np

# Positional Encoding
def positional_encoding(position, d_model):
    angle_rads = np.arange(position)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

# Scaled Dot-Product Attention
def scaled_dot_product_attention(q, k, v, mask=None):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

# Multi-Head Attention Layer
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.dense = Dense(d_model)

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
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights
       
 # Encoder Layer
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

# Decoder Layer
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(dff, activation='relu'),
            Dense(d_model)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)
        return out3, attn_weights_block1, attn_weights_block2
        
# Encoder
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = Dropout(rate)

    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x

# Decoder
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            attention_weights[f'decoder_layer{i+1}_block1'] = block1
            attention_weights[f'decoder_layer{i+1}_block2'] = block2
        return x, attention_weights
        
# Transformer
class Transformer(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
        self.final_layer = Dense(target_vocab_size)

    def call(self, inp,[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/AmalLight/Sequence_Models/tree/8204904509a14146fca9ffe20db73a90b311825a/fourth%2FEmbedding_plus_Positional_encoding.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "1")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/laurent-twd/seq2spell/tree/fa31d1de8ffe88e7a928719c97b7babd521649ef/Transformer%2Fmodel.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "2")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/hxstarklin/STSAN/tree/7c186ae4f1e7421eda5b37d7ae1d0f3bcb20e25c/models.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "3")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/NNDEV1/NMTWithTransformers/tree/9bb7529d9589de1ddda1434268b53e94b9d65379/model%2Fmodel_utils.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "4")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/Abluceli/H2G-MAAC/tree/144542e098932d61b744bf337b3c6637b358a3b7/MIX_Graph_Multi_Agent%2FAttention%2Fattention.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "5")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/TobiasNorlund/language-model/tree/e8b6d19e6cc56ca5d23b442b561ffae8c5d16742/main.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "6")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/okason97/STONK/tree/22f009f29a070230b86602fde02e6dc44ffc76b9/src%2Fmodels%2Flstm_selfattention.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "7")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/noa/naacl2021/tree/8ca2fe6d49571d89965cebde237bc4751cc7c75a/aid%2Flayers.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "8")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/phaphuang/triq/tree/22c34f96ac966b7530cc20a3e50c6f6def884074/models%2Ftransformer_iqa.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "9")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/kawasaki-kento/Transformer/tree/98716f561f76bc0f018298709f79ad6f88a2aadb/model.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "10")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/YangHan-Morningstar/Transformer-PyTorch/tree/13b59d30ab68db54c8892edd18c0d1959e90e8b3/Transformer%2FModules.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "11")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/UgurDURA/Transformer_DL/tree/6e22e569df627c88de8814471e7993089c9c6515/PortToEnglish.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "12")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/pritam-dey3/e2e_nlg_tf/tree/83eb067d3d218eaf8a9ba9d439a83a9cef5d9d1f/e2e_transformers%2Flayers.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "13")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/mpafla/question-generation/tree/dd7299302eb7926677a064392c077f996372e1d4/models%2Flayers%2Fencoder_layer.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "14")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/YeongSeokJeong/Post_Editing-Seq2Seq/tree/238d25c45a1912fec2927a38b4570be7a710b636/Transformer%2Ftransformer.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "15")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/bachvarv/NLP_Test/tree/ce5e4bcb0bb545ff3552c29ffdc3a7b76d2e3c8a/transformer%2Fdata%2FEncoderLayer.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "16")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/tx781854760/text_classification/tree/5a6168d76bbdd3ee883939defd71b5985b952494/c13_transformer_nmt.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "17")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/Karlydiamond214/readerbenchpy/tree/1a070ae678f58ccd6f358c0802bdf0b3b3dde9d3/rb%2Fprocessings%2Fgec%2Ftransformer%2Fdecoder_layer.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "18")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/cha-no/transformer/tree/c67a1f74eb888c7d8b58f147b1569f703c6827bb/model.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "19")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/mpafla/question-generation/tree/dd7299302eb7926677a064392c077f996372e1d4/models%2Flayers%2Fdecoder_layer.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "20")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/kdsar/Video_Game_Music_NN/tree/85a22753f5a4e2dd37a47d64d939bba681693d98/network.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "21")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/YoungloLee/char-bert-lm/tree/89bf0277d7ead352c219801cc926974de1d08efd/bert.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "22")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/Eddie6382/Transformer/tree/cd6987ea98b7afcba217408b309581b367166cde/src%2Fmodel.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "23")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/alvinrindra/Tensorflow_2.x_Tutorial/tree/6c624c401df6bdcb308b74d35109986b78c67953/21-CN-EN-Translation-BERT%2Ftransformer.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "24")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/PacktMrunal/Advanced-NLP-with-TensorFlow-2/tree/b05215a281268802bb439970856120a70db20942/chapter-7-image-cap-multimodal-transformers%2Fvisual_transformer.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "25")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/oct301/peptide_project/tree/79835d64ba398e01b9c94fa3b604331846df44d8/model.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "26")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/saahiluppal/langtret/tree/542f39f1687b4e2426d5ec5189ee41e6ba83f388/transformer_utils.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "27")[43dcd9a7-70db-4a1f-b0ae-981daa162054](https://github.com/zhaojiawen-coding/transformer-classfication/tree/81708a12071b4190b1a55ab3532bf3a6a85ccce3/model%2Ftransformer_tf2%2Fmodel.py?citationMarker=43dcd9a7-70db-4a1f-b0ae-981daa162054 "28")