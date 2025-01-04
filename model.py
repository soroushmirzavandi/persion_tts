import tensorflow as tf
import numpy as np
from hyperprameter import *
from tensorflow.keras import layers, models
from load_data import *

text_vectorization = load_data.text_vectorization
max_length = 100
vocab_size = text_vectorization.vocabulary_size
dropout_rate = .1
n_units = 128
embed_size = 256
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_length, embed_size, dtype=tf.float32, **kwargs):
        super().__init__(dtype=dtype, **kwargs)
        assert embed_size % 2 == 0, "embed_size must be even"
        p, i = np.meshgrid(np.arange(max_length),
                           2 * np.arange(embed_size // 2))
        pos_emb = np.empty((1, max_length, embed_size))
        pos_emb[0, :, ::2] = np.sin(p / 10_000 ** (i / embed_size)).T
        pos_emb[0, :, 1::2] = np.cos(p / 10_000 ** (i / embed_size)).T
        self.pos_encodings = tf.constant(pos_emb.astype(self.dtype))
        self.supports_masking = True

    def call(self, inputs):
        batch_max_length = tf.shape(inputs)[1]
        return inputs + self.pos_encodings[:, :batch_max_length]
class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.norm_1 = tf.keras.layers.LayerNormalization()
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=4, dropout=0.1, key_dim=hp.embedding_size)
        self.dropout_1 = tf.keras.layers.Dropout(0.1)

        self.norm_2 = tf.keras.layers.LayerNormalization()
        self.linear_1 = tf.keras.layers.Dense(hp.dim_feedforward)
        self.dropout_2 = tf.keras.layers.Dropout(0.1)

        self.linear_2 = tf.keras.layers.Dense(hp.embedding_size)
        self.dropout_3 = tf.keras.layers.Dropout(0.1)
        self.norm_3 = tf.keras.layers.LayerNormalization()

    def call(self, x, training=True, attn_mask=None):

        x_out = self.norm_1(x, training = training)
        x_out = self.attn(query=x_out,value=x_out,key=x_out,attention_mask=attn_mask)

        x_out = self.dropout_1(x_out, training=training)
        x = x + x_out

        x_out = self.norm_2(x, training = training)
        x_out = self.linear_1(x_out)
        x_out = tf.nn.relu(x_out)
        x_out = self.dropout_2(x_out, training=training)
        x_out = self.linear_2(x_out)
        x_out = self.dropout_3(x_out, training=training)
        x = x + x_out

        return x
class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.norm_1 = tf.keras.layers.LayerNormalization()
        self.self_attn = tf.keras.layers.MultiHeadAttention(num_heads=4,dropout=0.1, key_dim=hp.embedding_size)
        self.dropout_1 = tf.keras.layers.Dropout(0.1)

        self.norm_2 = tf.keras.layers.LayerNormalization()
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=4,dropout=0.1, key_dim=hp.embedding_size)
        self.dropout_2 = tf.keras.layers.Dropout(0.1)

        self.norm_3 = tf.keras.layers.LayerNormalization()
        self.linear_1 = tf.keras.layers.Dense(hp.dim_feedforward)
        self.dropout_3 = tf.keras.layers.Dropout(0.1)
        self.linear_2 = tf.keras.layers.Dense(hp.embedding_size)
        self.dropout_4 = tf.keras.layers.Dropout(0.1)

    def call(self, inputs, training=True):
        x = inputs['x']
        memory = inputs['memory']
        causul_mask = inputs.get('causul_mask', None)
        decoder_pad_mask = inputs.get('decoder_pad_mask', None)
        encoder_pad_mask = inputs.get('encoder_pad_mask', None)
        # Self-attention
        x_out = self.self_attn(
            query=x,
            value=x,
            key=x,
            attention_mask=causul_mask & decoder_pad_mask,
            training=training
        )

        x_out = self.dropout_1(x_out, training=training)
        x = self.norm_1(x + x_out)

        # Cross-attention
        x_out = self.attn(
            query=x,
            value=memory,
            key=memory,
            attention_mask=encoder_pad_mask,
            training=training
        )
        x_out = self.dropout_2(x_out, training=training)
        x = self.norm_2(x + x_out)

        # Feed-forward network
        x_out = self.linear_1(x)
        x_out = tf.nn.relu(x_out)
        x_out = self.dropout_3(x_out, training=training)
        x_out = self.linear_2(x_out)
        x_out = self.dropout_4(x_out, training=training)
        x = x + x_out
        x = self.norm_3(x)

        return x


class EncoderPreNet(tf.keras.Model):
    def __init__(self):
        super(EncoderPreNet, self).__init__()

        self.embedding = layers.Embedding(
            43,
            hp.encoder_embedding_size
        )

        self.linear_1 = layers.Dense(
            units=hp.encoder_embedding_size
        )

        self.linear_2 = layers.Dense(
            units=hp.embedding_size
        )

        self.conv_1 = layers.Conv1D(
            filters=hp.encoder_embedding_size,
            kernel_size=hp.encoder_kernel_size,
            strides=1,
            padding='same',
            dilation_rate = 2
        )
        self.bn_1 = layers.BatchNormalization()
        self.dropout_1 = layers.Dropout(0.5)

        self.conv_2 = layers.Conv1D(
            filters=hp.encoder_embedding_size,
            kernel_size=hp.encoder_kernel_size,
            strides=1,
            padding='same',
            dilation_rate = 4
        )
        self.bn_2 = layers.BatchNormalization()
        self.dropout_2 = layers.Dropout(0.5)

        self.conv_3 = layers.Conv1D(
            filters=hp.encoder_embedding_size,
            kernel_size=hp.encoder_kernel_size,
            strides=1,
            padding='same',
            dilation_rate = 8
        )
        self.bn_3 = layers.BatchNormalization()
        self.dropout_3 = layers.Dropout(0.5)

    def call(self, text, training=True, mask=None):

        x = self.embedding(text)  # (N, S, E)

        x = self.linear_1(x)

        x = self.conv_1(x)  # (N, S, E)
        x = self.bn_1(x)
        x = tf.nn.relu(x)
        x = self.dropout_1(x)

        x = self.conv_2(x)  # (N, S, E)
        x = self.bn_2(x)
        x = tf.nn.relu(x)
        x = self.dropout_2(x)

        x = self.conv_3(x)  # (N, S, E)
        x = self.bn_3(x)
        x = tf.nn.relu(x)
        x = self.dropout_3(x)

        x = self.linear_2(x)  # (N, S, E)

        x = PositionalEncoding(150, hp.embedding_size)(x)
        return x
    
class PostNet(tf.keras.Model):
    def __init__(self):
        super(PostNet, self).__init__()

        self.conv_1 = layers.Conv1D(
            filters=hp.postnet_embedding_size,
            kernel_size=hp.postnet_kernel_size,
            strides=1,
            padding='same',
            dilation_rate=1
        )
        self.bn_1 = layers.BatchNormalization()
        self.dropout_1 = layers.Dropout(0.5)

        self.conv_2 = layers.Conv1D(
            filters=hp.postnet_embedding_size,
            kernel_size=hp.postnet_kernel_size,
            strides=1,
            padding='same',
            dilation_rate=2
        )
        self.bn_2 = layers.BatchNormalization()
        self.dropout_2 = layers.Dropout(0.5)

        self.conv_3 = layers.Conv1D(
            filters=hp.postnet_embedding_size,
            kernel_size=hp.postnet_kernel_size,
            strides=1,
            padding='same',
            dilation_rate=4
        )
        self.bn_3 = layers.BatchNormalization()
        self.dropout_3 = layers.Dropout(0.5)

        self.conv_4 = layers.Conv1D(
            filters=hp.postnet_embedding_size,
            kernel_size=hp.postnet_kernel_size,
            strides=1,
            padding='same',
            dilation_rate=8
        )
        self.bn_4 = layers.BatchNormalization()
        self.dropout_4 = layers.Dropout(0.5)

        self.conv_5 = layers.Conv1D(
            filters=hp.postnet_embedding_size,
            kernel_size=hp.postnet_kernel_size,
            strides=1,
            padding='same',
            dilation_rate=16
        )
        self.bn_5 = layers.BatchNormalization()
        self.dropout_5 = layers.Dropout(0.5)

        self.conv_6 = layers.Conv1D(
            filters=hp.mel_freq,
            kernel_size=hp.postnet_kernel_size,
            strides=1,
            padding='same',
            dilation_rate=32
        )
        self.bn_6 = layers.BatchNormalization()
        self.dropout_6 = layers.Dropout(0.5)

    def call(self, x, training=True, mask=None):
        # x - (N, TIME, FREQ)

        x = self.conv_1(x)
        x = tf.tanh(x)
        x = self.bn_1(x)
        x = self.dropout_1(x)  # (N, TIME, POSTNET_DIM)

        x = self.conv_2(x)
        x = tf.tanh(x)
        x = self.bn_2(x)
        x = self.dropout_2(x)  # (N, TIME, POSTNET_DIM)

        x = self.conv_3(x)
        x = tf.tanh(x)
        x = self.bn_3(x)
        x = self.dropout_3(x)  # (N, TIME, POSTNET_DIM)

        x = self.conv_4(x)
        x = tf.tanh(x)
        x = self.bn_4(x)
        x = self.dropout_4(x)  # (N, TIME, POSTNET_DIM)

        x = self.conv_5(x)
        x = tf.tanh(x)
        x = self.bn_5(x)
        x = self.dropout_5(x)  # (N, TIME, POSTNET_DIM)

        x = self.conv_6(x)
        x = self.bn_6(x)
        x = self.dropout_6(x)  # (N, TIME, POSTNET_DIM)

        return x

class DecoderPreNet(tf.keras.Model):
    def __init__(self):
        super(DecoderPreNet, self).__init__()
        self.linear_1 = layers.Dense(
            units=hp.embedding_size
        )

        self.linear_2 = layers.Dense(
            units=hp.embedding_size
        )

        self.dropout = layers.Dropout(0.5)

    def call(self, x, training=True, mask=None):
        x = self.linear_1(x)
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        x = self.linear_2(x)
        x = tf.nn.relu(x)
        x = self.dropout(x, training=training)

        return x

class TransformerTTS(tf.keras.Model):
    def __init__(self):
        super(TransformerTTS, self).__init__()

        self.encoder_prenet = EncoderPreNet()
        self.decoder_prenet = DecoderPreNet()
        self.postnet = PostNet()

        self.pos_encoding = layers.Embedding(
            input_dim=hp.max_mel_time,
            output_dim=hp.embedding_size
        )

        self.encoder_block_1 = EncoderBlock()
        self.encoder_block_2 = EncoderBlock()
        self.encoder_block_3 = EncoderBlock()
        self.encoder_block_4 = EncoderBlock()

        self.decoder_block_1 = DecoderBlock()
        self.decoder_block_2 = DecoderBlock()
        self.decoder_block_3 = DecoderBlock()
        self.decoder_block_4 = DecoderBlock()

        self.linear_1 = layers.Dense(hp.mel_freq)
        self.linear_2 = layers.Dense(1)

        self.norm_memory = layers.LayerNormalization(epsilon=1e-6)

        self.pos_encoding = tf.keras.layers.Embedding(
            hp.max_mel_time,
            hp.embedding_size
            )

    def call(self, inputs, training=True, mask=None):
        text, mel , mel_len_x= inputs
        text = text_vectorization(text)

        TIME = tf.shape(mel)[1]
        encoder_pad_mask = tf.math.not_equal(text, 0)[:, tf.newaxis]
        decoder_pad_mask = tf.math.not_equal(tf.reduce_sum(mel, axis=-1), 0)[:, tf.newaxis, :]
        causul_mask = tf.linalg.band_part(tf.ones((TIME, TIME), tf.bool), -1, 0)[tf.newaxis]

        text_x = self.encoder_prenet(text)  # (N, S, E)

        text_x = self.encoder_block_1(
            text_x, training=True, attn_mask=encoder_pad_mask
        )
        text_x = self.encoder_block_2(
            text_x, training=True, attn_mask=encoder_pad_mask
        )
        text_x = self.encoder_block_3(
            text_x, training=True, attn_mask=encoder_pad_mask
        )  # (N, S, E)
        text_x = self.encoder_block_4(
            text_x, training=True, attn_mask=encoder_pad_mask
        )  # (N, S, E)
        text_x = self.norm_memory(text_x)

        mel_x = self.decoder_prenet(mel)  # (N, TIME, E)
        mel_len = mel_x[0].shape[0]
        pos_code = self.pos_encoding
        #mel_x = mel_x + pos_code(tf.range(mel_len))
        mel_x = PositionalEncoding(mel_len, 256)(mel_x)




        decoder_in = {'x':mel_x, 'memory':text_x, 'encoder_pad_mask':encoder_pad_mask,
                      'decoder_pad_mask':decoder_pad_mask, 'causul_mask':causul_mask}
        mel_x = self.decoder_block_1(
            decoder_in
        )

        decoder_in = {'x':mel_x, 'memory':text_x,'encoder_pad_mask':encoder_pad_mask,
                      'decoder_pad_mask':decoder_pad_mask, 'causul_mask':causul_mask}
        mel_x = self.decoder_block_2(
            decoder_in
        )
        decoder_in = {'x':mel_x, 'memory':text_x,'encoder_pad_mask':encoder_pad_mask,
                      'decoder_pad_mask':decoder_pad_mask, 'causul_mask':causul_mask}
        mel_x = self.decoder_block_3(
            decoder_in
        )  # (N, TIME, E)
        decoder_in = {'x':mel_x, 'memory':text_x,'encoder_pad_mask':encoder_pad_mask,
                      'decoder_pad_mask':decoder_pad_mask, 'causul_mask':causul_mask}
        mel_x = self.decoder_block_4(
            decoder_in
        )  # (N, TIME, E)

        mel_linear = self.linear_1(mel_x)  # (N, TIME, FREQ)

        mel_postnet = self.postnet(mel_linear, training = True)  # (N, TIME, FREQ)

        mel_postnet += mel_linear  # (N, TIME, FREQ)





        stop_token = self.linear_2(mel_x)  # (N, TIME, 1)



        bool_mel_mask = tf.squeeze(causul_mask, 0)[0]
        bool_mel_mask = tf.cast(bool_mel_mask, tf.int32)
        bool_mel_mask = tf.expand_dims(bool_mel_mask, -1)
        stop_token = tf.where(bool_mel_mask == 0, stop_token, 1e3)
        stop_token = tf.squeeze(stop_token, -1)

        return mel_postnet, mel_linear, stop_token
    
    
model = TransformerTTS()