import tensorflow as tf
from hyperprameter import *

def TTSLoss():
    mse_loss = tf.keras.losses.MeanSquaredError()
    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    @tf.function
    def loss_fn(y_true, y_pred):
        mel_target, stop_token_target = y_true
        mel_out, mel_linear_out, stop_token_out = y_pred
        stop_token_target = tf.reshape(stop_token_target, [-1, 1])
        stop_token_out = tf.reshape(stop_token_out, [-1, 1])

        mel_loss = mse_loss(mel_target, mel_out) + mse_loss(mel_target, mel_linear_out)
        stop_token_loss = bce_loss(stop_token_target, stop_token_out) * hp.r_gate

        return mel_loss + stop_token_loss

    return loss_fn

tts_loss = TTSLoss()