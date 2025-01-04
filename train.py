from preprocessing import *
from hyperprameter import *
from model import *
from load_data import *
from tts_loss import *

optimizer = tf.keras.optimizers.AdamW(hp.lr)
mean_loss = tf.keras.metrics.Mean()
metrics = [tf.keras.metrics.MeanAbsoluteError()]

n_steps = len(dataset)*.92 // hp.batch_size


def print_status_bar(step, total, loss, metrics=None):
    metrics = " - ".join([f"{m.name}: {m.result():.4f}"
                          for m in [loss] + (metrics or [])])
    end = "" if step < total else "\n"
    print(f"\r{step}/{total} - " + metrics, end=end)

for i in train_ds.take(1):# this is just because of building the model
  (text , spectrogram, mel_len_x), (tgt_spectrogram, stop_tokens) = i
  mel_postnet, mel_linear, predicted_stop_token = model((text, spectrogram, mel_len_x))


  for epoch in range(1, hp.n_epochs + 1):
    tf.keras.backend.clear_session()
    print(f'Epoch {epoch}/{hp.n_epochs}')
    step = 0
    for (text , spectrogram, mel_len_x), (tgt_spectrogram, stop_tokens) in train_ds:
        step += 1
        with tf.GradientTape() as tape:
            mel_postnet, mel_linear, predicted_stop_token = model((text, spectrogram, mel_len_x))
            loss = tf.reduce_mean(tts_loss((tgt_spectrogram, stop_tokens), (mel_postnet, mel_linear, predicted_stop_token)))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        mean_loss(loss)
        for metric in metrics:
            metric(mel_postnet, spectrogram)
        print_status_bar(step, n_steps, mean_loss, metrics)
    for metric in [mean_loss] + metrics:
        metric.reset_state()

    for (val_text, val_spectrogram, mel_lens), (val_tgt_spectrogram, val_stop_tokens) in valid_ds:
        val_mel_postnet, val_mel_linear, val_stop_token = model((val_text, val_spectrogram, mel_lens), training=False)
        val_loss_value = tts_loss((val_tgt_spectrogram, val_stop_tokens), (val_mel_postnet, val_mel_linear, val_stop_token))
        mean_loss(val_loss_value)
        for metric in metrics:
            metric(val_tgt_spectrogram, val_mel_postnet)

    print(f'Validation Loss: {mean_loss.result()}')

    model.save_weights('./.weights.h5')
