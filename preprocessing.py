import tensorflow as tf
from hyperprameter import *
def compute_spectrogram(waveform, n_fft, win_length, hop_length, power):
    stft = tf.signal.stft(
        waveform,
        frame_length=win_length,
        frame_step=hop_length,
        fft_length=n_fft
    )

    # محاسبه spectrogram
    spectrogram = tf.abs(stft) ** power

    return spectrogram


def compute_mel_scale_transform(spectrogram, n_mels, sample_rate, n_stft):
    # ایجاد ماتریس Mel
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=n_stft,
        sample_rate=sample_rate,
        lower_edge_hertz=0.0,
        upper_edge_hertz=sample_rate / 2.0
    )

    # اعمال ماتریس Mel بر روی spectrogram
    mel_spectrogram = tf.tensordot(spectrogram, mel_weight_matrix, axes=1)

    # اضافه کردن بعد به Mel spectrogram (برای حفظ سازگاری شکل)
    mel_spectrogram.set_shape(spectrogram.shape[:-1] + (n_mels,))

    return mel_spectrogram


def pow_to_db_mel_spec(mel_spec, ampl_multiplier, amin, db_multiplier, top_db, scale_db):
    # تبدیل به مقیاس dB
    mel_spec_db = ampl_multiplier * tf.math.log(tf.maximum(amin, mel_spec)) / tf.math.log(10.0)

    # مقیاس dB را اعمال کنید
    mel_spec_db = db_multiplier * mel_spec_db

    # اعمال top_db (برش بالایی)
    if top_db is not None:
        max_spec = tf.reduce_max(mel_spec_db)
        mel_spec_db = tf.maximum(mel_spec_db, max_spec - top_db)

    # نرمال‌سازی مقادیر dB
    mel_spec_db = mel_spec_db / scale_db

    return mel_spec_db


def get_stop_token(sequence_lengths, max_length):
    batch_size = tf.shape(sequence_lengths)[0]
    ones = tf.ones([batch_size, max_length], dtype=tf.int32)
    range_tensor = tf.math.cumsum(ones, axis=1)
    mask = tf.greater_equal(sequence_lengths[:, tf.newaxis], range_tensor)
    return mask


def get_spectrogram(waveform):

    spec = compute_spectrogram(waveform, hp.n_fft, hp.win_length, hp.hop_length, hp.power)
    mel_spec = compute_mel_scale_transform(spec, 129, hp.sr, hp.n_stft)
    db_mel_spec = pow_to_db_mel_spec(mel_spec, hp.ampl_multiplier, 1e-6, hp.db_multiplier, 80, hp.scale_db)
    print(db_mel_spec)

    spectrogram = db_mel_spec
    
    spectrogram_len = tf.shape(spectrogram)[0]
    if spectrogram_len > 1024:
        spectrogram = spectrogram[:1024, :]
        SOS  = tf.zeros((1, 129))
        EOS = tf.zeros((1,129))
        input_spectrogram = tf.concat([SOS, spectrogram], axis=0)[:1024, :]
        spectrogram = tf.concat([spectrogram, EOS], axis=0)[:1024, :]
    else:
        #spectrogram = tf.pad(spectrogram, paddings=[[0, 0], [0, 1024 - spectrogram_len]],constant_values=0)
        padding = tf.zeros((1024 - spectrogram_len, tf.shape(spectrogram)[1]))
        SOS  = tf.zeros((1, 129))
        spectrogram = tf.concat([spectrogram, padding], axis=0)
        input_spectrogram = tf.concat([SOS, spectrogram], axis=0)[:1024, :]
    if spectrogram_len > 1024:
        spectrogram_len = 1024
    return input_spectrogram, spectrogram, spectrogram_len