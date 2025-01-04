from hyperprameter import *
import numpy as np
from load_data import *
import matplotlib.pyplot as plt
from model import *

model.load_weights('.weights.h5')
def predict(text, model):
    phoneme_text1 = text_to_seq(text)
    phoneme_text = np.array([phoneme_text1])
    phoneme_text = tf.constant(phoneme_text, dtype = tf.string)
    mel_input = np.zeros((1, 1, hp.mel_freq))
    i = 0
    while True:

        predicted_mel_postnet, predicted_mel_linear, stop_token = model((phoneme_text, mel_input, tf.constant([len(mel_input[0])])))
        print('mel', mel_input.shape)
        mel_input = tf.concat([mel_input, predicted_mel_postnet[:, -1:, :]], axis=1)
        if stop_token is not None and tf.sigmoid(stop_token[0][-1]) < 0.5:
            break
        i+=1

    predicted_mel_spectrogram = predicted_mel_postnet[0]
    return predicted_mel_spectrogram

predicted_mel_spectrogram = predict("جهت یابی با درختان از ان جا که سمت ِ شمالی ِ درختان در معرض ِ افتاب ِ کمتری است ، درختان ِ در این سمتشان شاخ و برگ ِ کمتری دارند .", model)

# Optionally, visualize the predicted mel spectrogram
plt.imshow(tf.transpose(predicted_mel_spectrogram, perm=[1, 0]))
plt.colorbar()
plt.title("Predicted Mel Spectrogram")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.show()



def mel_scale_transform(stfts):
    num_spectrogram_bins = stfts.shape[-1]
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=hp.mel_freq,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=hp.sr,
        lower_edge_hertz=0.0,
        upper_edge_hertz=hp.sr / 2.0
    )
    mel_spectrograms = tf.tensordot(stfts, linear_to_mel_weight_matrix, 1)
    mel_spectrograms.set_shape(stfts.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    return mel_spectrograms


def norm_mel_spec_db(mel_spec):
    mel_spec = ((2.0 * mel_spec - hp.min_level_db) / (hp.max_db / hp.norm_db)) - 1.0
    mel_spec = tf.clip_by_value(mel_spec, -hp.ref * hp.norm_db, hp.ref * hp.norm_db)
    return mel_spec


def denorm_mel_spec_db(mel_spec):
    mel_spec = (((1.0 + mel_spec) * (hp.max_db / hp.norm_db)) + hp.min_level_db) / 2.0
    return mel_spec


def pow_to_db_mel_spec(mel_spec):
    mel_spec = tfio.audio.dbscale(
        mel_spec,
        multiplier=hp.ampl_multiplier,
        amin=hp.ampl_amin,
        top_db=hp.max_db
    )
    mel_spec = mel_spec / hp.scale_db
    return mel_spec


def db_to_amplitude(dB, power=1.0):
    # Convert dB to amplitude or power
    return tf.pow(10.0, dB * 0.05 * power)


def db_to_power_mel_spec(mel_spec):
    mel_spec = mel_spec * hp.scale_db
    mel_spec = db_to_amplitude(mel_spec, power=hp.ampl_power)
    return mel_spec


def griffin_lim(spectrogram, n_fft, n_iter=50):
    """Implement Griffin-Lim algorithm to invert a spectrogram to a waveform."""
    # Initialize random phase
    angles = tf.random.uniform(tf.shape(spectrogram), 0.0, 2.0 * np.pi)
    complex_spectrogram = tf.complex(spectrogram * tf.cos(angles), spectrogram * tf.sin(angles))

    for i in range(n_iter):
        # Inverse STFT to get the waveform
        waveform = tf.signal.inverse_stft(complex_spectrogram, frame_length=hp.win_length, frame_step=hp.hop_length,
                                          fft_length=hp.n_fft)

        # Recompute the STFT of the waveform
        est_spectrogram = tf.signal.stft(waveform, frame_length=hp.win_length, frame_step=hp.hop_length,
                                         fft_length=hp.n_fft - 1)

        # Update the phase information
        phase = tf.math.angle(est_spectrogram)

        complex_spectrogram = tf.complex(tf.abs(spectrogram), 0.0) * tf.complex(tf.cos(phase), tf.sin(phase))

    # Final waveform reconstruction
    waveform = tf.signal.inverse_stft(complex_spectrogram, frame_length=hp.win_length, frame_step=hp.hop_length,
                                      fft_length=hp.n_fft)

    return waveform


def mel_to_linear_transform(mel_spec, num_spectrogram_bins):
    # Compute the linear to mel weight matrix
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=hp.mel_freq,
        num_spectrogram_bins=num_spectrogram_bins - 1,
        sample_rate=hp.sr,
        lower_edge_hertz=0.0,
        upper_edge_hertz=hp.sr / 2.0
    )

    # Compute the pseudoinverse of the mel weight matrix
    mel_to_linear_matrix = tf.linalg.pinv(linear_to_mel_weight_matrix)

    # Convert mel spectrogram back to linear spectrogram using the pseudoinverse
    linear_spectrogram = tf.tensordot(mel_spec, mel_to_linear_matrix, 1)
    return linear_spectrogram


def convert_to_mel_spec(wav):
    stfts = tf.signal.stft(wav, frame_length=hp.win_length, frame_step=hp.hop_length, fft_length=hp.n_fft)
    spectrogram = tf.abs(stfts)
    mel_spec = mel_scale_transform(spectrogram)
    db_mel_spec = pow_to_db_mel_spec(mel_spec)
    db_mel_spec = tf.squeeze(db_mel_spec, 0)
    return db_mel_spec


def inverse_mel_spec_to_wav(mel_spec):
    power_mel_spec = db_to_power_mel_spec(mel_spec)
    linear_spectrogram = mel_to_linear_transform(power_mel_spec, hp.n_stft)

    pseudo_wav = griffin_lim(linear_spectrogram, hp.n_fft)
    return pseudo_wav


def save_waveform_as_wav(waveform, sample_rate, filename):
    # Reshape waveform to add channels dimension (needed for encode_wav)
    waveform = tf.expand_dims(waveform, axis=-1)
    wav_encoded = tf.audio.encode_wav(waveform, sample_rate)
    tf.io.write_file(filename, wav_encoded)
waveform = inverse_mel_spec_to_wav(predicted_mel_spectrogram)
save_waveform_as_wav(waveform , 22050, 'output.wav')

