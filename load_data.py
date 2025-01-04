import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from preprocessing import *
from hyperprameter import *

def text_to_seq(text):# we did this hear because it so much faster hear
    seq = []
    for s in text:
        if s != ' ':
            seq.append(s)
    seq.append("EOS")
    return ' '.join(seq).replace('  ', ' ')

dataset = pd.read_csv('metadata.csv', sep='|', header=None, names=['0', '1'])
dataset['0'], dataset['1'] = dataset['1'], dataset['0'] # just because we need a dataset's format like Ljspeech

# Preprocess the text to phonemes
dataset['1'] = dataset['1'].apply(text_to_seq)

train_ds1, valid_ds1 = train_test_split(dataset, test_size=0.08, random_state=hp.seed)

# Convert the dataframe to TensorFlow dataset
def df_to_dataset(dataframe):
    df = dataframe.copy()
    ds = tf.data.Dataset.from_tensor_slices((df['0'].values, df['1'].values))
    return ds

train_ds1 = df_to_dataset(train_ds1)
valid_ds1 = df_to_dataset(valid_ds1)

AUTOTUNE = tf.data.AUTOTUNE

# Prepare the datasets
train_ds1 = train_ds1.shuffle(buffer_size=10, seed=42).batch(hp.batch_size, drop_remainder=True)
valid_ds1 = valid_ds1.batch(hp.batch_size, drop_remainder=True)

# Function to load data
def load_data(audio_paths, texts):
    def load_wav_file(audio_path):
        audio_path = tf.strings.join(['/content/drive/MyDrive/wavs/', audio_path])
        audio_binary = tf.io.read_file(audio_path)
        audio, sample_rate= tf.audio.decode_wav(audio_binary,  desired_channels=1, desired_samples=-1)
        expected_sample_rate = 22050
        tf.debugging.assert_equal(sample_rate, expected_sample_rate, message="Sample rate does not match 22050 Hz")

        audio = tf.squeeze(audio, axis=-1)  # Remove channels dimension
        spectrogram = get_spectrogram(audio)
        return spectrogram

    spectrograms = tf.map_fn(load_wav_file, audio_paths, dtype=(tf.float32, tf.float32, tf.int32))
    input_spectrogram, output_spectrogram ,spectrograms_len= spectrograms
    input_spectrogram = tf.ensure_shape(input_spectrogram, (hp.batch_size, 1024, 129))
    output_spectrogram = tf.ensure_shape(output_spectrogram, (hp.batch_size, 1024, 129))
    target_spectrogram = get_stop_token(spectrograms_len, hp.max_mel_time)
    return (texts, input_spectrogram, spectrograms_len), (output_spectrogram,target_spectrogram)

train_ds = train_ds1.map(lambda x, y: load_data(x, y))
valid_ds = valid_ds1.map(lambda x, y: load_data(x, y))

text_vectorization = tf.keras.layers.TextVectorization(max_tokens=100, output_sequence_length=150)
text_vectorization.adapt(dataset['1'])