import tensorflow.keras as keras
import tensorflow as tf
from sagemaker_tensorflow import PipeModeDataset


INPUT_TENSOR_NAME = 'masking_input'
SIGNATURE_NAME = 'predictions'
PREFETCH_SIZE = 10
BATCH_SIZE = 128
NUM_PARALLEL_BATCHES = 10
MAX_EPOCHS = 10
SHUFFLE_BUFFER_SIZE = 10000


def get_model(hyperparameters):
    
   # learning_rate = hyperparameters['learning_rate']
    n_neurons = int(hyperparameters['n_neurons'])
    n_hidden = int(hyperparameters['n_hidden'])
    
    model = keras.models.Sequential()
    model.add(keras.layers.Masking(mask_value=99, input_shape=(None, 3)))
    
    for layer in range(n_hidden):
        model.add(keras.layers.LSTM(n_neurons, return_sequences=True))
    model.add(keras.layers.LSTM(n_neurons))
    model.add(keras.layers.Dense(10, activation='softmax'))
    
  #  optimizer = keras.optimizers.Adam(lr=learning_rate)
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    
    return model


def _input_fn(channel):
    """Returns a Dataset for reading from a SageMaker PipeMode channel."""
    features={
            'label': tf.FixedLenFeature([], tf.int64),
            'feature': tf.FixedLenFeature([480], tf.int64)
        }

    def parse(record):
        parsed = tf.parse_single_example(record, features)
        
        data = tf.reshape(parsed['feature'], [160, 3])
        data = tf.cast(data, tf.float32)
        
        label = tf.cast(parsed['label'], tf.int32)
        
        return ({INPUT_TENSOR_NAME: data}, label)

    ds = PipeModeDataset(channel=channel, record_format='TFRecord')
    ds = ds.shuffle(SHUFFLE_BUFFER_SIZE)
    ds = ds.repeat(MAX_EPOCHS)
    ds = ds.prefetch(PREFETCH_SIZE)
    ds = ds.map(parse, num_parallel_calls=NUM_PARALLEL_BATCHES)
    ds = ds.batch(BATCH_SIZE)
    
    return ds

def train_input_fn(training_dir, params):
    """Returns input function that would feed the model during training"""
    return _input_fn('train')

def eval_input_fn(training_dir, params):
    """Returns input function that would feed the model during evaluation"""
    return _input_fn('eval')


