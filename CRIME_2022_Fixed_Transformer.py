
# %%
from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib
from scipy import signal
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import numpy.ma as ma
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler, StandardScaler, Normalizer
from sklearn.metrics import mean_squared_error
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
# !pip install cupy
# import cupy
import pylab as pl
import seaborn as sns
from pathlib import Path
import shutil
from tqdm.notebook import tqdm_notebook
# !pip install googlemaps
# import googlemaps
import sys
import math

# %%
# %%
import tensorflow.keras.backend as K
# !pip install tensorflow-addons 
import tensorflow_addons as tfa
import tensorflow_probability as tfp
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.optimizers import RMSprop,Adam,SGD,Adadelta,Adagrad,Adamax
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.compat.v1.keras.backend import set_session
from tensorflow.keras.utils import plot_model, to_categorical, normalize
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()
np.set_printoptions(threshold=sys.maxsize)

# %%
# tf.debugging.experimental.enable_dump_debug_info('.', tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
# tf.debugging.set_log_device_placement(True)
from tensorflow.python.client import device_lib
try:
  device_name = tf.test.gpu_device_name()
  if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
  print('Found GPU at: {}'.format(device_name))

  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = True
  config.gpu_options.per_process_gpu_memory_fraction = 0.1
  sess = tf.compat.v1.InteractiveSession(config=config)
  set_session(sess)
  print(device_lib.list_local_devices())
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  print("Num GPUs Available: ", len(gpus))

except Exception as error:
  print("error trying to configure computing device")
  print(error)

# %% [markdown]
# This was developed using Python 3.6 (Anaconda) and package versions:

# %%
from tensorflow.python.platform import build_info as tf_build_info
print("Tensorflow verison: ",tf.__version__)
print("CUDA verison: ", tf_build_info.build_info['cuda_version'])
print("CUDNN verison: ", tf_build_info.build_info['cudnn_version'])

# %%
# tf.keras.backend.floatx()
# tf.keras.backend.set_floatx('float16')
# tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
tf.keras.backend.floatx()


# %% [markdown]
# ### Load  Data
# 
# 

# %%
file=r'crimeDataFixed.csv'

myData = pd.read_csv(file, delimiter=',', index_col=0) 
# myData.round(decimals=6)
# myData=myData.astype(np.float32)
# myData=myData.astype(np.float16)
myData.describe()

# %%
# myData.convert_dtypes('float16')
myData.dtypes, myData.shape


# %% [markdown]
# List of the variables used in the data-set.

# %% [markdown]
# These are the top rows of the data-set.

# %%
data_top = myData.columns.values
data_top

# %%
myData.head(20)

# %%
myData.values.astype(np.float32)
myData.shape



# %%
input_names = myData.columns[:-18]
target_names = myData.columns[-18:]

# %%
myData_corr = myData.corr()[target_names][:-18]
myData_corr.where(np.tril(np.ones(myData_corr.shape)).astype(np.bool_)).style.background_gradient(cmap='coolwarm').set_properties(**{'font-size': '5'})
myData_corr.style.background_gradient(cmap='coolwarm').set_properties(**{'font-size': '5'})
# 'RdBu_r', 'BrBG_r', & PuOr_r are other good diverging colormaps
# newFilled_corr.describe()

# %% [markdown]
# ### Missing Data
# 
#  
# 
# Because we are using resampled data, we have filled in the missing values with new values that are linearly interpolated from the neighbouring values, which appears as long straight lines in these plots.
# 
# This may confuse the neural network. For simplicity, we will simply remove these two signals from the data.
# 
# But it is only short periods of data that are missing, so you could actually generate this data by creating a predictive model that generates the missing data from all the other input signals. Then you could add these generated values back into the data-set to fill the gaps.

# %%
# plt.figure(figsize=(30,5*16))
# myData.plot(subplots=True, figsize=(30,5*16))
# plt.grid(color='b', linestyle='-.', linewidth=0.5)
# plt.show()

# %% [markdown]
# ### Target Data for Prediction
# 
# We will try and predict the future Forex-data.

# %%
input_names = myData.columns[:-18]
target_names = myData.columns[-18:]
# input_names = myData.columns[:]
# target_names = myData.columns[:]

# %% [markdown]
# We will try and predict these signals.

# %%
df_targets = myData[target_names]
df_targets

# %% [markdown]
# ### NumPy Arrays
# 
# We now convert the Pandas data-frames to NumPy arrays that can be input to the neural network. We also remove the last part of the numpy arrays, because the target-data has `NaN` for the shifted period, and we only want to have valid data and we need the same array-shapes for the input- and output-data.
# 
# These are the input-signals:

# %%
x_data = myData[input_names].values.astype(np.float32)
x_data, x_data.dtype, np.isinf(x_data).any(), np.isnan(x_data).any()

# %%
print(type(x_data))
print("Shape:", x_data.shape)

# %% [markdown]
# These are the output-signals (or target-signals):

# %%
y_data = df_targets.values.astype(np.float32, casting='unsafe')
y_data, y_data.dtype, np.isinf(y_data).any(), np.isnan(y_data).any()

# %%
print(type(y_data))
print("Shape:", y_data.shape)

# %% [markdown]
# This is the number of observations (aka. data-points or samples) in the data-set:

# %%
num_data = len(x_data)
num_data

# %% [markdown]
# This is the fraction of the data-set that will be used for the training-set:

# %%
batch_size = 10 
train_split = 0.8
num_train = int(train_split * num_data)
num_val = int(0.5*(num_data - num_train))
num_test = (num_data - num_train) - num_val
#steps_per_epoch = int((num_train/batch_size)/40)
steps_per_epoch=1
#train_validation_steps = int((num_val/batch_size))
train_validation_steps = 1
#test_validation_steps = int((num_test/batch_size))
test_validation_steps = 1
print('num_train:',num_train, 'num_val:',num_val, 'num_test:',num_test)
print('steps_per_epoch:', steps_per_epoch)
print('train_validation_steps:', train_validation_steps, 'test_validation_steps:', test_validation_steps)

# %%
x_scaler = Normalizer().fit(myData[input_names].values.astype(np.float32))
y_scaler = Normalizer().fit(myData[target_names].values.astype(np.float32))

x_data_scaled = x_scaler.transform(x_data) + 1e-5
y_data_scaled = y_scaler.transform(y_data) + 1e-5

x_train, x_test, y_train, y_test = train_test_split(x_data_scaled, y_data_scaled, train_size=train_split, random_state=None, shuffle=True)
# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=train_split, random_state=None, shuffle=True)


# %% [markdown]
# This is the number of observations in the training-set:

# %%
num_train = len(x_train)
num_train

# %% [markdown]
# This is the number of observations in the test-set:

# %%
num_test = len(x_test)
num_test

# %% [markdown]
# These are the input-signals for the training- and test-sets:

# %%
len(x_train) + len(x_test)

# %% [markdown]
# These are the output-signals for the training- and test-sets:

# %%
len(y_train) + len(y_test)

# %% [markdown]
# This is the number of input-signals:

# %%
num_x_signals = x_data.shape[1]
num_x_signals

# %% [markdown]
# This is the number of output-signals:

# %%
num_y_signals = y_data.shape[1]
num_y_signals

# %% [markdown]
# ### Scaled Data
# 
# The data-set contains a wide range of values:

# %%
print('x_train min:', x_train.min())
print('x_train max:', x_train.max())

print('y_train min:', y_train.min())
print('y_train max:', y_train.max())

print('x_test min:', x_test.min())
print('x_test max:', x_test.max())

print('y_test min:', y_test.min())
print('y_test max:', y_test.max())

# %% [markdown]
# ## Data Generator
# 
# The data-set has now been prepared as 2-dimensional numpy arrays. The training-data has almost 300k observations, consisting of 20 input-signals and 3 output-signals.
# 
# These are the array-shapes of the input and output data:

# %%
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

# %%

batch_size = 1
sequence_length = 1
mask_percentage =0.05


class CustomDataGen(tf.keras.utils.Sequence):
        
    def __init__ (self, x_data, y_data, batch_size=None, sequence_length=None, train=True, validation=True, mask_percentage=0.01, random_batch=False, random_idx=False):
        
        self.x_train = x_data[0]
        self.x_test = x_data[1]        
        self.y_train = y_data[0]
        self.y_test = y_data[1]
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.train = train
        self.validation = validation
        self.random_batch = random_batch
        self.random_idx = random_idx
        self.mask_percentage = mask_percentage
        self.n = int(self.x_train.shape[0])
    
    def on_epoch_end(self):
        #do nothing
        return

    def __getitem__(self, index):
        if self.train:
            # print('using train samples')
            x_samples = self.x_train 
            y_samples = self.y_train 
            self.n = x_samples.shape[0]

        elif self.validation:
            # print('using validation samples')
            x_samples = self.x_test[:num_val]
            y_samples = self.y_test[:num_val]
            self.n = x_samples.shape[0]
        else:
            # print('using test samples')
            x_samples = self.x_test[-num_test:]
            y_samples = self.y_test[-num_test:]
            self.n = x_samples.shape[0]

            # Allocate a new array for the batch of input-signals.
        if self.train:
            # sequence_length_ = np.random.randint(1,self.sequence_length)
            sequence_length_ = self.sequence_length
        else:
            sequence_length_ = self.sequence_length
        
        if self.random_batch:
            batch_size_ = np.random.randint(1,self.batch_size)
        else: 
            batch_size_ = batch_size
        
        x_shape = (batch_size_, x_samples.shape[1])
        y_shape = (batch_size_, y_samples.shape[1])
        x_batch = np.zeros(shape=x_shape, dtype=np.float32)  
        y_batch = np.zeros(shape=y_shape, dtype=np.float32)        
            
        # Fill the batch with random sequences of data.
        for i in range(batch_size_):
            # Get a random start-index.

            if self.random_idx:
                sample_idx = np.random.randint(1, x_samples.shape[-2])

            # This points somewhere into the training-data.
            x_batch[i] = x_samples[sample_idx]
            y_batch[i] = y_samples[sample_idx]

        # return np.ma.expand_dims(x_batch, axis=0), np.ma.expand_dims(y_batch, axis=0)
        return x_batch, y_batch   
    
    def __len__(self):
        return int(self.n / self.batch_size)


x_train_generator = CustomDataGen((x_train, x_test), (y_train, y_test), batch_size=batch_size, sequence_length=sequence_length, train=True, validation=False, mask_percentage=mask_percentage, random_batch=False, random_idx=True)
x_train_batch, y_train_batch=x_train_generator.__getitem__(1)

print('x_train shape: ', x_train_batch.shape, 'x_train dtype:', x_train_batch.dtype)  
print('y_train shape: ', y_train_batch.shape, 'y_train dtype:', y_train_batch.dtype)

x_val_generator = CustomDataGen((x_train, x_test), (y_train, y_test), batch_size=batch_size, sequence_length=sequence_length, train=False, validation=True, mask_percentage=mask_percentage, random_batch=False, random_idx=True)
x_val_batch, y_val_batch=x_val_generator.__getitem__(1)

print('x_val shape: ', x_val_batch.shape, 'x_val dtype:', x_val_batch.dtype)  
print('y_val shape: ', y_val_batch.shape, 'y_val dtype:', y_val_batch.dtype)

x_test_generator = CustomDataGen((x_train, x_test), (y_train, y_test), batch_size=batch_size, sequence_length=sequence_length, train=False, validation=False, mask_percentage=mask_percentage, random_batch=False, random_idx=True)
x_test_batch, y_test_batch=x_test_generator.__getitem__(1)

print('x_test shape: ', x_test_batch.shape, 'x_test dtype:', x_test_batch.dtype)  
print('y_test shape: ', y_test_batch.shape, 'y_test dtype:', y_test_batch.dtype)



# batch = 0   # First sequence in the batch.
# signal_ = 0  # First signal from the 20 input-signals.
# seq = x_train_batch[batch, : ]
# plt.figure(figsize=(15,5))
# plt.grid(color='b', linestyle='-.', linewidth=0.5)
# plt.plot(seq)
# seq = y_train_batch[batch, : ]
# plt.figure(figsize=(15,5))
# plt.grid(color='b', linestyle='-.', linewidth=0.5)
# plt.plot(seq)

# batch = 0   # First sequence in the batch.
# signal_ = 0  # First signal from the 20 input-signals.
# seq = x_val_batch[batch, : ]
# plt.figure(figsize=(15,5))
# plt.grid(color='b', linestyle='-.', linewidth=0.5)
# plt.plot(seq)
# seq = y_val_batch[batch, : ]
# plt.figure(figsize=(15,5))
# plt.grid(color='b', linestyle='-.', linewidth=0.5)
# plt.plot(seq)

# batch = 0   # First sequence in the batch.
# signal_ = 0  # First signal from the 20 input-signals.
# seq = x_test_batch[batch, : ]
# plt.figure(figsize=(15,5))
# plt.grid(color='b', linestyle='-.', linewidth=0.5)
# plt.plot(seq)
# seq = y_test_batch[batch, : ]
# plt.figure(figsize=(15,5))
# plt.grid(color='b', linestyle='-.', linewidth=0.5)
# plt.plot(seq)
  
np.isnan(x_train_batch).any(), np.isnan(x_val_batch).any(), np.isnan(x_test_batch).any()

# %%

# %%
learning_rate = 1e-4
weight_decay = 1e-4
dropout_rate = 0.5
projection_dim = num_y_signals                 # Dimension of the patch representation or embedding to be used (non-restrictive) in this case we are assuming that our embedding has the same dimensionality as our features which means we could technically sckip the embedding layer altogether.
num_heads = num_y_signals                      # Total number of differnt copies of Q K V matrices. These will be aggregated as you move from one layer to the next.
transformer_units = [projection_dim*2, projection_dim]
transformer_layers = 4                            # Total number of complete transformer blocks or layers to stack (non-restrictive). Mine computation, network size, GPU memory, speed, etc.
mlp_head_units = [num_y_signals*2, num_y_signals]         # This is your model output head (non-restrictive). Size and activation functions depend on task to be performed.


# %% [markdown]
# ## Implement Multilayer Perceptron (MLP)

# %%
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation='gelu')(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

# %% [markdown]
def create_model():
    inputs = keras.Input(name='InputLayer', shape=(num_x_signals))
    
    # Create Embedding.
    inputs = tf.math.multiply_no_nan(inputs, 1000)
    # embeddings = tf.expand_dims(inputs, axis=-1)
    embeddings = Embedding(input_dim=num_x_signals, output_dim=projection_dim)(inputs)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = LayerNormalization(epsilon=1e-3)(embeddings)                          # Normalize input 
        x1 = (embeddings)                          # Normalize input 
        # print(f'x1 shape {x1.shape}') 
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, 
                                                     key_dim=projection_dim, 
                                                     dropout=dropout_rate)(x1, x1)
                                                     
        # print(f'Attention Unit {i+1} Output shape {attention_output.shape}')                   # Don't undertand the (x1, x1) input
        
        # Skip connection 1.
        x2 = layers.Add()([attention_output, embeddings])                                 # Elementwise addition of attention_output matrix and initial or processed/received matrix of original encoded_patches
        # print(f'x2 shape {x2.shape}') 

        # Layer normalization 2.
        x3 = LayerNormalization(epsilon=1e-3)(x2)                                       # Normalize array
        # x3 = (x2)                                       # Normalize array
        # print(f'x3 shape {x3.shape}') 
        
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=dropout_rate)                         # Why not just a single layer having number of units equal to projection dimension?               
        # print(f'x3 MLP output shape {x3.shape}') 
        
        # Skip connection 2.
        embeddings = layers.Add()([x3, x2])                                               # Output of transformer block to be fed as an initial input to the next block or on to prediciton stage

    # Create a [batch_size, projection_dim] tensor.
    representation = LayerNormalization(epsilon=1e-3)(embeddings)
    # representation = (embeddings)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(dropout_rate)(representation)
    # print(f'Transformer encoded representation shape {representation.shape}') 

    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=dropout_rate)

    # Add Output.
    outputs = Dense(units=num_y_signals,  activation='gelu', use_bias=True)(features)    
    
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=outputs, name='Crime_Transformer_Model')

    # Create Optimizer.
    optimizer = Adam(learning_rate=1e-3, amsgrad=True)
    # moving_avg_Adam = tfa.optimizers.MovingAverage(optimizer)
    stocastic_avg_Adam = tfa.optimizers.SWA(optimizer)
    
    model.compile(loss='mse', optimizer=stocastic_avg_Adam, metrics=['mse', 'mae', 'mape', 'acc'], run_eagerly=True) 

    return model

CRIME_model = create_model()

# CRIME_model.summary()

plot_model(CRIME_model, show_shapes=True, to_file='CRIME_model.png', show_layer_names=True, rankdir='TB', expand_nested=True, dpi=50)


# batch = 0   # First sequence in the batch.
# signal_ = 0  # First signal from the 20 input-signals.
# seq = x_test_batch[batch, :, signal_]
# plt.figure(figsize=(15,5))
# plt.grid(color='b', linestyle='-.', linewidth=0.5)
# plt.plot(seq)

# seq = y_test_batch[batch, :, signal_]
# plt.figure(figsize=(15,5))
# plt.grid(color='b', linestyle='-.', linewidth=0.5)
# plt.plot(seq)

y_test_pred = CRIME_model.predict(x_test_batch, steps=1, verbose=1)
print('y_predict shape: ', y_test_pred.shape, 'y_predict dtype:', y_test_pred.dtype)
# seq = y_test_pred[signal_, :]
# plt.figure(figsize=(15,5))
# plt.grid(color='b', linestyle='-.', linewidth=0.5)
# plt.plot(seq)

# CRIME_model.fit(x_test_batch,y_test_batch, epochs=10, verbose=1)


# %% [markdown]
# ### Callback Functions
# 
# During training we want to save checkpoints and log the progress to TensorBoard so we create the appropriate callbacks for Keras.
# This is the callback for writing checkpoints during training.

# %%
path_checkpoint = r'CRIME_model_3.h5'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=False,
                                      restore_best_weights=True,
                                      save_best_only=True)

# %%
# %%
path_checkpoint_MA = r'CRIME_model_avg_MA.h5'
path_checkpoint_SWA = r'CRIME_model_avg_SWA.h5'

callback_MA = tfa.callbacks.AverageModelCheckpoint(filepath=path_checkpoint_MA, 
                                                    update_weights=True)

callback_SWA = tfa.callbacks.AverageModelCheckpoint(filepath=path_checkpoint_SWA, 
                                                    update_weights=True)


# %% [markdown]
# This is the callback for stopping the optimization when performance worsens on the validation-set.

# %%
callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=400, verbose=1) 

# %% [markdown]
# This is the callback for writing the TensorBoard log during training.

# %%
dirpaths = [Path('.\Tensorboard_3')]
for dirpath in dirpaths:
    if dirpath.exists() and dirpath.is_dir():
        try:        
            shutil.rmtree(dirpath, ignore_errors=True)
            os.chmod(dirpath, 0o777)
            os.rmdir(dirpath)
            os.removedirs(dirpath)
            print("Directory '%s' has been removed successfully", dirpath)
        except OSError as error:
            print(error)
            print("Directory '%s' can not be removed", dirpath)
            
    
callback_tensorboard = TensorBoard(log_dir=r'TensorBoard_3',
                                   histogram_freq=1,
                                   write_graph=True)
                                  #  profile_batch = '300,320')

# %% [markdown]
# This callback reduces the learning-rate for the optimizer if the validation-loss has not improved since the last epoch (as indicated by `patience=0`). The learning-rate will be reduced by multiplying it with the given factor. We set a start learning-rate of 1e-3 above, so multiplying it by 0.1 gives a learning-rate of 1e-4. We don't want the learning-rate to go any lower than this.

# %%
callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.9999,
                                       min_lr=1e-7,
                                       patience=1,
                                       verbose=1)
                                    

# %%
callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_SWA,
             callback_tensorboard,
             callback_reduce_lr]


# %% [markdown]
# #### Load weights from last checkpoint

# %%
filepath = r'CRIME_model_3.h5'
def train_model(resume, epochs, initial_epoch, batch_size, model):
    def fit_model():
        with tf.device('/device:GPU:0'):
            print(model.summary())
            history=model.fit(x_train, y_train, 
                              verbose=1, 
                              callbacks=callbacks,
                              validation_split=0.2, 
                              epochs=EPOCHS,
                              batch_size=32,
                              #validation_freq=5,
                              #class_weight=None, 
                              #max_queue_size=10, 
                              #workers=8, 
                              #use_multiprocessing=True,
                              shuffle=True)
            model.load_weights(path_checkpoint)            
            model.save(filepath)
            model.evaluate(x_test_generator, steps=test_validation_steps)
        
            return history
    
    if resume:
        try:
            #del model
            model = load_model(filepath, custom_objects = {"CustomLoss": CustomLoss})
            # model.load_weights(path_checkpoint)
            # print(model.summary())
            print("Model loading....")
            model.evaluate(x_test_generator, steps=test_validation_steps)
            
        except Exception as error:
            print("Error trying to load checkpoint.")
            print(error)
        
    # Training the Model
    return fit_model()
    


# %%
EPOCHS = 2000

steps_per_epoch = int(num_train/batch_size)

for _ in range(1):
  try:
    # CRIME_model.load_weights(r'CRIME_model_3.h5')/
    CRIME_model.save(r'CRIME_model_3.h5')
    CRIME_model = load_model(r'CRIME_model_3.h5')
    print("Checkpoint Loaded.")
  except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)
    
# Train model
with tf.device('/device:GPU:0'):
    history = train_model(resume=False, epochs=EPOCHS, initial_epoch=0, batch_size=batch_size, model=CRIME_model)
    CRIME_model.history

# %% [markdown]
# ## Performance on Test-Set

# %% [markdown]
# ### Load Checkpoint
# 
# Because we use early-stopping when training the model, it is possible that the model's performance has worsened on the test-set for several epochs before training was stopped. We therefore reload the last saved checkpoint, which should have the best performance on the test-set.

# %%
with tf.device('/device:GPU:0'):
    try:
        CRIME_model = load_model(r'CRIME_model_3.h5')
        # CRIME_model.load_weights(path_checkpoint)
    except Exception as error:
        print("Error trying to load checkpoint.")
        print(error)

# %% [markdown]
# We can now evaluate the model's performance on the test-set. This function expects a batch of data, but we will just use one long time-series for the test-set, so we just expand the array-dimensionality to create a batch with that one sequence.

# %%
with tf.device('/device:GPU:0'):
    CRIME_model.evaluate(x_train_generator, steps=5)
    CRIME_model.evaluate(x_val_generator, steps=5)
    CRIME_model.evaluate(x_test_generator, steps=5)
