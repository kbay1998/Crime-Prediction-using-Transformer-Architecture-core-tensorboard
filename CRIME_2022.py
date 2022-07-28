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
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from numpy import genfromtxt
from numba import njit, cuda,jit 
from sklearn.model_selection import train_test_split
import cupy
import pylab as pl
import seaborn as sns
from pathlib import Path
import shutil


# %%
# %%
import tensorflow.keras.backend as K
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

# %%
# tf.debugging.experimental.enable_dump_debug_info('.', tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
# tf.debugging.set_log_device_placement(True)
from tensorflow.python.client import device_lib

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
file=r'crime_data.csv'

myData = pd.read_csv(file, delimiter=',')
# myData.round(decimals=6)
# myData=myData.astype(np.float32)
# myData=myData.astype(np.float16)
myData.describe()

# %%
# myData.convert_dtypes('float16')
myData.dtypes, myData.shape


# %%
myData

# %%
unique_states = myData['state'].unique()
unique_communities = myData['communityname'].unique()
myData.set_index('communityname', inplace=True)
myData = myData.drop(columns = ['state'])
myData

# %%
myData = myData.replace('?', np.NaN)
myData.isnull().values.any()
myData

# %%
is_NaN = myData.isnull()
row_has_NaN = is_NaN.any(axis=1)
myDataDirty= myData[row_has_NaN]
myDataDirty.shape

# %%
myDataClean = myData.dropna()
myDataClean.shape

# %% [markdown]
# List of the variables used in the data-set.

# %%
data_top = myData.columns.values
data_top

# %% [markdown]
# These are the top rows of the data-set.

# %%
myData.head()

# %%
from tqdm.notebook import tqdm_notebook
import googlemaps

gmaps_key =  googlemaps.Client(key = r"AIzaSyCEVqDcocu8p57kRCE54gNZV-PL82xYLxI")


print(lat,lon)

myData['LAT'] = None
myData['LON'] = None

for idx in tqdm_notebook(range(len(myData)), desc = 'Loop 1'):
    try:
        print(myData.iloc[idx]['communityname'], myData.iloc[idx]['state'])
        location = myData.iloc[idx]['communityname'] + ' , ' +  myData.iloc[idx]['state']
        geocode_result = gmaps_key.geocode(location)
        lat =  geocode_result[0]['geometry']['location']['lat']
        lon =  geocode_result[0]['geometry']['location']['lng']
        myData['LAT'].iloc[idx] = lat
        myData['LON'].iloc[idx] = lon
    except Exception as error:
        print("Error trying to load weights.")
        print(error)

myData.to_csv('crimeData.csv') 

# %%
cleanDataValues = myDataClean.values[:,2:].astype(np.float32)
cleanDataValues

# %%
dirtyDataValues = myDataDirty.values[:,2:].astype(np.float32)
dirtyDataValues

# %%
tfp.stats.correlation( myDataClean, y=None, sample_axis=0, event_axis=-1, keepdims=False, name=None)
myDataClean_corr = myDataClean.corr()
myDataClean_corr.where(np.tril(np.ones(myDataClean_corr.shape)).astype(np.bool_)).style.background_gradient(cmap='coolwarm').set_properties(**{'font-size': '5'})
# 'RdBu_r', 'BrBG_r', & PuOr_r are other good diverging colormaps
myDataClean_corr.describe()


# %%

# tfp.stats.correlation( myDataClean, y=None, sample_axis=0, event_axis=-1, keepdims=False, name=None)
myData_corr = myData.corr()
myData_corr.where(np.tril(np.ones(myData_corr.shape)).astype(np.bool_)).style.background_gradient(cmap='coolwarm').set_properties(**{'font-size': '5'})
# 'RdBu_r', 'BrBG_r', & PuOr_r are other good diverging colormaps
myData_corr.describe()

# %%
# tfp.stats.correlation( myDataClean, y=None, sample_axis=0, event_axis=-1, keepdims=False, name=None)
myDataDirty_corr = myDataDirty.corr()
myDataDirty_corr.where(np.tril(np.ones(myDataDirty_corr.shape)).astype(np.bool_)).style.background_gradient(cmap='coolwarm').set_properties(**{'font-size': '5'})
# 'RdBu_r', 'BrBG_r', & PuOr_r are other good diverging colormaps
myDataDirty_corr.describe()
myDataDirty_corr.shape

# %%
# tfp.stats.correlation( myDataClean, y=None, sample_axis=0, event_axis=-1, keepdims=False, name=None)
new_corr = pd.concat([myDataClean_corr, myDataDirty_corr], axis=1).corr()
new_corr = new_corr.loc[:,~new_corr.columns.duplicated()]
new_corr = new_corr.iloc[:-int(len(new_corr)/2)]
new_corr = new_corr.where(np.tril(np.ones(new_corr.shape)).astype(np.bool_))
new_corr.style.background_gradient(cmap='coolwarm').set_properties(**{'font-size': '5'})
# 'RdBu_r', 'BrBG_r', & PuOr_r are other good diverging colormaps
new_corr.describe()

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
plt.figure(figsize=(30,5*16))
myDataClean.plot(subplots=True, figsize=(30,5*16))
plt.grid(color='b', linestyle='-.', linewidth=0.5)
plt.show()


# %% [markdown]
# Before removing these two signals, there are 20 input-signals in the data-set.

# %% [markdown]
# Now there are only 18 input-signals in the data.

# %% [markdown]
# We can verify that these two data-columns have indeed been removed.

# %%
myDataClean.head(1)

# %% [markdown]
# myData[0] = myData.index
# myData =myData.reindex()
# myData[0]

# %% [markdown]
# ### Target Data for Prediction
# 
# We will try and predict the future Forex-data.

# %%
input_names = myDataClean.columns[:]
target_names = myDataClean.columns[:]

# %% [markdown]
# We will try and predict these signals.

# %%
df_targets = myDataClean[target_names]
df_targets

# %% [markdown]
# The following is the first 5 rows of the time-shifted data-frame. This should be identical to the last 5 rows shown above from the original data, except for the time-stamp.

# %%
df_targets.head(5)

# %% [markdown]
# The time-shifted data-frame has the same length as the original data-frame, but the last observations are `NaN` (not a number) because the data has been shifted backwards so we are trying to shift data that does not exist in the original data-frame.

# %%
df_targets.tail()

# %% [markdown]
# ### NumPy Arrays
# 
# We now convert the Pandas data-frames to NumPy arrays that can be input to the neural network. We also remove the last part of the numpy arrays, because the target-data has `NaN` for the shifted period, and we only want to have valid data and we need the same array-shapes for the input- and output-data.
# 
# These are the input-signals:

# %%
x_data = myDataClean[input_names].values.astype(np.float32)
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
train_split = 0.9
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
# x_scaler = MinMaxScaler()
# y_scaler = MinMaxScaler()
x_scaler = MinMaxScaler().fit(myData[input_names].values.astype(np.float32))
y_scaler = MinMaxScaler().fit(myData[target_names].values.astype(np.float32))
# x_data_scaled = x_scaler.fit_transform(x_data)
# y_data_scaled = y_scaler.fit_transform(y_data)
x_data_scaled = x_scaler.transform(x_data)
y_data_scaled = y_scaler.transform(y_data)

x_train, x_test, y_train, y_test = train_test_split(x_data_scaled, y_data_scaled, train_size=train_split, random_state=None, shuffle=False)


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
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


batch_size = 1
sequence_length = 1
mask_percentage =.05

with tf.device('/device:GPU:0'):
    # @tf.function(experimental_relax_shapes=True)
    def mask_data(input_data, mask_percentage=0.05):
        num_masks = int(mask_percentage * input_data.shape[-1])
        mask_indices = np.random.randint(0, input_data.shape[0], size=num_masks)
        
        mask = np.zeros(shape = input_data.shape)
        for i in mask_indices:
            mask[i] = 1

        masked_array = ma.array(input_data, mask = mask)

        # pd.DataFrame(masked_array).plot()
        # pd.DataFrame(mask).plot()
        # pd.DataFrame(x_train[0]).plot()
        return ma.asarray(masked_array)


class CustomDataGen(tf.keras.utils.Sequence):
        
    def __init__ (self, x_data, y_data, batch_size=None, sequence_length=None, train=True, validation=True, mask_percentage=0.01, random_batch=False, random_idx=False):
        
        self.x_train = x_data[0]
        self.x_test = x_data[1]        
        self.y_train = y_data[0]
        self.y_test = y_data[1]
        # self.x_train = MinMaxScaler().fit_transform(x_data[0])
        # self.x_test = MinMaxScaler().fit_transform(x_data[1])        
        # self.y_train = MinMaxScaler().fit_transform(y_data[0])
        # self.y_test = MinMaxScaler().fit_transform(y_data[1])
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
        
        x_shape = (batch_size_, sequence_length_, x_samples.shape[1])
        y_shape = (batch_size_, sequence_length_, y_samples.shape[1])
        # x_batch = np.zeros(shape=x_shape, dtype=np.float32)  
        # y_batch = np.zeros(shape=y_shape, dtype=np.float32)        
            
        # Fill the batch with random sequences of data.
        for i in range(batch_size_):
            # Get a random start-index.

            if self.random_idx:
                sample_idx = np.random.randint(1, x_samples.shape[-2])


            # This points somewhere into the training-data.
            x_batch = mask_data(x_samples[sample_idx], mask_percentage=self.mask_percentage)  
            y_batch = y_samples[sample_idx]

        return np.ma.expand_dims(x_batch, axis=0), np.ma.expand_dims(y_batch, axis=0)    
    
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
# %% [markdown]
# ### Alternative Keras Sequence Generator

# %% [markdown]
# ## Create the Recurrent Neural Network

# %%
with tf.device('/device:GPU:0'):
    @tf.function(experimental_relax_shapes=True)
    def signalPower(x):
        return tf.math.reduce_mean(tf.math.square(x))


with tf.device('/device:GPU:0'):

    @tf.function(experimental_relax_shapes=True)
    def SNR_compute(y_true, y_pred):
        signal_power = signalPower(y_pred)
        noise_power = signalPower(tf.math.subtract(tf.math.abs(y_pred), tf.math.abs(y_true)))
        SNR_p = tf.math.divide_no_nan(tf.math.subtract(signal_power, noise_power), noise_power)
        SNR_p_DB = tf.math.multiply_no_nan(tf.experimental.numpy.log10(SNR_p), 10)
        return SNR_p_DB

with tf.device('/device:GPU:0'):
    class SNR_dB(tf.keras.metrics.Metric):
        def __init__(self, name="SNR_dB", **kwargs):
            super(SNR_dB, self).__init__(name=name, **kwargs)
            self.SNR_compute = SNR_compute
            self.SNR = None
    
        def update_state(self, y_true, y_pred, sample_weight=None):
            self.SNR = self.SNR_compute(y_true, y_pred)
            return
        
        def result(self):
            return self.SNR

        def reset_state(self):
            # The state of the metric will be reset at the start of each epoch.
            self.SNR = None
            return

with tf.device('/device:GPU:0'):
    class CustomScalingLayer(layers.Layer):
    
        def __init__(self, units=10, **kwargs):
            super(CustomScalingLayer, self).__init__(**kwargs)
            self.units = units
        
        # @tf.function( experimental_relax_shapes=True)
        def build(self, input_shape):
            self.built=True
            return

        # @tf.function( experimental_relax_shapes=True) 
        def call(self, inputs):
            return tf.vectorized_map(lambda x: tf.math.divide_no_nan(tf.math.subtract(x, tf.keras.backend.min(x, axis=0, keepdims=False)) , tf.math.subtract(tf.keras.backend.max(x, axis=0, keepdims=False) , tf.keras.backend.min(x, axis=0, keepdims=False)))
, inputs)

        # @tf.function( experimental_relax_shapes=True)
        def get_config(self):
            config = super().get_config().copy()
            config.update({'units': self.units})
            return config

        @classmethod
        def from_config(cls, config):
            return cls(**config)

    layer=CustomScalingLayer(10)


with tf.device('/device:GPU:0'):
    @tf.function( experimental_relax_shapes=True)
    def loss_mse(y_true, y_pred):
        # Calculate the MSE loss for each value in these tensors.
        # This outputs a 3-rank tensor of the same shape.
        loss = tf.keras.losses.mean_squared_error(y_true=y_true,y_pred=y_pred)
        loss = tf.math.multiply_no_nan(loss, 1, name=None)
        loss_mean = tf.reduce_mean(loss)
        return loss_mean

@tf.function( experimental_relax_shapes=True)
def CustomLoss(y_true, y_pred):
    # Calculate the MSE MAE MAPE MSLEloss for each value in these tensors.
    # This outputs a 3-rank tensor of the same shape.
    mse = tf.keras.losses.MeanSquaredError()
    mae = tf.keras.losses.MeanAbsoluteError()
    mape = tf.keras.losses.MeanAbsolutePercentageError()
    msle = tf.keras.losses.MeanSquaredLogarithmicError()
    y_true = tf.convert_to_tensor(y_true)
    y_pred = tf.convert_to_tensor(y_pred)
    mse = mse(y_true, y_pred)
    mae = mae(y_true, y_pred)
    mape = mape(y_true, y_pred)
    msle = msle(y_true, y_pred)
    loss = tf.add_n([mae, mse, mape, msle])
    # loss_mean = tf.reduce_mean(loss)
    return loss

# %%

with tf.device('/device:GPU:0'):
    
    def create_model():
        inputs = keras.Input(name='InputLayer', shape=(num_x_signals))
        dropout=0.001
        num_layers=25
        num_units = num_x_signals
        steps = 5

        x_encode = Dense(units=num_units,  activation='selu', use_bias=True)(inputs)
        x_encode = tf.expand_dims((x_encode), axis=-2)
        x_encode = Conv1D(filters=num_units, kernel_size=4, strides=1, activation='selu', data_format='channels_first')(x_encode)
        x_encode = Dropout(dropout)(x_encode)

        for i in range(num_layers):
            num_units = num_units - steps
            # x_encode = LayerNormalization()(x_encode)
            x_encode = Dense(units=num_units,  activation='selu', use_bias=True)(x_encode)
            # x_encode = tf.expand_dims((x_encode), axis=1)
            x_encode = Conv1D(filters=num_units, kernel_size=4, strides=1, activation='selu', data_format='channels_first')(x_encode)
            x_encode = Dropout(dropout)(x_encode)
            
        x_decode=x_encode
        
        for i in range(num_layers):
            num_units = num_units + steps
            # x_decode = LayerNormalization()(x_decode)
            # x_decode = tf.expand_dims((x_decode), axis=1)
            x_decode = Conv1D(filters=num_units, kernel_size=4, strides=1, activation='selu', data_format='channels_first')(x_decode)
            x_decode = Dense(units=num_units,  activation='selu', use_bias=True)(x_decode)
            x_decode = Dropout(dropout)(x_decode)

        # x_decode = LayerNormalization()(x_decode)
        # x_decode = tf.expand_dims((x_decode), axis=1)
        x_decode = Conv1D(filters=num_units, kernel_size=4, strides=1, activation='selu',  data_format='channels_first')(x_decode)
        x_decode = Flatten()(x_decode)
        x_decode = Dropout(dropout)(x_decode)
        x_decode = Dense(units=num_y_signals,  activation='selu', use_bias=True)(x_decode)
        outputs = Reshape(name='ReshapeLayer', target_shape=(-1, num_y_signals))(x_decode)
        
        optimizer = Adam(learning_rate=1e-3, amsgrad=True) 
        CRIME_model = Model(inputs, outputs, name='CRIME_model')
        CRIME_model.compile(loss='mse', optimizer=optimizer, run_eagerly=True)
        # CRIME_model.build((None, 1, num_x_signals))
        # CRIME_model.summary()

        return CRIME_model

    CRIME_model = create_model()

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
    
    y_test_batch1 = CRIME_model.predict(x_test_batch, steps=1, verbose=1)
    print('y_predict shape: ', y_test_batch1.shape, 'y_predict dtype:', y_test_batch1.dtype)
    # seq = y_test_batch1[batch, :, signal_]
    # plt.figure(figsize=(15,5))
    # plt.grid(color='b', linestyle='-.', linewidth=0.5)
    # plt.plot(seq)
    # CRIME_model.summary()

    # y_test_batch1 = CRIME_model.predict(x_test_batch, steps=1, verbose=1)
    # GRU_model.fit(x_test_batch,y_test_batch, epochs=10, verbose=1)/
    

# %% [markdown]
# ### Callback Functions
# 
# During training we want to save checkpoints and log the progress to TensorBoard so we create the appropriate callbacks for Keras.
# This is the callback for writing checkpoints during training.

# %%
path_checkpoint = r'CRIME_model2.h5'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=False,
                                      restore_best_weights=True,
                                      save_best_only=True)

# %% [markdown]
# This is the callback for stopping the optimization when performance worsens on the validation-set.

# %%
callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=400, verbose=1) #15

# %% [markdown]
# This is the callback for writing the TensorBoard log during training.

# %%
dirpaths = [Path('.\Tensorboard')]
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
            
    
callback_tensorboard = TensorBoard(log_dir=r'TensorBoard',
                                   histogram_freq=1,
                                   write_graph=True)
                                   # profile_batch = '500,520')

# %% [markdown]
# This callback reduces the learning-rate for the optimizer if the validation-loss has not improved since the last epoch (as indicated by `patience=0`). The learning-rate will be reduced by multiplying it with the given factor. We set a start learning-rate of 1e-3 above, so multiplying it by 0.1 gives a learning-rate of 1e-4. We don't want the learning-rate to go any lower than this.

# %%
callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.95,
                                       min_lr=1e-5,
                                       patience=10,
                                       verbose=1)
                                    

# %%
callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr,]


# %% [markdown]
# #### Load weights from last checkpoint

# %%
filepath = r'CRIME_model2.h5'
def train_model(resume, epochs, initial_epoch, batch_size, model):
    def fit_model():
        with tf.device('/device:GPU:0'):
            print(model.summary())
            history=model.fit(x_train_generator, 
                              steps_per_epoch=steps_per_epoch, 
                              epochs=EPOCHS, 
                              verbose=1, 
                              callbacks=callbacks,
                              validation_data=x_val_generator, 
                              validation_steps=train_validation_steps, 
                              initial_epoch=initial_epoch)
            model.load_weights(path_checkpoint)            
            model.save(filepath)
            model.evaluate(x_test_generator, steps=test_validation_steps)
        
            return history
    
    if resume:
        try:
            #del model
            model = load_model(filepath, custom_objects = {"CustomScalingLayer": CustomScalingLayer, "SNR_dB": SNR_dB, "CustomLoss": CustomLoss})
            # model.load_weights(path_checkpoint)
            print(model.summary())
            print("Model loading....")
            model.evaluate(x_test_generator, steps=test_validation_steps)
            
        except Exception as error:
            print("Error trying to load checkpoint.")
            print(error)
        
    # Training the Model
    return fit_model()
    
with tf.device('/device:GPU:0'):
    def plot_train_history(history, title):
        loss = history.history['loss']
        accuracy = history.history['acc']
        mape = history.history['mape']
        mae = history.history['mae']
        val_loss = cupy.asnumpy(history.history['val_loss'])
        val_accuracy = cupy.asnumpy(history.history['val_acc'])
        val_mae = cupy.asnumpy(history.history['val_mae'])
        val_mape = cupy.asnumpy(history.history['val_mape'])
        epochs = range(len(loss))
        plt.figure(figsize=(30,5))
        plt.plot(epochs, loss, label='training_loss') 
        plt.plot(epochs, val_loss, label='validation_loss')
        plt.show()
        plt.figure(figsize=(30,5))
        plt.plot(epochs, accuracy, label='training_accuracy') 
        plt.plot(epochs, val_accuracy, label='validation_accuracy')
        plt.show()
        plt.figure(figsize=(30,5))
        plt.plot(epochs, mae, label='training_mae') 
        plt.plot(epochs, val_mae, label='validation_mae')
        plt.show()
        plt.figure(figsize=(30,5))
        plt.plot(epochs, mape, label='training_mape') 
        plt.plot(epochs, val_mape, label='validation_mape')
        plt.show()
        return

# %%
EPOCHS = 2000

# steps_per_epoch = int((num_train/batch_size)/10)
steps_per_epoch = int(235)

for _ in range(1):
  try:
    CRIME_model.load_weights(r'CRIME_model2.h5')
    # CRIME_model = load_model(r'CRIME_model.h5')
    optimizer = Adam(learning_rate=1e-4, amsgrad=True) 
    CRIME_model.compile(loss='mse', optimizer=optimizer, metrics=['mse', 'mae', 'mape'], run_eagerly=True)
    CRIME_model.trainable=True
    print("Loaded checkpoint.")
  except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)

# Train model
with tf.device('/device:GPU:0'):
    history = train_model(resume=False, epochs=EPOCHS, initial_epoch=0, batch_size=batch_size, model=CRIME_model)
    plot_train_history(history, 'Model Training History')
    CRIME_model.history

# %% [markdown]
# ### Load Checkpoint
# 
# Because we use early-stopping when training the model, it is possible that the model's performance has worsened on the test-set for several epochs before training was stopped. We therefore reload the last saved checkpoint, which should have the best performance on the test-set.

# %%
with tf.device('/device:GPU:0'):
    try:
        CRIME_model = load_model(r'CRIME_model2.h5')
        # CRIME_model.load_weights(path_checkpoint)
    except Exception as error:
        print("Error trying to load checkpoint.")
        print(error)

# %% [markdown]
# ## Performance on Test-Set
# 
# We can now evaluate the model's performance on the test-set. This function expects a batch of data, but we will just use one long time-series for the test-set, so we just expand the array-dimensionality to create a batch with that one sequence.

# %%
with tf.device('/device:GPU:0'):
    CRIME_model.evaluate(x_train_generator, steps=train_validation_steps)
    CRIME_model.evaluate(x_val_generator, steps=train_validation_steps)
    CRIME_model.evaluate(x_test_generator, steps=test_validation_steps)

