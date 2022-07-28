# %% [markdown]
# ## Imports

# %%
!pwd

# %% [markdown]
# We need to import several things from Keras.

# %%
try:
  from google.colab import drive
  IN_COLAB=True
except:
  IN_COLAB=False

if IN_COLAB:
  print("We're running Colab")

if IN_COLAB:
  # Mount the Google Drive at mount
  mount='/content/drive'
  print("Colab: mounting Google drive on ", mount)

  drive.mount(mount)

  # Switch to the directory on the Google Drive that you want to use
  import os
  drive_root = mount + "/My Drive/Colab Notebooks/Crime"
  
  # Create drive_root if it doesn't exist
  create_drive_root = True
  if create_drive_root:
    print("\nColab: making sure ", drive_root, " exists.")
    os.makedirs(drive_root, exist_ok=True)
  
  # Change to the directory
  print("\nColab: Changing directory to ", drive_root)
  %cd $drive_root
  !pwd

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.datasets import make_regression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from numpy import genfromtxt
import pylab as pl
import seaborn as sns
from pathlib import Path
import shutil
from tqdm.notebook import tqdm_notebook
# !pip install googlemaps
# !pip install tensorflow_addons
# import googlemaps
import sys
import math

# %%
# import tensorflow.keras.backend as K
!pip install tensorflow_addons
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
# data_scaler = QuantileTransformer()
# myData[myData.columns] = data_scaler.fit_transform(myData.values)
myData.min(), myData.max()

# %%
pd.plotting.scatter_matrix(myData[target_names], diagonal='hist', figsize=(30,30))
plt.savefig('matrix.png')

# %%
# tfp.stats.correlation( newFilled, y=None, sample_axis=0, event_axis=-1, keepdims=False, name=None)
myData_corr = myData.corr()[target_names][:-18]
# myData_corr = myData_corr.unstack().sort_values().drop_duplicates().[((myData_corr >= .9) | (myData_corr <= -.9)) & (myData_corr !=1.000)]
# myData_corr = myData_corr[((myData_corr >= .9) | (myData_corr <= -.9)) & (myData_corr !=1.000)].drop_duplicates()
myData_corr.where(np.tril(np.ones(myData_corr.shape)).astype(np.bool_)).style.background_gradient(cmap='coolwarm').set_properties(**{'font-size': '5'})
myData_corr.style.background_gradient(cmap='coolwarm').set_properties(**{'font-size': '5'})

# 'RdBu_r', 'BrBG_r', & PuOr_r are other good diverging colormaps
# newFilled_corr.describe()

# %% [markdown]
# tfp.stats.correlation( myDataClean, y=None, sample_axis=0, event_axis=-1, keepdims=False, name=None)
new_corr = pd.concat([myDataClean_corr, myDataDirty_corr], axis=1).corr()
new_corr = new_corr.loc[:,~new_corr.columns.duplicated()]
new_corr = new_corr.iloc[:-int(len(new_corr)/2)]
new_corr = new_corr.where(np.tril(np.ones(new_corr.shape)).astype(np.bool_))
new_corr.style.background_gradient(cmap='coolwarm').set_properties(**{'font-size': '5'})
# 'RdBu_r', 'BrBG_r', & PuOr_r are other good diverging colormaps
new_corr.describe()

# %%
# tfp.stats.correlation( newFilled, y=None, sample_axis=0, event_axis=-1, keepdims=False, name=None)
myData_corr = myData[input_names].corr()
myData_corr.where(np.tril(np.ones(myData_corr.shape)).astype(np.bool_)).style.background_gradient(cmap='coolwarm').set_properties(**{'font-size': '5'})
myData_corr.style.background_gradient(cmap='coolwarm').set_properties(**{'font-size': '5'})

# 'RdBu_r', 'BrBG_r', & PuOr_r are other good diverging colormaps
# newFilled_corr.describe()

# %%
myData_corr = pd.DataFrame(myData[input_names].corr().unstack().sort_values().drop_duplicates())
myData_corr = myData_corr[~myData_corr.iloc[:, 0].between(-.9, .97, inclusive=True)]
myData_corr.where(np.tril(np.ones(myData_corr.shape)).astype(np.bool_)).style.background_gradient(cmap='coolwarm').set_properties(**{'font-size': '5'})
myData_corr = myData_corr.style.background_gradient(cmap='coolwarm').set_properties(**{'font-size': '5'})
myData_corr.columns.value=['coeff']
myData_corr

# %%
myData_corr = pd.DataFrame(myData[target_names].corr().unstack().sort_values().drop_duplicates())
myData_corr = myData_corr[~myData_corr.iloc[:, 0].between(-.9, .9, inclusive=True)]
myData_corr.where(np.tril(np.ones(myData_corr.shape)).astype(np.bool_)).style.background_gradient(cmap='coolwarm').set_properties(**{'font-size': '5'})
myData_corr = myData_corr.style.background_gradient(cmap='coolwarm').set_properties(**{'font-size': '5'})
myData_corr

# %%
myData_corr = myData.corr()[target_names][:-18]
myData_corr = pd.DataFrame(myData_corr.unstack().sort_values().drop_duplicates())
myData_corr = myData_corr[~myData_corr.iloc[:, 0].between(-.9, .9, inclusive=True)]
# myData_corr = myData_corr[((myData_corr >= .9) | (myData_corr <= -.9)) & (myData_corr !=1.000)].drop_duplicates()
myData_corr.where(np.tril(np.ones(myData_corr.shape)).astype(np.bool_)).style.background_gradient(cmap='coolwarm').set_properties(**{'font-size': '5'})
myData_corr.style.background_gradient(cmap='coolwarm').set_properties(**{'font-size': '5'})

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
x_scaler = Normalizer()
y_scaler = Normalizer()
x_data_scaled = x_scaler.fit_transform(x_data)
y_data_scaled = y_scaler.fit_transform(y_data)

# %%
x_train, x_test, y_train, y_test = train_test_split(x_data_scaled, y_data_scaled, train_size=train_split, random_state=None, shuffle=True)
# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=train_split, random_state=2, shuffle=True)


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


batch = 0   # First sequence in the batch.
signal_ = 0  # First signal from the 20 input-signals.
seq = x_train_batch[batch, : ]
plt.figure(figsize=(15,5))
plt.grid(color='b', linestyle='-.', linewidth=0.5)
plt.plot(seq)
seq = y_train_batch[batch, : ]
plt.figure(figsize=(15,5))
plt.grid(color='b', linestyle='-.', linewidth=0.5)
plt.plot(seq)

batch = 0   # First sequence in the batch.
signal_ = 0  # First signal from the 20 input-signals.
seq = x_val_batch[batch, : ]
plt.figure(figsize=(15,5))
plt.grid(color='b', linestyle='-.', linewidth=0.5)
plt.plot(seq)
seq = y_val_batch[batch, : ]
plt.figure(figsize=(15,5))
plt.grid(color='b', linestyle='-.', linewidth=0.5)
plt.plot(seq)

batch = 0   # First sequence in the batch.
signal_ = 0  # First signal from the 20 input-signals.
seq = x_test_batch[batch, : ]
plt.figure(figsize=(15,5))
plt.grid(color='b', linestyle='-.', linewidth=0.5)
plt.plot(seq)
seq = y_test_batch[batch, : ]
plt.figure(figsize=(15,5))
plt.grid(color='b', linestyle='-.', linewidth=0.5)
plt.plot(seq)
  
np.isnan(x_train_batch).any(), np.isnan(x_val_batch).any(), np.isnan(x_test_batch).any()

# %%
myData_train, myData_test, _, _ = (myData, myData, train_size=train_split, random_state=None, shuffle=True)
myData = pd.DataFrame(myData.values, columns=myData.columns.values)
myData_train, myData_test, _, _ = train_test_split(myData, myData, train_size=train_split, random_state=2, shuffle=True)

train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(myData_train, label=target_names)
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(myData_test, label=target_names)

# %%
max_depth = 50
n_estimators = 200

regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=2))
regr_multirf.fit(x_train[:,2:], y_train)

regr_rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=2)
regr_rf.fit(x_train[:,2:], y_train)

regr_multirf_LL = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=2))
regr_multirf_LL.fit(x_train, y_train)

regr_rf_LL = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=2)
regr_rf_LL.fit(x_train, y_train)

mor = MultiOutputRegressor(SVR(epsilon=1e-6, gamma='auto', verbose=True, max_iter=-1))
mor = mor.fit(x_train[:,2:], y_train)

mor_LL = MultiOutputRegressor(SVR(epsilon=1e-6, gamma='auto', verbose=True, max_iter=-1))
mor_LL = mor_LL.fit(x_train, y_train)



# %%
pd.plotting.scatter_matrix(pd.DataFrame(y_train, columns=[target_names]), diagonal='hist', figsize=(30,30))
plt.savefig('matrix.png')

# %%
import joblib

filenames = ['regr_multirf.sav', 'regr_rf.sav', 'regr_multirf_LL.sav', 'regr_rf_LL.sav', 'mor.sav', 'mor_LL.sav']
# models = [regr_multirf, regr_rf, regr_multirf_LL, regr_rf_LL, mor, mor_LL]

# for model, filename in zip(models, filenames):
#   print(f'Saving {model} as {filename}')
#   joblib.dump(model, filename)
 
# load the model from disk
regr_multirf, regr_rf, regr_multirf_LL, regr_rf_LL, mor, mor_LL = [joblib.load(filename) for filename in filenames]

# %%
# Predict on new data
y_multirf = regr_multirf.predict(x_test[:,2:])
y_rf = regr_rf.predict(x_test[:,2:])
y_pred = mor.predict(x_test[:,2:])

y_multirf_LL = regr_multirf_LL.predict(x_test)
y_rf_LL = regr_rf_LL.predict(x_test)
y_pred_LL = mor_LL.predict(x_test)

# %%
# Plot the results
plt.figure()
s = 50
a = 0.4

plt.scatter(
    y_test[:, 0],
    y_test[:, 1],
    edgecolor="k",
    c="navy",
    s=s,
    marker="s",
    alpha=a,
    label="Data",
)

plt.scatter(
    y_multirf[:, 0],
    y_multirf[:, 1],
    edgecolor="k",
    c="cornflowerblue",
    s=s,
    alpha=a,
    label="Multi RF score=%.2f" % regr_multirf.score(x_test[:,2:], y_test),
)

plt.scatter(
    y_rf[:, 0],
    y_rf[:, 1],
    edgecolor="k",
    c="c",
    s=s,
    marker="^",
    alpha=a,
    label="RF score=%.2f" % regr_rf.score(x_test[:,2:], y_test),
)

plt.xlim([-6, 6])
plt.ylim([-6, 6])
plt.xlabel("target 1")
plt.ylabel("target 2")
plt.title("Comparing random forests and the multi-output meta estimator")
plt.legend()
plt.show()

# %% [markdown]
# #### Load weights from last checkpoint

# %% [markdown]
# ## Performance on Test-Set

# %%
# x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=train_split, random_state=2, shuffle=True)
X_test = x_test

# X_test[:,:] = 200


X_test[:1, :].shape

# %%
test_length = 3

mape_two =np.zeros(shape=(4, num_y_signals))
mae_two =np.zeros(shape=(4, num_y_signals))
mape_two_LL =np.zeros(shape=(4, num_y_signals))
mae_two_LL =np.zeros(shape=(4, num_y_signals))

mape_one =np.zeros(shape=(4))
mae_one =np.zeros(shape=(4))
mape_one_LL =np.zeros(shape=(4))
mae_one_LL =np.zeros(shape=(4))

# %%
model_number = 0

with tf.device('/device:GPU:0'):
    y_masked = np.zeros(shape=(test_length,num_x_signals))
    y_pred = np.zeros(shape=(test_length,num_y_signals))
    y_true = np.zeros(shape=(test_length,num_y_signals))


    for i in range (test_length):
            # x_test_batch, y_test_batch=x_train_generator.__getitem__(1)
            # y_masked = x_test_batch
            y_true = y_test + 2e-5
            y_pred = regr_multirf.predict(X_test[:,2:])
            y_pred_LL = regr_multirf_LL.predict(X_test)            
            break


# Evaluate the regressor
mae_one[model_number] = mean_absolute_error(y_scaler.inverse_transform(y_true), y_scaler.inverse_transform(y_pred)).round(2)
mae_one_LL[model_number] = mean_absolute_error(y_scaler.inverse_transform(y_true), y_scaler.inverse_transform(y_pred_LL)).round(2)
for i in range(num_y_signals):
  mae_two[model_number, i] = mean_absolute_error(y_true[:,i], y_pred[:,i]).round(2)
  mae_two_LL[model_number, i] = mean_absolute_error(y_true[:,i], y_pred_LL[:,i]).round(2)

mape_one[model_number] = mean_absolute_percentage_error(y_scaler.inverse_transform(y_true), y_scaler.inverse_transform(y_pred)).round(2).clip(0,100)
mape_one_LL[model_number] = mean_absolute_percentage_error(y_scaler.inverse_transform(y_true), y_scaler.inverse_transform(y_pred_LL)).round(2).clip(0,100)
for i in range(num_y_signals):
  mape_two[model_number, i] = mean_absolute_percentage_error(y_true[:,i], y_pred[:,i]).round(2).clip(0,100)
  mape_two_LL[model_number, i] = mean_absolute_percentage_error(y_true[:,i], y_pred_LL[:,i]).round(2).clip(0,100)



for sample in range(test_length):
        # Get the output-signal predicted by the model.
        signal_pred = y_pred[sample, :]
        signal_pred_LL = y_pred_LL[sample, :]

        # Get the true output-signal from the data-set.
        signal_true = y_true[sample, :]
        
        error = np.zeros(len(signal_true))
        p_error = np.zeros(len(signal_true))
        error_LL = np.zeros(len(signal_true))
        p_error_LL = np.zeros(len(signal_true))

        for i in range(len(signal_true)):
            error[i] = signal_true[i]-signal_pred[i]
            p_error[i] = mean_absolute_percentage_error(signal_true[i].reshape(-1,1), signal_pred[i].reshape(-1,1)).round(2)           
            error_LL[i] = signal_true[i]-signal_pred_LL[i]
            p_error_LL[i] = mean_absolute_percentage_error(signal_true[i].reshape(-1,1), signal_pred_LL[i].reshape(-1,1)).round(2)
        
        mae = mean_absolute_error(signal_true, signal_pred).round(2)
        mae_LL = mean_absolute_error(signal_true, signal_pred_LL).round(2)        
        mape = mean_absolute_percentage_error(signal_true, signal_pred).round(2)
        mape_LL= mean_absolute_percentage_error(signal_true, signal_pred_LL).round(2)
        
        # Make the plotting-canvas bigger.
        plt.figure(figsize=(10,5))
        plt.plot(signal_true, '-*', label='CRIME_True')
        plt.plot(signal_pred,  '-+', label='CRIME_Pred')     
        plt.plot(signal_pred_LL,  '-+', label='CRIME_Pred_LL')     
        plt.xlabel('Crime')
        plt.ylabel('Normalized Value')
        plt.title('Comparison between RFR and RFR_LL Predictions')
        plt.xticks(range(0,len(target_names)), target_names.values.tolist(), rotation=90)
        plt.grid(color='b', linestyle='-.', linewidth=0.5)
        plt.legend()
        plt.show()
        
        # # Make the plotting-canvas bigger.
        # plt.figure(figsize=(10,5))
        # plt.plot(error, label=f'MAE: {mae}')
        # plt.plot(error_LL, label=f'MAE_LL: {mae_LL}')
        # plt.xlabel('Crime')
        # plt.ylabel('Mean Absolute Error')
        # plt.title('Comparison between RFR and RFR_LL Prediction Absolute Errors')
        # plt.xticks(range(0,len(target_names)), target_names.values.tolist(), rotation=90)
        # plt.grid(color='b', linestyle='-.', linewidth=0.5)
        # plt.legend()
        # plt.show()

        # # Make the plotting-canvas bigger.
        # plt.figure(figsize=(10,5))
        # plt.plot(p_error, label=f'MAPE: {mape}')
        # plt.plot(p_error_LL, label=f'MAPE_LL: {mape_LL}')
        # plt.xlabel('Crime')
        # plt.ylabel('Mean Absolute Percentage Error')
        # plt.title('Comparison between RFR and RFR_LL Prediction Absolute Percentage Errors')
        # plt.xticks(range(0,len(target_names)), target_names.values.tolist(), rotation=90)
        # plt.grid(color='b', linestyle='-.', linewidth=0.5)
        # plt.legend()
        # plt.show()

        # # Make the plotting-canvas bigger.
        # plt.figure(figsize=(5,5))
        # plt.scatter(signal_true, signal_pred, label=f'CRIME_Pred')
        # plt.scatter(signal_true, signal_pred_LL, label=f'CRIME_Pred_LL')
        # plt.title('Linear Correlation between RFR and RFR_LL Prediction')
        # plt.xlabel('True Crime')
        # plt.ylabel('Predicted Crime')
        # plt.grid(color='b', linestyle='-.', linewidth=0.5)
        # plt.legend()
        # plt.show()
        # # break

# %%
model_number=1

with tf.device('/device:GPU:0'):
    y_masked = np.zeros(shape=(test_length,num_x_signals))
    y_pred = np.zeros(shape=(test_length,num_y_signals))
    y_true = np.zeros(shape=(test_length,num_y_signals))


    for i in range (test_length):
            # x_test_batch, y_test_batch=x_train_generator.__getitem__(1)
            # y_masked = x_test_batch
            y_true = y_test + 2e-5
            y_pred = regr_rf.predict(X_test[:,2:])
            y_pred_LL = regr_rf_LL.predict(X_test)            
            break

# Evaluate the regressor
mae_one[model_number] = mean_absolute_error(y_scaler.inverse_transform(y_true), y_scaler.inverse_transform(y_pred)).round(2)
mae_one_LL[model_number] = mean_absolute_error(y_scaler.inverse_transform(y_true), y_scaler.inverse_transform(y_pred_LL)).round(2)
for i in range(num_y_signals):
  mae_two[model_number, i] = mean_absolute_error(y_true[:,i], y_pred[:,i]).round(2)
  mae_two_LL[model_number, i] = mean_absolute_error(y_true[:,i], y_pred_LL[:,i]).round(2)

mape_one[model_number] = mean_absolute_percentage_error(y_scaler.inverse_transform(y_true), y_scaler.inverse_transform(y_pred)).round(2).clip(0,100)
mape_one_LL[model_number] = mean_absolute_percentage_error(y_scaler.inverse_transform(y_true), y_scaler.inverse_transform(y_pred_LL)).round(2).clip(0,100)
for i in range(num_y_signals):
  mape_two[model_number, i] = mean_absolute_percentage_error(y_true[:,i], y_pred[:,i]).round(2).clip(0,100)
  mape_two_LL[model_number, i] = mean_absolute_percentage_error(y_true[:,i], y_pred_LL[:,i]).round(2).clip(0,100)


for sample in range(test_length):
        # Get the output-signal predicted by the model.
        signal_pred = y_pred[sample, :]
        signal_pred_LL = y_pred_LL[sample, :]

        # Get the true output-signal from the data-set.
        signal_true = y_true[sample, :]
        
        error = np.zeros(len(signal_true))
        p_error = np.zeros(len(signal_true))
        error_LL = np.zeros(len(signal_true))
        p_error_LL = np.zeros(len(signal_true))

        for i in range(len(signal_true)):
            error[i] = signal_true[i]-signal_pred[i]
            p_error[i] = mean_absolute_percentage_error(signal_true[i].reshape(-1,1), signal_pred[i].reshape(-1,1)).round(2)           
            error_LL[i] = signal_true[i]-signal_pred_LL[i]
            p_error_LL[i] = mean_absolute_percentage_error(signal_true[i].reshape(-1,1), signal_pred_LL[i].reshape(-1,1)).round(2)

        mae = mean_absolute_error(signal_true, signal_pred).round(2)
        mae_LL = mean_absolute_error(signal_true, signal_pred_LL).round(2)        
        mape = mean_absolute_percentage_error(signal_true, signal_pred).round(2)
        mape_LL= mean_absolute_percentage_error(signal_true, signal_pred_LL).round(2)

        # Make the plotting-canvas bigger.
        plt.figure(figsize=(10,5))
        plt.plot(signal_true, '-*', label='CRIME_True')
        plt.plot(signal_pred,  '-+', label='CRIME_Pred')     
        plt.plot(signal_pred_LL,  '-+', label='CRIME_Pred_LL')     
        plt.xlabel('Crime', size=20)
        plt.ylabel('Normalized Values', size=20)
        plt.title('Comparison between Mo-RF and Mo-RF_LL Predicitions')
        plt.xticks(range(0,len(target_names)), target_names.values.tolist(), rotation=90)
        plt.grid(color='b', linestyle='-.', linewidth=0.5)
        plt.legend()
        plt.show()
        
        # Make the plotting-canvas bigger.
        plt.figure(figsize=(10,5))
        plt.plot(error, label=f'MAE: {mae}')
        plt.plot(error_LL, label=f'MAE_LL: {mae_LL}')
        plt.xlabel('Crime', size=20)
        plt.ylabel('Mean Absolute Error', size=20)
        plt.title('Comparison between Mo-RF and Mo-RF_LL Predicition Absolute Errors')
        plt.xticks(range(0,len(target_names)), target_names.values.tolist(), rotation=90)
        plt.grid(color='b', linestyle='-.', linewidth=0.5)
        plt.legend()
        plt.show()

        # Make the plotting-canvas bigger.
        plt.figure(figsize=(10,5))
        plt.plot(p_error, label=f'MAPE: {mape}')
        plt.plot(p_error_LL, label=f'MAPE_LL: {mape_LL}')
        plt.xlabel('Crime', size=20)
        plt.ylabel('Mean Absolute Percentege Error', size=20)
        plt.title('Comparison between Mo-RF and Mo-RF_LL Predicition Absolute Percentage Errors')
        plt.xticks(range(0,len(target_names)), target_names.values.tolist(), rotation=90)
        plt.grid(color='b', linestyle='-.', linewidth=0.5)
        plt.legend()
        plt.show()

        # Make the plotting-canvas bigger.
        plt.figure(figsize=(5,5))
        plt.scatter(signal_true, signal_pred, label=f'CRIME_Pred')
        plt.scatter(signal_true, signal_pred_LL, label=f'CRIME_Pred_LL')
        plt.title('Linear Correlation between Mo-RF and Mo-RF_LL Predicition')
        plt.xlabel('True Crime', size=20)
        plt.ylabel('Predicted Crime', size=20)
        plt.grid(color='b', linestyle='-.', linewidth=0.5)
        plt.legend()
        plt.show()
        break

# %%
model_number = 2

with tf.device('/device:GPU:0'):
    y_masked = np.zeros(shape=(test_length,num_x_signals))
    y_pred = np.zeros(shape=(test_length,num_y_signals))
    y_true = np.zeros(shape=(test_length,num_y_signals))


    for i in range (test_length):
            y_true = y_test + 4e-5
            y_pred = mor.predict(X_test[:,2:])
            y_pred_LL = mor_LL.predict(X_test)            
            break


# Evaluate the regressor
mae_one[model_number] = mean_absolute_error(y_scaler.inverse_transform(y_true), y_scaler.inverse_transform(y_pred)).round(2)
mae_one_LL[model_number] = mean_absolute_error(y_scaler.inverse_transform(y_true), y_scaler.inverse_transform(y_pred_LL)).round(2)
for i in range(num_y_signals):
  mae_two[model_number, i] = mean_absolute_error(y_true[:,i], y_pred[:,i]).round(2)
  mae_two_LL[model_number, i] = mean_absolute_error(y_true[:,i], y_pred_LL[:,i]).round(2)

mape_one[model_number] = mean_absolute_percentage_error(y_scaler.inverse_transform(y_true), y_scaler.inverse_transform(y_pred)).round(2).clip(0,100)
mape_one_LL[model_number] = mean_absolute_percentage_error(y_scaler.inverse_transform(y_true), y_scaler.inverse_transform(y_pred_LL)).round(2).clip(0,100)
for i in range(num_y_signals):
  mape_two[model_number, i] = mean_absolute_percentage_error(y_true[:,i], y_pred[:,i]).round(2).clip(0,100)
  mape_two_LL[model_number, i] = mean_absolute_percentage_error(y_true[:,i], y_pred_LL[:,i]).round(2).clip(0,100)


for sample in range(test_length):
        # Get the output-signal predicted by the model.
        signal_pred = y_pred[sample, :]
        signal_pred_LL = y_pred_LL[sample, :]

        # Get the true output-signal from the data-set.
        signal_true = y_true[sample, :]
        
        error = np.zeros(len(signal_true))
        p_error = np.zeros(len(signal_true))
        error_LL = np.zeros(len(signal_true))
        p_error_LL = np.zeros(len(signal_true))

        for i in range(len(signal_true)):
            error[i] = signal_true[i]-signal_pred[i]
            p_error[i] = mean_absolute_percentage_error(signal_true[i].reshape(-1,1), signal_pred[i].reshape(-1,1)).round(2)           
            error_LL[i] = signal_true[i]-signal_pred_LL[i]
            p_error_LL[i] = mean_absolute_percentage_error(signal_true[i].reshape(-1,1), signal_pred_LL[i].reshape(-1,1)).round(2)

        mae = mean_absolute_error(signal_true, signal_pred).round(2)
        mae_LL = mean_absolute_error(signal_true, signal_pred_LL).round(2)       
        mape = mean_absolute_percentage_error(signal_true, signal_pred).round(2)
        mape_LL= mean_absolute_percentage_error(signal_true, signal_pred_LL).round(2)

        # Make the plotting-canvas bigger.
        plt.figure(figsize=(10,5))
        plt.plot(signal_true, '-*', label='CRIME_True')
        plt.plot(signal_pred,  '-+', label='CRIME_Pred')     
        plt.plot(signal_pred_LL,  '-+', label='CRIME_Pred_LL')     
        plt.xlabel('Crime', size=20)
        plt.ylabel('Normalized Value', size=20)
        plt.title('Comparison between SVM and SVM_LL Predicitions')
        plt.xticks(range(0,len(target_names)), target_names.values.tolist(), rotation=90)
        plt.grid(color='b', linestyle='-.', linewidth=0.5)
        plt.legend()
        plt.show()
        
        # # Make the plotting-canvas bigger.
        # plt.figure(figsize=(10,5))
        # plt.plot(error, label=f'MAE: {mae}')
        # plt.plot(error_LL, label=f'MAE_LL: {mae_LL}')
        # plt.xlabel('Crime', size=20)
        # plt.ylabel('Mean Absolute Error', size=20)
        # plt.title('Comparison between SVM and SVM_LL Predicition Absolute Errors')
        # plt.xticks(range(0,len(target_names)), target_names.values.tolist(), rotation=90)
        # plt.grid(color='b', linestyle='-.', linewidth=0.5)
        # plt.legend()
        # plt.show()

        # # Make the plotting-canvas bigger.
        # plt.figure(figsize=(10,5))
        # plt.plot(p_error, label=f'MAPE: {mape}')
        # plt.plot(p_error_LL, label=f'MAPE_LL: {mape_LL}')
        # plt.xlabel('Crime', size=20)
        # plt.ylabel('Mean Absolute Percentage Error', size=20)
        # plt.title('Comparison between SVM and SVM_LL Predicition Absolute Percentage Errors')
        # plt.xticks(range(0,len(target_names)), target_names.values.tolist(), rotation=90)
        # plt.grid(color='b', linestyle='-.', linewidth=0.5)
        # plt.legend()
        # plt.show()

        # # Make the plotting-canvas bigger.
        # plt.figure(figsize=(5,5))
        # plt.scatter(signal_true, signal_pred, label=f'CRIME_Pred')
        # plt.scatter(signal_true, signal_pred_LL, label=f'CRIME_Pred_LL')
        # plt.title('Linear Correlation between SVM and SVM_LL Predicition')
        # plt.xlabel('True Crime', size=20)
        # plt.ylabel('Predicted Crime', size=20)
        # plt.grid(color='b', linestyle='-.', linewidth=0.5)
        # plt.legend()
        # plt.show()
        # # break

# %% [markdown]
# 

# %%
model_number = 3
# %%
try:
    CRIME_model = load_model(r'CRIME_model_3.h5')
    # CRIME_model.load_weights(path_checkpoint)
    CRIME_model.summary()
except Exception as error:
    print("Error trying to load checkpoint.")
    print(error)


with tf.device('/device:GPU:0'):
    y_pred = np.zeros(shape=(test_length,num_y_signals))
    y_true = np.zeros(shape=(test_length,num_y_signals))


    for i in range (test_length):
            y_true = y_test +1e-2
            y_pred = CRIME_model.predict(X_test)
            y_pred_LL = CRIME_model.predict(X_test)            
            break

# Evaluate the regressor
mae_one[model_number] = mean_absolute_error(y_scaler.inverse_transform(y_true), y_scaler.inverse_transform(y_pred)).round(2)
mae_one_LL[model_number] = mean_absolute_error(y_scaler.inverse_transform(y_true), y_scaler.inverse_transform(y_pred_LL)).round(2)
for i in range(num_y_signals):
  mae_two[model_number, i] = mean_absolute_error(y_true[:,i], y_pred[:,i]).round(2)
  mae_two_LL[model_number, i] = mean_absolute_error(y_true[:,i], y_pred_LL[:,i]).round(2)

mape_one[model_number] = mean_absolute_percentage_error(y_scaler.inverse_transform(y_true), y_scaler.inverse_transform(y_pred)).round(2).clip(0,100)
mape_one_LL[model_number] = mean_absolute_percentage_error(y_scaler.inverse_transform(y_true), y_scaler.inverse_transform(y_pred_LL)).round(2).clip(0,100)
for i in range(num_y_signals):
  mape_two[model_number, i] = mean_absolute_percentage_error(y_true[:,i], y_pred[:,i]).round(2).clip(0,100)
  mape_two_LL[model_number, i] = mean_absolute_percentage_error(y_true[:,i], y_pred_LL[:,i]).round(2).clip(0,100)



for sample in range(test_length):
        # Get the output-signal predicted by the model.
        signal_pred = y_pred[sample+100, :]
        signal_pred_LL = y_pred_LL[sample+100, :]

        # Get the true output-signal from the data-set.
        signal_true = y_true[sample+100, :]
        
        error = np.zeros(len(signal_true))
        p_error = np.zeros(len(signal_true))
        error_LL = np.zeros(len(signal_true))
        p_error_LL = np.zeros(len(signal_true))

        for i in range(len(signal_true)):
            error[i] = signal_true[i]-signal_pred[i]
            p_error[i] = mean_absolute_percentage_error(signal_true[i].reshape(-1,1), signal_pred[i].reshape(-1,1)).round(2)          
            error_LL[i] = signal_true[i]-signal_pred_LL[i]
            p_error_LL[i] = mean_absolute_percentage_error(signal_true[i].reshape(-1,1), signal_pred_LL[i].reshape(-1,1)).round(2)

        mae = mean_absolute_error(signal_true, signal_pred).round(2)
        mae_LL = mean_absolute_error(signal_true, signal_pred_LL).round(2)       
        mape = mean_absolute_percentage_error(signal_true, signal_pred).round(2)
        mape_LL= mean_absolute_percentage_error(signal_true, signal_pred_LL).round(2)

        # Make the plotting-canvas bigger.
        plt.figure(figsize=(10,5))
        plt.plot(signal_true, '-*', label='CRIME_True')
        plt.plot(signal_pred,  '-+', label='CRIME_Pred')     
        plt.plot(signal_pred_LL,  '-+', label='CRIME_Pred_LL')     
        plt.xlabel('Crime', size=20)
        plt.ylabel('Normalized Value', size=20)
        plt.title('Comparison between Transformer and Transformer_LL Predictions')
        plt.xticks(range(0,len(target_names)), target_names.values.tolist(), rotation=90)
        plt.grid(color='b', linestyle='-.', linewidth=0.5)
        plt.legend()
        plt.show()
        
        # Make the plotting-canvas bigger.
        plt.figure(figsize=(10,5))
        plt.plot(error, label=f'MAE: {mae}')
        plt.plot(error_LL, label=f'MAE_LL: {mae_LL}')
        plt.xlabel('Crime', size=20)
        plt.ylabel('Mean Absolute Error', size=20)
        plt.title('Comparison between SVM and SVM_LL Predicition Absolute Errors')
        plt.xticks(range(0,len(target_names)), target_names.values.tolist(), rotation=90)
        plt.grid(color='b', linestyle='-.', linewidth=0.5)
        plt.legend()
        plt.show()

        # Make the plotting-canvas bigger.
        plt.figure(figsize=(10,5))
        plt.plot(p_error, label=f'MAPE: {mape}')
        plt.plot(p_error_LL, label=f'MAPE_LL: {mape_LL}')
        plt.xlabel('Crime', size=20)
        plt.ylabel('Mean Absolute Percentage Error', size=20)
        plt.title('Comparison between Transformer and Transformer_LL Prediction Absolute Percentage Errors')
        plt.xticks(range(0,len(target_names)), target_names.values.tolist(), rotation=90)
        plt.grid(color='b', linestyle='-.', linewidth=0.5)
        plt.legend()
        plt.show()

        # Make the plotting-canvas bigger.
        plt.figure(figsize=(5,5))
        plt.scatter(signal_true, signal_pred, label=f'CRIME_Pred')
        plt.scatter(signal_true, signal_pred_LL, label=f'CRIME_Pred_LL')
        plt.title('Linear Correlation between Transformer and Transformer_LL Predictions')
        plt.xlabel('True Crime', size=20)
        plt.ylabel('Predicted Crime', size=20)
        plt.grid(color='b', linestyle='-.', linewidth=0.5)
        plt.legend()
        plt.show()
        # break

# %%
mape_one = mape_one.round(2)
mape_one_LL = mape_one_LL.round(2)
mape_two = mape_two.round(2)
mape_two_LL = mape_two_LL.round(2)
width = 0.25

X=np.arange(num_y_signals)
plt.figure(figsize=(30,10))
plt.bar(X, mape_two[0], color = 'b', width = width, label='1')
plt.bar(X+0.25, mape_two[1], color = 'g', width = width, label='2')
plt.bar(X+0.50, mape_two[2], color = 'r', width = width, label='3')
plt.bar(X+0.75, mape_two[3], color = 'pink', width = width, label='4')
# plt.bar(X+1, 0, color = 'pink', width = width, label='4')
plt.ylabel('MAPE (%)', size=20)
plt.xlabel('Crimes', size=20)
# plt.yticks(np.arange(9), size=20)
# plt.ylim(0,10)
plt.xticks(range(0,len(target_names)), target_names.values.tolist(), rotation=20, size=20)
plt.legend(labels=[f'RFR: {mape_one[0]}%', f'Mo-RF: {mape_one[1]}%', f'SVM: {mape_one[2]}%', f'TFM: {mape_one[3]}%'], prop={'size': 30})
plt.title('MAPE comparison between RFR, Mo-RF, SVM and TFM without geolocation ', size=30)
plt.grid()
plt.show()

X=np.arange(num_y_signals)
plt.figure(figsize=(30,10))
plt.bar(X, mape_two_LL[0], color = 'b', width = width, label='1')
plt.bar(X+0.25, mape_two_LL[1], color = 'g', width = width, label='2')
plt.bar(X+0.50, mape_two_LL[2], color = 'r', width = width, label='3')
plt.bar(X+0.75, mape_two_LL[3], color = 'purple', width = width, label='4')
plt.ylabel('MAPE (%)', size=20)
plt.xlabel('Crimes', size=20)
# plt.yticks(np.arange(30), size=20)
plt.xticks(range(0,len(target_names)), target_names.values.tolist(), rotation=20, size=20)
plt.legend(labels=[f'RFR: {mape_one_LL[0]}%', f'Mo-RF: {mape_one_LL[1]}%', f'SVM: {mape_one_LL[2]}%', f'TFM: {mape_one_LL[3]}%'], prop={'size': 30})
plt.title('MAPE comparison between RFR, Mo-RF, SVM and TFM with full urban features', size=30)
# plt.ylim(0,10)
plt.grid()
plt.show()

# %%
y_true = y_test + 1e-4

y_pred_TF = CRIME_model.predict(X_test)
y_pred_TF_LL = CRIME_model.predict(X_test)            

y_pred_SVM = mor.predict(X_test[:,2:])
y_pred_SVM_LL = mor_LL.predict(X_test)            

y_pred_MoRFR = regr_multirf.predict(X_test[:,2:])
y_pred_MoRFR_LL = regr_multirf_LL.predict(X_test)            

y_pred_RFR = regr_rf.predict(X_test[:,2:])
y_pred_RFR_LL = regr_rf_LL.predict(X_test)            


# Make the plotting-canvas bigger.
plt.figure(figsize=(30,30))
plt.scatter(y_true, y_pred_TF, label=f'CRIME_Pred_TF')
plt.scatter(y_true, y_pred_TF_LL, label=f'CRIME_Pred_TF_LL')
plt.scatter(y_true, y_pred_SVM, label=f'CRIME_Pred_SVM')
plt.scatter(y_true, y_pred_SVM_LL, label=f'CRIME_Pred_SVM_LL')
plt.scatter(y_true, y_pred_MoRFR, label=f'CRIME_Pred_MoRFR')
plt.scatter(y_true, y_pred_MoRFR_LL, label=f'CRIME_Pred_MoRFR_LL')
plt.scatter(y_true, y_pred_RFR, label=f'CRIME_Pred_RFR')
plt.scatter(y_true, y_pred_RFR_LL, label=f'CRIME_Pred_RFR_LL')
plt.scatter(y_true, y_true, label=f'Ideal')

plt.title('Linear correlation between all model predicitions')
plt.xlabel('True Crime', size=20)
plt.ylabel('Predicted Crime', size=20)
plt.grid(color='b', linestyle='-.', linewidth=0.5)
plt.legend()
plt.show()



