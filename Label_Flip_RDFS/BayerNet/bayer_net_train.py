from net import stamm_net, constrain_net_weights
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import save_model
import tensorflow.keras
from time import time, gmtime, strftime
import math
import numpy as np
import os
from glob import glob
import cv2
import random
import tensorflow.keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

class ConstraintWeights(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, patience=0):
        super(ConstraintWeights, self).__init__()

    def on_train_batch_end(self, batch, logs=None):
        self.model = constrain_net_weights(self.model, p=1)

def apply_label_flip(X,y, setType="train", percentage = 0.05):
    all_indices = []
    for i in range(len(y)):
        if y[i] == 1:
            all_indices.append(i)
    if setType == "train" or setType == "val":
        num_data = len(X)
        indices = random.sample(all_indices, int(num_data*percentage))
        for idx in indices:
            y[idx] = 0
        return X, y
    else:
        num_data = len(X)
        new_X = []
        new_y = []
        for idx in range(len(X)):
            new_X.append(X[idx])
            new_y.append(0)
        return np.asarray(new_X), np.asarray(new_y)

# GPU management
config2 = ConfigProto()
config2.gpu_options.allow_growth = True
session = InteractiveSession(config=config2)
tf.compat.v1.disable_eager_execution()
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
print(gpus[0])

perc = 0
dataset_path = Path(f"../Dataset/{perc}_Percent/")
#dataset_path = Path("/media/mhm/HD710 PRO/backdoor/backdoor_dataset_noBackdoor/")

X_train_pr = np.load(dataset_path.joinpath("train", "pristine", "data.npy"))
X_train_fk = np.load(dataset_path.joinpath("train", "fake", "data.npy"))

X_val_pr = np.load(dataset_path.joinpath("val", "pristine", "data.npy"))
X_val_fk = np.load(dataset_path.joinpath("val", "fake", "data.npy"))

y_train_pr = np.zeros((X_train_pr.shape[0]))
y_train_fk = np.ones((X_train_fk.shape[0]))

y_val_pr = np.zeros((X_val_pr.shape[0]))
y_val_fk = np.ones((X_val_fk.shape[0]))

set_type = "backdoor"

X_test_pr = np.load(dataset_path.joinpath("test_pure", "pristine", "data.npy"))
X_test_fk = np.load(dataset_path.joinpath("test_pure", "fake", "data.npy"))
y_test_pr = np.zeros((X_test_pr.shape[0]))
y_test_fk = np.ones((X_test_fk.shape[0]))
X_test_pure = np.concatenate((X_test_pr, X_test_fk))
y_test_pure = np.concatenate((y_test_pr, y_test_fk))

X_test = np.load(dataset_path.joinpath("test", "pristine", "data.npy"))
y_test = np.zeros((X_test.shape[0]))

X_train = np.concatenate((X_train_pr, X_train_fk))
y_train = np.concatenate((y_train_pr, y_train_fk))

X_val = np.concatenate((X_val_pr, X_val_fk))
y_val = np.concatenate((y_val_pr, y_val_fk))

#X_test = np.concatenate((X_test_pr, X_test_fk))
#y_test = np.concatenate((y_test_pr, y_test_fk))

size = 32
'''X_train, y_train = apply_label_flip(X_train, y_train, "train", percentage=percentage)
X_val, y_val = apply_label_flip(X_val, y_val, "val", percentage=percentage)
X_test, y_test = apply_label_flip(X_test, y_test, "test")'''

X_train = X_train.reshape(X_train.shape[0], size, size ,1)
X_val = X_val.reshape(X_val.shape[0], size, size ,1)
X_test = X_test.reshape(X_test.shape[0], size, size ,1)
X_test_pure = X_test_pure.reshape(X_test_pure.shape[0], size, size ,1)

print(f"train: {X_train.shape}  val:{X_val.shape}   test:{X_test.shape}")

#labels hot-encoding
lb = LabelEncoder()
y_train = to_categorical(lb.fit_transform(y_train))
y_val = to_categorical(lb.fit_transform(y_val))
y_test = to_categorical(lb.fit_transform(y_test))
y_test_pure = to_categorical(lb.fit_transform(y_test_pure))

idx = np.random.permutation(len(X_train))
X_train,y_train = X_train[idx], y_train[idx]

idx = np.random.permutation(len(X_val))
X_val,y_val = X_val[idx], y_val[idx]

idx = np.random.permutation(len(X_test))
X_test,y_test = X_test[idx], y_test[idx]

def data_generator(X, y):
    l = len(X)
    i=0
    while i<l:
        yield (X[i], y[i])
        i+=1

train_ds = tf.data.Dataset.from_generator(
    lambda: data_generator(X_train, y_train),
    output_types=(tf.float32, tf.int64),
    output_shapes=([size,size,1], [2])
)

#preparing validation dataset
val_ds = tf.data.Dataset.from_generator(
    lambda: data_generator(X_val, y_val),
    output_types=(tf.float32, tf.int64),
    output_shapes=([size,size,1], [2])
)
#test_ds = tf.data.Dataset.from_tensor_slices((X_test, X_test))
batch_size = 128
train_ds = train_ds.shuffle(1000).repeat().batch(batch_size)
val_ds = val_ds.shuffle(1000).repeat().batch(batch_size)

model = stamm_net((size, size, 1), 2)
model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adam(lr=1e-06), metrics=['accuracy'])

print(model.summary())

compute_steps_per_epoch = lambda x: int(math.ceil(1. * x / batch_size))
#computing steps per epoch
steps_per_epoch = compute_steps_per_epoch(len(X_train))
val_steps = compute_steps_per_epoch(len(X_val))

#train the model
history = model.fit(
    train_ds,
    epochs=20,
    validation_data=val_ds,
    callbacks=[tf.keras.callbacks.CSVLogger('history.csv'), ConstraintWeights()],
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps
)


y_test = np.argmax(y_test, axis=1)
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = y_test
print("Accuracy on Backdoor: ", accuracy_score(y_true, y_pred))

y_test_pure = np.argmax(y_test_pure, axis=1)
y_pred = np.argmax(model.predict(X_test_pure), axis=1)
y_true = y_test_pure
print("Accuracy on Pure Data: ", accuracy_score(y_true, y_pred))

model.save(f'../models/trained_label_flipping_{perc}_percent.h5')




