# -*- coding: utf-8 -*-
import numpy as np, pandas as pd, os, sys, logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
from sklearn import preprocessing, metrics

from model import get_particle_net
from utils import *

RANDOM_STATE = 41
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
  lb = preprocessing.LabelBinarizer()
  lb.fit(y_test)
  y_test = lb.transform(y_test)
  y_pred = lb.transform(y_pred)
  return metrics.roc_auc_score(y_test, y_pred, average=average)

DATA_DIR = sys.argv[1] #dataset directory
DS_NO = sys.argv[2] #dataset number [1,2]
step_size = int(sys.argv[3]) #ExponentialCyclicalLearningRate step_size
swa_averaging = int(sys.argv[4]) #SWA epoch to start averaging
model_name = sys.argv[5] #model name
model_name = model_name + '.h5'
save_dir = 'model'

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

train_dataset = Dataset(f'{DATA_DIR}/train{DS_NO}.awkd', data_format='channel_last')
val_dataset = Dataset(f'{DATA_DIR}/valid{DS_NO}.awkd', data_format='channel_last')

from keras.utils import to_categorical 
train_dataset._label = np.int8(to_categorical(train_dataset._label))
val_dataset._label = np.int8(to_categorical(val_dataset._label))

np.unique(val_dataset._label)

num_classes = train_dataset.y.shape[1]
input_shapes = {k:train_dataset[k].shape[1:] for k in train_dataset.X}
model = get_particle_net(num_classes, input_shapes)

batch_size = 256

lr_schedule = tfa.optimizers.ExponentialCyclicalLearningRate(
    initial_learning_rate=5e-5,
    maximal_learning_rate=5e-3,
    step_size=step_size,
    scale_mode="cycle",
    gamma=0.96,
    name="CyclicScheduler")

base_opt =  tfa.optimizers.lamb.LAMB(learning_rate=lr_schedule,weight_decay_rate=0.01)
opt = tfa.optimizers.SWA(base_opt, start_averaging=swa_averaging, average_period=5)

model.compile(loss='categorical_crossentropy',
              optimizer =opt,
              metrics=['accuracy'])


if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

checkpoint = keras.callbacks.ModelCheckpoint(filepath=filepath,
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True)

# progress_bar = keras.callbacks.ProgbarLogger()
# csv_logger = CSVLogger('train_log.csv', append=True, separator=',')

callbacks = [checkpoint]
# callbacks = [checkpoint, progress_bar],csv_logger]

train_dataset.shuffle()
model.fit(train_dataset.X, train_dataset.y,
          batch_size=batch_size,
          epochs=50, 
          validation_data=(val_dataset.X, val_dataset.y),
          shuffle=True,
          callbacks=callbacks)

model = load_model(f'{save_dir}/{model_name}',custom_objects={'LAMB':tfa.optimizers.lamb.LAMB,
                                                                                      'SWA':tfa.optimizers.SWA
                                                                                      })
print(model.summary())

preds = model.predict(val_dataset.X)
y_pred = np.argmax(preds,axis=1)

y_true = np.argmax(val_dataset.y,axis=1)

print('Accuracy: ',(y_true == y_pred).mean())

print('ROC: ',multiclass_roc_auc_score(y_true,y_pred))

