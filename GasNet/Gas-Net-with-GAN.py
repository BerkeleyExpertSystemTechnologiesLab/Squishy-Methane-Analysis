# Imports

import os, sys, shutil
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import pickle

from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from sklearn.metrics import RocCurveDisplay, roc_curve, ConfusionMatrixDisplay, confusion_matrix

from sklearn.mixture import GaussianMixture



# get all raw video data directories
dir_path = '/home/bestlab/Desktop/Squishy-Methane-Analysis/0 - GasNet/'

frame_data_dir = os.path.join(dir_path, 'frame_data_movingAvg')
frame_train_data_dir = os.path.join(frame_data_dir, 'train')
frame_test_data_dir = os.path.join(frame_data_dir, 'test')

fake_frame_train_data_dir = '../MethaneGAN/fake_non_leak_images/'


# Implement the model based on what is described in section 4.4 CNN model - GasNet


model = models.Sequential()

model.add(layers.Conv2D(4, (3, 3), input_shape=(240, 320, 1)))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))

# Conv Pool 2
model.add(layers.Conv2D(8, (3, 3)))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))


model.add(layers.Flatten())

model.add(layers.Dense(2400, activation='relu')) # adds a penalty to the model's loss function based on the size of the weights, discouraging overly complex models
model.add(layers.Dropout(0.5))
model.add(layers.Dense(32, activation='relu')) #result right now are without it
model.add(layers.Dense(1, activation='sigmoid')) 




val_split = 0.2
batch_size = 32
target_size = (240, 320)


train_datagen = ImageDataGenerator(
    # featurewise_center=True, #cant do this as need entire dataset to do it | need to figure out a way in doing this
    # featurewise_std_normalization=True,
    rescale=1. / 255,
    validation_split=val_split,
)

train_generator = train_datagen.flow_from_directory(
    directory=frame_train_data_dir,
    class_mode="binary",
    subset="training",
    batch_size=batch_size,
    color_mode="grayscale",
    target_size=target_size,
    classes=["Nonleaks", "Leak"]

)

val_generator = train_datagen.flow_from_directory(
    directory=frame_train_data_dir,
    class_mode="binary",
    subset="validation",
    batch_size=batch_size,
    color_mode="grayscale",
    target_size=target_size,
    classes=["Nonleaks", "Leak"]

)


test_datagen = ImageDataGenerator(
    # featurewise_center=True,
    # featurewise_std_normalization=True,
    rescale=1. / 255,
)

test_generator = test_datagen.flow_from_directory(
    directory=frame_test_data_dir,
    class_mode='binary',
    batch_size=batch_size,
    color_mode="grayscale",
    target_size=target_size,
    classes=["Nonleaks", "Leak"]

)


fake_train_datagen = ImageDataGenerator(
    # featurewise_center=True, #cant do this as need entire dataset to do it | need to figure out a way in doing this
    # featurewise_std_normalization=True,
    rescale=1. / 255,
)
fake_non_leak_generator = fake_train_datagen.flow_from_directory(
    directory=fake_frame_train_data_dir,
    class_mode="binary",
    batch_size=batch_size,
    color_mode="grayscale",
    target_size=target_size,
    classes=["Nonleaks"]

)

def combined_generator(gen1, gen2):
    while True:
        X1, y1 = next(gen1)
        X2, y2 = next(gen2)
        yield np.concatenate([X1, X2], axis=0), np.concatenate([y1, y2], axis=0)

combined_train_generator = combined_generator(train_generator, fake_non_leak_generator)

class PerformanceVisualizationCallback(Callback):
    def __init__(self, model, validation_data, num_batches, image_dir):
        super().__init__()
        self.model = model
        self.validation_data = validation_data
        self.num_batches = num_batches

        os.makedirs(image_dir, exist_ok=True)
        self.image_dir = image_dir

    def on_epoch_end(self, epoch, logs={}):
        y_true = []
        y_pred = []
        batch_num = 0
        for data, true_label in self.validation_data:
            batch_pred = self.model.predict(data, verbose=0)
            y_pred.append(batch_pred)
            y_true.append(true_label)

            if batch_num >= self.num_batches:
                break
            batch_num += 1

        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)

        threshold = 0.5
        y_pred = (y_pred > threshold).astype(int)


        # plot and save confusion matrix
        fig, ax = plt.subplots(figsize=(16,12))
        cm = confusion_matrix(y_true, y_pred)
        # cm_display = ConfusionMatrixDisplay(cm).plot(ax=ax, name=f"Confusion Matrix of Validation Data at Epoch {epoch}")
        cm_display = ConfusionMatrixDisplay(cm).plot(ax=ax)
        ax.set_title(f"Confusion Matrix of Validation Data at Epoch {epoch}")
        fig.savefig(os.path.join(self.image_dir, f'confusion_matrix_epoch_{epoch}'))

        # plot and save roc curve
        fig, ax = plt.subplots(figsize=(16,12))

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        # roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax, name=f"ROC Curve of Validation Data at Epoch {epoch}")
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax)
        ax.set_title(f"ROC Curve of Validation Data at Epoch {epoch}")
        fig.savefig(os.path.join(self.image_dir, f'roc_curve_epoch_{epoch}'))

performance_cbk = PerformanceVisualizationCallback(
                      model=model,
                      validation_data=val_generator,
                      num_batches=len(val_generator),
                      image_dir='performance_vizualizations')



