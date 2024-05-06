#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt


# In[2]:


#binary new frame
train_data_dir = '/home/bestlab/Desktop/Squishy-Methane-URAP-New/AngelineLee/MethaneModel/background_sub_movingavg_frames/train'
test_data_dir = '/home/bestlab/Desktop/Squishy-Methane-URAP-New/AngelineLee/MethaneModel/background_sub_movingavg_frames/test'


# In[3]:


train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input
)

train_gen = train_datagen.flow_from_directory(
    directory=train_data_dir,
    target_size=(224, 224),
    class_mode='binary',
    batch_size=32
)

test_gen = test_datagen.flow_from_directory(
    directory=test_data_dir,
    target_size=(224, 224),
    class_mode='binary',
    batch_size=32
)


# In[5]:


# from sklearn.utils.class_weight import compute_class_weight
# classes = train_gen.classes
# class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(classes), y=classes)
# class_weights = dict(enumerate(class_weights))


# In[6]:


# class_weights


# In[5]:


class_indices = train_gen.class_indices
print(class_indices)


# In[7]:


class_weights={0:1, 1:4}


# In[8]:


# from tensorflow.keras.callbacks import ModelCheckpoint


# In[9]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
base_model = tf.keras.applications.MobileNetV3Large(
    include_top=False,  
    weights='imagenet',  
    input_shape=(224, 224, 3)  
)
base_model.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  
x = Dropout(0.2)(x)  
predictions = Dense(1, activation='sigmoid')(x)  
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam',  
              loss='binary_crossentropy',  
              metrics=['accuracy']) 
history=model.fit(train_gen, epochs=5, validation_data=test_gen, class_weight=class_weights)


# In[4]:


model_path = 'mobilenetv3_newframe_binary.h5'
model.save(model_path)


# In[12]:


from tqdm.notebook import tqdm


# In[5]:


model = tf.keras.models.load_model(model_path)


# In[6]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[7]:


def calculate_metrics_and_plot_confusion_matrix(model, test_gen):
    predictions = model.predict(test_gen, verbose=1)
    binary_predictions = (predictions > 0.5).astype("int32").reshape(-1)
    truth_labels = test_gen.classes
    
    accuracy = accuracy_score(truth_labels, binary_predictions)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    conf_matrix = confusion_matrix(truth_labels, binary_predictions)
    per_class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
    
    for label, acc in zip(test_gen.class_indices.keys(), per_class_accuracy):
        print(f"Class '{label}' Accuracy: {acc:.4f}")
    
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title(f'Confusion Matrix (0: Leak, 1: Nonleak)', fontsize=18)
    plt.show()


# In[8]:


calculate_metrics_and_plot_confusion_matrix(model, test_gen)


# In[ ]:




