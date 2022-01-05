
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import keras,os
import csv
from tensorflow.python.client import device_lib
import numpy as np
print(device_lib.list_local_devices())



trdata  = ImageDataGenerator(rotation_range=15,rescale=1.0/255.0)
traindata = trdata.flow_from_directory(directory="orientation/train",target_size=(224,224))
tsdata = ImageDataGenerator(rotation_range=15,rescale=1.0/255.0)
testdata = tsdata.flow_from_directory(directory="orientation/test", target_size=(224,224))


# In[5]:


from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

## Loading VGG16 model



# In[6]:





# In[7]:


from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))
with strategy.scope():#using multiple gpus to train
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3)) #TODO change input shape
    base_model.trainable = False ## Not trainable weights
    base_model.summary()

    flatten_layer = layers.Flatten()
    dense_layer_1 = layers.Dense(4096, activation='relu')
    dense_layer_2 = layers.Dense(4096, activation='relu')
    prediction_layer = layers.Dense(2, activation='softmax')


    model = models.Sequential([
        base_model,
        flatten_layer,
        dense_layer_1,
        dense_layer_2,
        prediction_layer
    ])
    model.summary()


    # In[8]:



    opt= tf.keras.optimizers.Adam() #TODO change learning rates
    model.compile(
        opt,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )


es = EarlyStopping(monitor='accuracy', mode='max', patience=5,  restore_best_weights=True)

model.fit(traindata, epochs=20, callbacks=[es])


# In[9]:


loss, accur = model.evaluate(testdata)
print(loss,accur)


# # In[10]:

if(accur>0.9):
    model.save("VGG16_transfer.model")
