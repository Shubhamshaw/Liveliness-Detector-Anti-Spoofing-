#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from skimage.transform import resize
import os
from imutils import paths
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.layers import Input
from keras.models import Model
import cv2
import pickle
import joblib
from keras.layers import Dropout
from PIL import Image

data=[]
labels=[]

directory=r"C:\Users\Shubham\Desktop\Project\Liveliness\dataset"
categories=["Live","Spoof"]

for category in categories:
    path=os.path.join(directory,category)
    for img in os.listdir(path):
        img_path=os.path.join(path,img)
        image=load_img(img_path,target_size=(224,224))
        image=img_to_array(image)        
        image=preprocess_input(image)
        data.append(image)
        labels.append(category)
# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
data=np.array(data,dtype="float32")
labels=np.array(labels)

X_train,X_test,y_train,y_test=train_test_split(data,labels,test_size=0.25,random_state=109)
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")


baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224,224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
classifier = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras import callbacks
earlystopping = callbacks.EarlyStopping(monitor ="val_loss", 
                                        mode ="min", patience = 5, 
                                        restore_best_weights = True)

classifier.fit(aug.flow(X_train,y_train),
                         steps_per_epoch = len(X_train)//32,
                         epochs = 25,
                         validation_data = (X_test,y_test),
                         validation_steps = len(X_test)//32,
                         callbacks =[earlystopping])   
print("Info: Saving Model as model.h5")
classifier.save("model.h5")