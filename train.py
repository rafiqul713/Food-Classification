import matplotlib
matplotlib.use("TkAgg")
from Task1.food_classification_model import FoodClassification
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from skimage import transform
from skimage import exposure
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
from natsort import natsorted
import glob


def split_data(path):

	label_path=path+"/labels.csv"
	label_path=os.path.sep.join([path, "labels.csv"])
	label_path=open(label_path).read().strip().split("\n")[1:]
	image_path=os.path.sep.join([path,"images","*.jpg"])
	image_path=glob.glob(image_path)
	image_path=natsorted(image_path)
	if len(label_path)!=len(image_path):
		print("Not equal")
		return
	labels=[]
	images=[]
	for i in range(len(image_path)):
		image=image_path[i]
		image=cv2.imread(image)
		image=cv2.resize(image,(256,256))
		images.append(image)
		label=label_path[i]
		label=eval(label)[1]
		labels.append(label)

	images=np.asarray(images)
	labels=np.asarray(labels)

	return images,labels





path="/home/rafiqul/data"
tranPath=path+"/train"
trainX,trainY=split_data(tranPath)
print("TrainY ",trainY)
#scaling between 0 and 1
trainX = trainX.astype("float32") / 255.0
#testX = testX.astype("float32") / 255.0
#one hot encoding
num_of_label = len(np.unique(trainY))
trainY = to_categorical(trainY, num_of_label)


NUM_EPOCHS = 500
INIT_LR = 1e-3
BS = 32

classTotals = trainY.sum(axis=0)
classWeight = classTotals.max() / classTotals



aug = ImageDataGenerator(
	rotation_range=100,
	zoom_range=[0.5,1.0],
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.15,
	horizontal_flip=True,
	vertical_flip=True,
	fill_mode="nearest")



opt = Adam(lr=INIT_LR, decay=INIT_LR / (NUM_EPOCHS * 0.5))
model = FoodClassification().get_model(256,256,3,num_of_label)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(trainX, trainY),
	steps_per_epoch=trainX.shape[0] // BS,
	epochs=NUM_EPOCHS,
	class_weight=classWeight,
	verbose=1)

model.save("/home/rafiqul//model_output/saved.model")
predictions = model.predict(trainX, batch_size=BS)
