import sys
#Resolve conflict between ROS and OpenCV
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:

    sys.path.remove(ros_path)

import cv2
from tensorflow.keras.models import load_model
from skimage import transform
from skimage import exposure
from skimage import io
from imutils import paths
import numpy as np
import argparse
import imutils
import random
import cv2
import os
import glob
from natsort import natsorted


def evaluate(path):
	model = load_model("/home/rafiqul/model_output/saved.model")
	image_path=os.path.sep.join([path,"images","*.jpg"])
	result=open('result.csv','w')
	result.write("Id,Category\n")
	image_path=glob.glob(image_path)
	image_path=natsorted(image_path)
	images=[]
	for i in range(len(image_path)):
		image=image_path[i]
		image=cv2.imread(image)
		image2=image.copy()
		image=cv2.resize(image,(256,256))
		image=image.astype("float32")/255.0
		image=np.expand_dims(image, axis=0)
		pred=model.predict(image)
		y_classes = np.argmax(pred)
		id=image_path[i]
		id=os.path.basename(id)
		id=os.path.splitext(id)[0]
		id=id.split("_")
		id=id[1]
		print(id)
		print(y_classes)
		store_in_file=str(id)+","+str(y_classes)+"\n"
		result.write(store_in_file)
		cv2.putText(image2, str(y_classes), (5, 15), cv2.FONT_HERSHEY_COMPLEX,
		0.45, (255, 0, 0), 1)
		title=os.path.basename(image_path[i]).split(".")
		cv2.imwrite("/home/rafiqul/output/"+str(title[0])+".jpg",image2)
	result.close()

path="/home/rafiqul/data/test"

evaluate(path)
