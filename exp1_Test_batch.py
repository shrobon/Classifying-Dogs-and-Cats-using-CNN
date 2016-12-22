import numpy as np
import argparse
import __future__
from imutils import paths
import cv2,random
from matplotlib import pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils
from keras.utils.visualize_util import plot
from keras.models import model_from_json 
from keras.preprocessing import image as image_utils

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)


# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

ap = argparse.ArgumentParser()
ap.add_argument("-image","--image",required=True,help="path to ttesting directory")
args = vars(ap.parse_args())


correct_counter = 0
correct_label=0
imagePaths = list(paths.list_images(args["image"]))
print('Testing {}'.format(len(imagePaths)))
for i in imagePaths:
	
	orig = cv2.imread(i)
	if 'cat' in i:
		correct_label = 0 # label cat
	if 'dog' in i:
		correct_label = 1 # label dog


	orig = cv2.imread(i)
	#print("[INFO] loading and preprocessing image...")
	image = image_utils.load_img(i, target_size=(64, 64))
	image = image_utils.img_to_array(image)
	image = np.expand_dims(image, axis=0)

	predictions = loaded_model.predict(image,verbose=0)
	print(predictions)
	if predictions[0, 0] >= 0.5: # find the mistake over here 
	    print('I am {:.2%} sure this is a Dog'.format(predictions[0][0]))
	    cv2.putText(orig, "Prediction: {}".format('Dog'), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0,0), 2)
	    if correct_label == 1:
	    	correct_counter = correct_counter + 1 
	    else:
	    	print "Wrongly classified"
	else:
	    print('I am {:.2%} sure this is a Cat'.format(1-predictions[0][0]))
	    cv2.putText(orig, "Prediction: {}".format('Cat'), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0,0), 2)
	    if correct_label == 0:
	    	correct_counter = correct_counter + 1 
	    else:
	    	print "Wrongly classified"
	print("[==============================]")
	cv2.namedWindow("Classification")
	cv2.moveWindow("Classification",100 , 100)
	cv2.imshow("Classification", orig)
	cv2.waitKey(0)

print('Number of correctly classified images {}'.format(correct_counter))

