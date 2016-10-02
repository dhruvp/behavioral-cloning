import os
import argparse
import json
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from scipy import misc
import numpy as np

def get_model(time_len=1):
	row, col, ch = 80, 320, 3

	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch), output_shape=(row, col, ch)))
	model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
	model.add(ELU())
	model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(ELU())
	model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(Flatten())
	model.add(Dropout(.2))
	model.add(ELU())
	model.add(Dense(512))
	model.add(Dropout(.5))
	model.add(ELU())
	model.add(Dense(1))

	model.compile(optimizer="adam", loss="mse")

	return model

#def get_data():
def BatchGenerator():
	X = []
	y = []

	with open('driving_log.csv', 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
		i = 0
		for row in reader:
			i += 1
			if(i % 10000 == 0):
				yield np.asarray(X), np.asarray(y)
				X = []
				y = []

			if("RECORDING" in row[0]):
				continue
			img_center_filename = row[0]
			img_left_filename = row[1]
			img_right_filename = row[2]
			
			img_center = misc.imread(img_center_filename)[80:,:,:]
			img_left = misc.imread(img_left_filename)[80:,:,:]
			img_right = misc.imread(img_right_filename)[80:,:,:]
	
			steering_angle = float(row[3])
			steering_angle_left = steering_angle + abs(steering_angle)*0.3
			steering_angle_right = steering_angle - abs(steering_angle)*0.3
		
			num_samples = 1
			if(steering_angle > 0):
				num_samples = 3	

			for j in range(num_samples):
				X.append(img_center)
				y.append(steering_angle)
				X.append(img_left)
				y.append(steering_angle_left)
				X.append(img_right)
				y.append(steering_angle_right)

	yield np.asarray(X), np.asarray(y)

if __name__ == "__main__":
	#print("MAIN")
	#X, y = get_data()
	#print(X.shape)
	#print(y.shape)
	model = get_model()
	#X = np.reshape(X, (X.shape[0], 3, 160, 320))
	#print("FIT")
	for i in (range(3)):
		print("Epoch: %d\n" % i)
		for X, y in BatchGenerator():
			model.fit(X, y, batch_size=32, nb_epoch=1)
	#model.save('./comma.h5')
	model.save_weights("./comma.keras")
	with open('./comma.json', 'w') as outfile:
		json.dump(model.to_json(), outfile)
