import numpy as np 
import pandas as pd 
import cv2
import h5py
import os
from glob import glob

#extracting file name from path for sorting comparison
def sortkeyfunc(s):
	return int(os.path.basename(s)[:-4])


# images is list of abs_paths of image_file 
def create_h5_data(images,name_of_data):
	
	#4dimensional array to store images
	no_of_image = len(images)
	data = np.zeros([no_of_image,64,64,3])

	# defining image size to store
	HEIGHT = 64
	WIDTH = 64
	CHANNEL = 3

	print("starting the process") 
	# taking all images and putting in numpy array with resizing image to 64,64,3
	i=0
	for img_name in images:
		img = cv2.imread(img_name)
		img = cv2.resize(img,(WIDTH,HEIGHT),interpolation=cv2.INTER_CUBIC)
		data[i,:,:,:] = img
		i = i+1

	with h5py.File("data.h5","a") as hf:
		hf.create_dataset(name_of_data,data=data)


def make_image_path_list():
	#abs_path of current directory
	PATH = os.path.abspath('.')

	#join the test and train folders path
	Source_train = os.path.join(PATH,'train')
	Source_test = os.path.join(PATH,'test')

	#search for png files in folder then sort the images according to their names
	images = glob(os.path.join(Source_train,"*.png"))
	images_train_list = sorted(images,key=sortkeyfunc)

	images = glob(os.path.join(Source_test,"*.png"))
	images_test_list = sorted(images,key=sortkeyfunc)

	return images_train_list,images_test_list


def create_label_data(name_of_data):
	print("creating labels")
	labels = pd.read_csv('Train.csv',sep=',')
	label1 = np.array(labels.values[:,1])
	label = np.zeros([1,len(label1)])
	label[:,:] = label1.reshape(1,len(label1))
	print(label)
	with h5py.File("data.h5","a") as hf:
		hf.create_dataset(name_of_data,data=label)



#getting path list
# train_list,test_list = make_image_path_list()

#run this three once to create data.h5 files
# create_h5_data(train_list,"Train_X")
# create_h5_data(test_list,"Test_X")
# create_label_data("Train_Y")


#retrieving data array from data.h5
def load_data_set():
	with h5py.File("data.h5","r") as hf:
		Train_X = hf["Train_X"]
		Train_Y = hf["Train_Y"]
		Test_X = hf["Test_X"]

		return Train_X[:],Train_Y[:],Test_X[:]
		# return Train_Y[:]


Train_X,Train_Y,Test_X = load_data_set()
# Train_Y = load_data_set()

print(Train_X.shape)
print(Train_Y.shape)
print(Test_X.shape)



