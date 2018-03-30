import numpy as np
#import pandas as pd
import scipy.special
#import matplotlib.pyplot as plt
import scipy.misc
#import threading
#from multiprocessing import Process, Pool
#from sklearn.model_selection import cross_val_score
#from sklearn.tree import DecisionTreeClassifier
#from sklearn import svm
#from skimage.feature import hog
import cv2
from tqdm import tqdm
from os import listdir
from os.path import isfile,join
import sys
import hickle as hkl
#img_array = scipy.misc.imread("Untitled1.png",flatten=True)

#img_data = 255.0 - img_array.reshape(784)
#img_data = (img_data/255.0 * 0.99) + 0.01

values = ['0.0','1.0','2.0','3.0','4.0','5.0','6.0','7.0','8.0','9.0','10.0','11.0','12.0','13.0','14.0','15.0','16.0','17.0','18.0','19.0','20.0','21.0','22.0','23.0','24.0','25.0','26.0','27.0','28.0','29.0','30.0','31.0','32.0','33.0','34.0','35.0']

k = 0
dic = {}
for p in values:
	dic[p] = k
	k += 1

class NeuralNetwork:

	def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
		self.inodes = input_nodes
		self.hnodes = hidden_nodes
		self.onodes = output_nodes
		self.lr = learning_rate
		#self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
		self.wih = np.random.randn(self.hnodes, self.inodes).astype(np.float32) * np.sqrt(2.0/(self.inodes))
		#self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)
		self.who = np.random.randn(self.onodes, self.hnodes).astype(np.float32) * np.sqrt(2.0/(self.hnodes))
		self.activation_func = lambda x: scipy.special.expit(x) 

	def relu_activation(self, x):
		result = np.maximum(x, -0.1)
		result = np.minimum(result, 1)
		return result	


	def train(self, inputs_list, target_list):
		inputs = np.array(inputs_list,ndmin=2).T
		targets = np.array(target_list,ndmin=2).T

		#Calculate the final output
		hidden_inputs = np.dot(self.wih,inputs)
		hidden_outputs = self.activation_func(hidden_inputs)
		#hidden_outputs = self.relu_activation(hidden_inputs)
		final_inputs = np.dot(self.who, hidden_outputs)
		final_outputs = self.activation_func(final_inputs)
		#final_outputs = self.relu_activation(final_inputs)
	

		loss = sum((targets - final_outputs)**2)
		#calculate the error
		error = targets - final_outputs
		hidden_error = np.dot(self.who.T, error)

		#updating link weights
		self.who += self.lr * np.dot(error*final_outputs*(1.0-final_outputs),hidden_outputs.T)
		self.wih += self.lr * np.dot(hidden_error*hidden_outputs*(1.0-hidden_outputs),inputs.T)
		return loss
		

	def classify(self, inputs_list):
		inputs = np.array(inputs_list,ndmin=2).T
		hidden_inputs = np.dot(self.wih, inputs)
		hidden_outputs = self.activation_func(hidden_inputs)
		#hidden_outputs = self.relu_activation(hidden_inputs)

		final_inputs = np.dot(self.who, hidden_outputs)
		final_outputs = self.activation_func(final_inputs)
		#final_outputs = self.relu_activation(final_inputs)
		return final_outputs


input_nodes = 625
output_nodes = 36
hidden_nodes = 200
learning_rate = 0.01

f = open("ann_train.csv","r")
f_list = f.readlines()[1:11500]
# print f_list[1]
# print f_list[1]
f.close()

nn = NeuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
#i = 0
epochs = 100
ep = 1
for i in range(epochs):
	for record in tqdm(f_list):
		all_values = record.split(',')
		scaled_input = np.asfarray(all_values[1:])/255.0 * 0.99
		targets = np.zeros(output_nodes) + 0.01
		targets[int(dic[all_values[0]])] = 0.99
		loss = nn.train(scaled_input,targets)	
	print "Loss after epoch " + str(ep) + " is : " + str(loss)
	if loss < 0.004:
		break
	ep += 1

data = {'wih': nn.wih, 'who': nn.who}

hkl.dump(data, 'new_data_file.hkl')

#data2 = hkl.load('new_data_file.hkl')

#nn.wih = data2['wih']
#nn.who = data2['who']

mypath = sys.argv[1]
#print mypath
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#print files
for file_name in files:
	file_name = mypath+"/"+file_name
	img = cv2.imread(file_name)
	img = cv2.resize(img,(25,25))
	print img.shape
	# img = cv2.imread("../project/BadImag/Bmp/Sample005/img005-00001.png")
	# img = cv2.resize(img, (25,25,3))	
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(1,625)
	# (thresh, im_bw) = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	# im_bw = im_bw.reshape(1,400)
	# # # print lr.predict(gray_img)
	res = nn.classify((255 - np.asfarray(gray_img))/255.0 * 0.99)
	arr = np.argmax(res)
	print file_name
	print arr



