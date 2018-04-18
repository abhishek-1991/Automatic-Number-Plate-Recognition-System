import numpy as np
#import pandas as pd
import scipy.special
#import matplotlib.pyplot as plt
import scipy.misc
import os
#import threading
#from multiprocessing import Process, Pool
#from sklearn.model_selection import cross_val_score
#from sklearn.tree import DecisionTreeClassifier
#from sklearn import svm
#from skimage.feature import hog
import cv2
import pandas as pd
from tqdm import tqdm
from os import listdir
from os.path import isfile,join
import sys
import hickle as hkl
from sklearn.neural_network import MLPClassifier
#img_array = scipy.misc.imread("Untitled1.png",flatten=True)

#img_data = 255.0 - img_array.reshape(784)
#img_data = (img_data/255.0 * 0.99) + 0.01

values = ['0.0','1.0','2.0','3.0','4.0','5.0','6.0','7.0','8.0','9.0','10.0','11.0','12.0','13.0','14.0','15.0','16.0','17.0','18.0','19.0','20.0','21.0','22.0','23.0','24.0','25.0','26.0','27.0','28.0','29.0','30.0','31.0','32.0','33.0','34.0','35.0']

values2 = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

k = 0
dic = {}
dic2 = {}
for p in range(0,len(values)):
	dic[values[p]] = k
	dic2[k] = values2[p] 
	k += 1

class NeuralNetwork:

	def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate, activate):
		self.inodes = input_nodes
		self.hidden1_nodes = hidden_nodes[0]
		self.hidden2_nodes = hidden_nodes[1]
		self.onodes = output_nodes
		self.lr = learning_rate
		self.theta1 = np.random.randn(self.hidden1_nodes, self.inodes).astype(np.float32) * np.sqrt(2.0/(self.inodes))
		self.b1 = np.random.randn(self.hidden1_nodes, 1).astype(np.float32)
		self.theta2 = np.random.randn(self.hidden2_nodes, self.hidden1_nodes).astype(np.float32) * np.sqrt(2.0/(self.hidden1_nodes))
		self.b2 = np.random.randn(self.hidden2_nodes, 1).astype(np.float32)
		self.theta3 = np.random.randn(self.onodes, self.hidden2_nodes).astype(np.float32) * np.sqrt(2.0/(self.hidden2_nodes))
		self.b3 = np.random.randn(self.onodes, 1).astype(np.float32)
		self.activate = activate

	def sigmoid_activation(self, x):
		return scipy.special.expit(x)

	def tanh_activation(self, x):
		return np.tanh(x)

	def relu_activation(self, x):
		result = np.maximum(x, -0.1)
		result = np.minimum(result, 1)
		return result	

	def activation(self, x):
		if self.activate == 'relu':
			return self.relu_activation(x)
		elif self.activate == 'sigmoid':
			return self.sigmoid_activation(x)
		else:
			return self.tanh_activation(x)
		

	def train(self, x_list, target_list):
		x = np.array(x_list,ndmin=2).T
		y = np.array(target_list,ndmin=2).T

		z1 = np.dot(self.theta1,x) + self.b1
		a1 = self.activation(z1)
		z2 = np.dot(self.theta2,a1) + self.b2
		a2 = self.activation(z2)
		z3 = np.dot(self.theta3, a2) + self.b3
		a3 = self.activation(z3)

		loss = sum((y - a3)**2)
		e3 = y - a3
		e2 = np.dot(self.theta3.T, e3) * a2 * (1.0 - a2) 
		e1 = np.dot(self.theta2.T, e2) * a1 * (1.0 - a1)

		self.theta3 += self.lr * np.dot(e3, a2.T)
		self.theta2 += self.lr * np.dot(e2, a1.T)
		self.theta1 += self.lr * np.dot(e1, x.T)
		self.b3 += self.lr * e3 
		self.b2 += self.lr * e2
		self.b1 += self.lr * e1
		return loss
		

	def classify(self, x_list):
		x = np.array(x_list,ndmin=2).T
		z1 = np.dot(self.theta1,x)
		a1 = self.activation(z1)
		z2 = np.dot(self.theta2,a1)
		a2 = self.activation(z2)
		z3 = np.dot(self.theta3, a2)
		a3 = self.activation(z3)
		
		return a3


input_nodes = 625
output_nodes = 36
hidden_nodes1 = 200
hidden_nodes2 = 100
learning_rate = 0.01

f = open("full_train.csv","r")
# f = open("new_g_train.csv","r")
f_list = f.readlines()[1:45500]
# print f_list[1]
# print f_list[1]
f.close()


def validate(files):
	score = 0.0
	total_files = len(files)
	wrong_pred = []
	for file_name in files:
		file_name = mypath+"/"+file_name
		img = cv2.imread(file_name)
		result = file_name[14]
		# print result
		img = cv2.resize(img,(25,25))
		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(1,625)
		res = nn.classify(np.asfarray(gray_img)/255.0 * 0.99 + 0.01)
		
		arr = np.argmax(res)
		# print max(res), dic2[arr], result
		#print file_name
		if dic2[arr] == result:
			score += 1.0
		else:
			wrong_pred.append((result, dic2[arr]))
		
	final_score = float(float(score)/float(total_files))
	return final_score*100.0, wrong_pred


mypath = sys.argv[1]
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
hidden_inp1 = int(sys.argv[2])
hidden_inp2 = int(sys.argv[3])
activation = sys.argv[4]


print "--- Training with activation: " + activation + " hidden1 size: " + str(hidden_inp1) + " and hidden2 size: " + str(hidden_inp2)

#print mypath

nn = NeuralNetwork(input_nodes,(hidden_inp1, hidden_inp2),output_nodes,learning_rate, activation)
#i = 0
epochs = 30
ep = 1
past_loss = [20,20]
# k = 0
validation_accuracy = 0.0
'''

for i in range(epochs):
	loss = 0.0
	# for record in tqdm(f_list):
	for record in tqdm(f_list):
		all_values = record.split(',')
		scaled_input = np.asfarray(all_values[1:])/255.0 * 0.99 + 0.01
		y = np.zeros(output_nodes) + 0.01
		y[int(dic[all_values[0]])] = 0.99
		loss += nn.train(scaled_input,y)	
	validation_accuracy, wrong_predictions = validate(files)
	print "Loss after epoch " + str(ep) + " is : " + str(float(loss)/float(11500))
	print "Validation accuracy after epoch " + str(ep) + " is " + str(validation_accuracy)
	past_loss.append(float(loss)/float(11500))
	ep += 1
	if (past_loss[-1] > past_loss[-2] - 0.000005) and (past_loss[-1] > past_loss[-3] - 0.000005):
		print "stopped due to lack of improvement"
		break


'''
'''
file_output = "Result for " + activation + " of hidden layer 1 size: " + str(hidden_inp1) + " and of hidden layer 2 size: " + str(hidden_inp2) + "\n"

wrong_predictions = ' '.join(wrong_predictions) + "\n"

final_accuracy = "Accuracy of this model: " + str(validation_accuracy) + "\n\n"

f = open('output.txt','a')

f.write(file_output + wrong_predictions + final_accuracy)

data = {'theta1': nn.theta1, 'theta2': nn.theta2, 'theta3': nn.theta3, 'b1': nn.b1, 'b2': nn.b2, 'b3':nn.b3}

hickle_file = './hickle_files/1' + activation + '_' + str(hidden_inp1) + '_' + str(hidden_inp2) + '_' + str(int(validation_accuracy)) + '.hkl'

hkl.dump(data, hickle_file)
'''

hickle_file = './hickle_files/1' + activation + '_' + str(hidden_inp1) + '_' + str(hidden_inp2) + '_' + str(82) + '.hkl'

data2 = hkl.load(hickle_file)
nn.theta1 = data2['theta1']
nn.theta2 = data2['theta2']
nn.theta3 = data2['theta3']
nn.b1 = data2['b1']
nn.b2 = data2['b2']
nn.b3 = data2['b3']

final_score, wrong_preds = validate(files)

print final_score, wrong_preds




