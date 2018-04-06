import numpy as np
#import pandas as pd
import scipy.special
#import matplotlib.pyplot as plt
import scipy.misc
import os

import cv2
import pandas as pd
from tqdm import tqdm
from os import listdir
from os.path import isfile,join
import sys
import hickle as hkl


# values = ['0.0','1.0','2.0','3.0','4.0','5.0','6.0','7.0','8.0','9.0','10.0','11.0','12.0','13.0','14.0','15.0','16.0','17.0','18.0','19.0','20.0','21.0','22.0','23.0','24.0','25.0','26.0','27.0','28.0','29.0','30.0','31.0','32.0','33.0','34.0','35.0']
values = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35']
values2 = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']


k = 0
dic = {}
dic2 = {}

for p in range(0,len(values)):
	dic[values[p]] = k
	dic2[k] = values2[p] 
	k += 1

class NeuralNetwork:

	def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate, activate, batch_size):
		self.inodes = input_nodes
		self.hidden1_nodes = hidden_nodes[0]
		self.hidden2_nodes = hidden_nodes[1]
		self.onodes = output_nodes
		self.lr = learning_rate
		self.batch_size = batch_size
		self.theta1 = np.random.randn(self.hidden1_nodes, self.inodes).astype(np.float32) * np.sqrt(2.0/(self.inodes))
		self.b1 = np.random.randn(self.hidden1_nodes, self.batch_size).astype(np.float32)
		self.theta2 = np.random.randn(self.hidden2_nodes, self.hidden1_nodes).astype(np.float32) * np.sqrt(2.0/(self.hidden1_nodes))
		self.b2 = np.random.randn(self.hidden2_nodes, self.batch_size).astype(np.float32)
		self.theta3 = np.random.randn(self.onodes, self.hidden2_nodes).astype(np.float32) * np.sqrt(2.0/(self.hidden2_nodes))
		self.b3 = np.random.randn(self.onodes, self.batch_size).astype(np.float32)
		self.activate = activate
		self.Beta1 = 0.9
		self.Beta2 = 0.999
		self.mt = []
		self.v = []
		self.gamma = 0.9
		self.cache = []
		for i in range(0,6):
			self.mt.append(0.0)
			self.cache.append(0.0)
			self.v.append(0.0)
		self.ephsilon = 10 ** -8
		self.t = 0
		self.alpha = 0.001

	def sigmoid_activation(self, x):
		return scipy.special.expit(x)

	def tanh_activation(self, x):
		return np.tanh(x)

	def relu_activation(self, x):
		result = x * (x>0)
		return result	

	def softmax_activation(self, x):
		e_x = np.exp(x - np.max(x))
		return e_x/e_x.sum(axis=0)

	def relu_backprop(self,x):
		return 1.0 * (x > 0)

	def sigmoid_backprop(self, x):
		return x * (1.0 - x)

	def tanh_backprop(self, x):
		return (1 - x**2)

	def activation(self, x):
		if self.activate == 'relu':
			return self.relu_activation(x)
		elif self.activate == 'sigmoid':
			return self.sigmoid_activation(x)
		else:
			return self.tanh_activation(x)
	
	def backprop(self, x):
		if self.activate == 'relu':
			return self.relu_backprop(x)
		elif self.activate == 'sigmoid':
			return self.sigmoid_backprop(x)
		else:
			return self.tanh_backprop(x)

	def adam_optimization(self, theta, derivative, i):
		self.mt[i] = self.Beta1 * self.mt[i] + (1.0 - self.Beta1) * derivative
		self.v[i] = self.Beta2 * self.v[i] + (1.0 - self.Beta2) * (derivative**2)
		new_mt = self.mt[i]/(1.0 - self.Beta1 ** self.t)
		new_v = self.v[i]/(1.0 - self.Beta2 ** self.t)
		new_theta = theta + self.alpha * new_mt/(np.sqrt(new_v) + self.ephsilon)
		return new_theta

	def sgd_optimization(self, theta, derivative):
		new_theta = theta + derivative
		return new_theta

	def rmsprop_optimization(self, theta, derivative, i):
		self.cache[i] = self.gamma * self.cache[i] + (1.0 - self.gamma) * (derivative ** 2)
		theta = theta - self.alpha * derivative / (np.sqrt(self.cache[i]) + self.ephsilon)
		return theta 

	def train(self, x_list, target_list):
		x = np.array(x_list,ndmin=2).T
		y = np.array(target_list,ndmin=2).T

		z1 = np.dot(self.theta1,x) + self.b1
		a1 = self.activation(z1)
		z2 = np.dot(self.theta2,a1) + self.b2
		a2 = self.activation(z2)
		z3 = np.dot(self.theta3, a2) + self.b3
		# a3 = self.softmax_activation(z3)
		a3 = self.activation(z3)

		loss = sum((y - a3)**2)
		e3 = y - a3
		e2 = np.dot(self.theta3.T, e3) * self.backprop(a2)
		e1 = np.dot(self.theta2.T, e2) * self.backprop(a1)

		self.theta3 = self.adam_optimization(self.theta3, np.dot(e3, a2.T),2) 
		self.theta2 = self.adam_optimization(self.theta2, np.dot(e2, a1.T),1)
		self.theta1 = self.adam_optimization(self.theta1, np.dot(e1, x.T),0)
		self.b3 = self.adam_optimization(self.b3, e3, 5)
		self.b2 = self.adam_optimization(self.b2, e2, 4)
		self.b1 = self.adam_optimization(self.b1, e1, 3)

		return sum(loss)
		

	def classify(self, x_list):
		x = np.array(x_list,ndmin=2).T
		z1 = np.dot(self.theta1,x)
		a1 = self.activation(z1)
		z2 = np.dot(self.theta2,a1)
		a2 = self.activation(z2)
		z3 = np.dot(self.theta3, a2)
		# a3 = self.softmax_activation(z3)
		a3 = self.activation(z3)
		
		return a3


input_nodes = 625
output_nodes = 36
output_nodes_alphabets = 26
learning_rate = 0.01

f = open("full_train.csv","r")
f_list = f.readlines()[1:49601]
f.close()


def validate(files):
	score = 0.0
	total_files = len(files)
	wrong_pred = []
	d = {}
	for file_name in files:
		file_name = mypath+"/"+file_name
		img = cv2.imread(file_name)
		result = file_name[15].upper()
		# print result
		img = cv2.resize(img,(25,25))
		gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(1,625)
		res = nn.classify(np.asfarray(gray_img)/255.0 * 0.99 + 0.01)
		arr = np.argmax(res)
		prediction = dic2[arr]
		# print prediction, result
		if prediction == result:
			score += 1.0
		else:
			if result not in d:
				d[result] = 1
			else:
				d[result] += 1
			wrong_pred.append(result)
		
	final_score = float(float(score)/float(total_files))
	print d
	return final_score*100.0, wrong_pred


mypath = sys.argv[1]
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
hidden_inp1 = int(sys.argv[2])
hidden_inp2 = int(sys.argv[3])
activation = sys.argv[4]


print "--- Training with activation: " + activation + " hidden1 size: " + str(hidden_inp1) + " and hidden2 size: " + str(hidden_inp2)
batch_size = 32
nn = NeuralNetwork(input_nodes,(hidden_inp1, hidden_inp2),output_nodes,learning_rate, activation, batch_size)
epochs = 60
ep = 1
past_loss = [20,20]
validation_accuracy = 0.0



X_train = []
Y_train = []

for record in f_list:
	all_values = record.split(',')
	scaled_input = np.asfarray(all_values[1:])/255.0 * 0.99 + 0.01
	# print scaled_input.shape
	y = np.zeros(output_nodes) + 0.01
	y[int(dic[all_values[0]])] = 0.99
	X_train.append([scaled_input])
	Y_train.append([y])


X_train = np.concatenate(X_train)
Y_train = np.concatenate(Y_train)


for i in range(epochs):
	k = 0
	loss = 0.0
	nn.t = 0
	for i in tqdm(range(0,len(X_train), batch_size)):
		x_train = X_train[i:i+batch_size]
		y_train = Y_train[i:i+batch_size]
		nn.t += 1
		loss += nn.train(x_train, y_train)

	validation_accuracy, wrong_predictions = validate(files)
	print "Loss after epoch " + str(ep) + " is : " + str(float(loss)/float(48500))
	print "Validation accuracy after epoch " + str(ep) + " is " + str(validation_accuracy)
	past_loss.append(float(loss)/float(48500))
	ep += 1	



file_output = "Result for " + activation + " of hidden layer 1 size: " + str(hidden_inp1) + " and of hidden layer 2 size: " + str(hidden_inp2) + "\n"

wrong_predictions = ' '.join(wrong_predictions) + "\n"

final_accuracy = "Accuracy of this model: " + str(validation_accuracy) + "\n\n"

f = open('output.txt','a')

f.write(file_output + wrong_predictions + final_accuracy)

data = {'theta1': nn.theta1, 'theta2': nn.theta2, 'theta3': nn.theta3, 'b1': nn.b1, 'b2': nn.b2, 'b3':nn.b3}

hickle_file = './hickle_files/1' + activation + '_' + str(hidden_inp1) + '_' + str(hidden_inp2) + '_' + str(int(validation_accuracy)) + '.hkl'

hkl.dump(data, hickle_file)

'''
hickle_file = './hickle_files/1' + activation + '_' + str(hidden_inp1) + '_' + str(hidden_inp2) + '_' + str(85) + '.hkl'

data2 = hkl.load(hickle_file)
nn.theta1 = data2['theta1']
nn.theta2 = data2['theta2']
nn.theta3 = data2['theta3']
nn.b1 = data2['b1']
nn.b2 = data2['b2']
nn.b3 = data2['b3']

final_score, wrong_preds = validate(files)

print final_score
'''