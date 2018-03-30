import numpy as np
import pandas as pd
import scipy.special
import matplotlib.pyplot as plt
import scipy.misc
import threading
from multiprocessing import Process, Pool
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from skimage.feature import hog
import cv2
from tqdm import tqdm

img_array = scipy.misc.imread("Untitled1.png",flatten=True)

img_data = 255.0 - img_array.reshape(784)
img_data = (img_data/255.0 * 0.99) + 0.01

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
		self.wih = (np.random.rand(self.hnodes, self.inodes) - 0.5)
		self.who = (np.random.rand(self.onodes, self.hnodes) - 0.5)
		self.activation_func = lambda x: scipy.special.expit(x)


	def train(self, inputs_list, target_list):
		inputs = np.array(inputs_list,ndmin=2).T
		targets = np.array(target_list,ndmin=2).T

		#Calculate the final output
		hidden_inputs = np.dot(self.wih,inputs)
		hidden_outputs = self.activation_func(hidden_inputs)
		final_inputs = np.dot(self.who, hidden_outputs)
		final_outputs = self.activation_func(final_inputs)


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

		final_inputs = np.dot(self.who, hidden_outputs)
		final_outputs = self.activation_func(final_inputs)
		return final_outputs


input_nodes = 625
output_nodes = 36
hidden_nodes = 100
learning_rate = 0.01

f = open("ann_train.csv","r")
f_list = f.readlines()[1:11500]
# print f_list[1]
# print f_list[1]
f.close()

nn = NeuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
#i = 0
epochs = 50
ep = 1
for i in range(epochs):
	for record in tqdm(f_list):
		all_values = record.split(',')
		scaled_input = np.asfarray(all_values[1:])/255.0 + 0.1
		targets = np.zeros(output_nodes) + 0.01
		targets[int(dic[all_values[0]])] = 0.99
		loss = nn.train(scaled_input,targets)	
	print "Loss after epoch " + str(ep) + " is : " + str(loss)
	if loss < 0.004:
		break
	ep += 1

# test_data = open("mnist_test.csv","r")
# test_data = open("test.csv","r")
# test_d = test_data.readlines()[1:]
# test_data.close()

# # scorecard = [("ImageId","Label")]
# scorecard = []
# output = 0
# i = 0
# for record in test_d:
# 	all_values = record.split(',')
# 	answer = all_values[0]
# 	print answer
# 	scaled_input = np.asfarray(all_values[:])/255.0 * 0.99
# 	res = nn.classify(scaled_input)
# 	# print res
# 	# break
# 	output = np.argmax(res)
# 	i += 1
# 	# scorecard.append((str(i),str(output)))
# 	scorecard.append(output)
# 	if str(output) == answer:
# 		# print "yes"
# 		scorecard.append(1)
# 	else:
# 		scorecard.append(0)

img = cv2.imread("../project/training_data/all_train/2_1.jpg")
print img.shape
# img = cv2.imread("../project/BadImag/Bmp/Sample005/img005-00001.png")
# img = cv2.resize(img, (25,25,3))	
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(1,625)
# (thresh, im_bw) = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# im_bw = im_bw.reshape(1,400)
# # # print lr.predict(gray_img)
res = nn.classify(np.asfarray(gray_img)/255.0 + 0.01)
arr = np.argmax(res)
print arr


# print "The accuracy is: " + str(float(sum(scorecard)/len(scorecard)))
# 	# print scorecard	
# print scorecard
# import csv


# from keras.models import Sequential
# from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
 
# def createModel():
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
#     model.add(Conv2D(64, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
 
#     model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#     model.add(Conv2D(64, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
 
#     model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#     model.add(Conv2D(64, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
 
#     model.add(Flatten())
#     model.add(Dense(512, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(nClasses, activation='softmax'))
     
#     return model


# df = pd.read_csv('mnist_train.csv')
# target = df.iloc[:,0]
# train_set = df.iloc[:,1:]
# # train_set = train_set.reshape(train_set.shape[0],img_rows,img_cols,1)
# nClasses = 10
# test_df = pd.read_csv('mnist_test.csv')
# test_targets = test_df[test_df.columns[0]]
# test_data = test_df.iloc[:,1:]
# # test_data = test_data.reshape(test_data.shape[0],img_rows,img_cols,1)
# img_rows, img_cols = 28, 28
# train_set = np.array(train_set).reshape(train_set.shape[0],img_rows,img_cols,1)
# test_data = np.array(test_data).reshape(test_data.shape[0],img_rows,img_cols,1)
# print train_set.shape

# input_shape = (img_rows, img_cols, 1)
# model1 = createModel()
# batch_size = 256
# epochs = 2
# model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
 
# history = model1.fit(train_set, target, batch_size=batch_size, epochs=epochs, verbose=1, 
#                    validation_data=(test_data, test_targets))
 
# score = model1.evaluate(test_data, test_targets)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

# with open("output.csv", "wb") as f:
#     writer = csv.writer(f,quoting=csv.QUOTE_ALL)
#     # writer.writerows(["ImageId","Label"])
#     # for row in scorecard:
#     writer.writerows(scorecard)

# res = nn.classify(img_data)
# print np.argmax(res)


# df = pd.read_csv('train.csv')
# target = df.iloc[:,0]
# train_set = df.iloc[:,1:]
# print train_set, target
# input_data = df[]
# clf = DecisionTreeClassifier(random_state=0)
# clf.fit(train_set,target)

# test_df = pd.read_csv('test.csv')
# test_set = test_df
# print test_set
# res = clf.predict(test_set)

# clf = svm.SVC(kernel='linear')
# clf.fit(train_set,target)
# # res = clf.predict(test_set)
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(max_depth=2, random_state=0)
# clf.fit(train_set,target)
# res = clf.predict(test_set)
# # print res
# i = 0
# scorecard = [("ImageId","Label")]
# for row in res:
# 	i += 1
# 	scorecard.append((i,row))

# # print scorecard

# with open("output.csv", "wb") as f:
#     writer = csv.writer(f,quoting=csv.QUOTE_ALL)
# # #     # writer.writerows(["ImageId","Label"])
# #     i = 0
#     writer.writerows(scorecard)

# print res
# j = 0
# for i in range(len(res)):
# 	if res[i] == scorecard[i]:
# 		j += 1

# print i, len(res)

# import cv2
# # import numpy as np
# # import matplotlib.pyplot as plt

# img = cv2.imread('photo_1.jpg')

# im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# im_gray = cv2.GaussianBlur(im_gray, (5,5), 0)

# ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

# ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# rects = [cv2.boundingRect(ctr) for ctr in ctrs]
# output1 = 1
# for rect in rects:
#     cv2.rectangle(im_gray,(rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]),(0, 255, 0), 3) 
#     leng = int(rect[3] * 1.6)
#     pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
#     pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
#     roi = im_gray[pt1:pt1+leng, pt2:pt2+leng]
#     # nbr[0] = '1'
#     # print roi.shape
#     cv2.drawContours(im_th.copy(),list, 0, (123,123,123), 2)
#     if((roi.shape[0]) * (roi.shape[1]) > 784):
#     	roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
#     	roi = cv2.dilate(roi, (3, 3))
#         # roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
#     	print roi.shape
#         test = np.array(roi).reshape(784)
#         scaled_test = np.asfarray(test)/255.0 * 0.99
#         res1 = nn.classify(scaled_test)
# 	output1 = np.argmax(res1)
# 	print "Hello" + str(output1)
#     cv2.putText(img, str(output1), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)




# cv2.imshow("Resulting Image with Rectangular ROIs", img)
# cv2.waitKey()
