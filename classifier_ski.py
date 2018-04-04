from sklearn.neural_network import MLPClassifier
import numpy as np
import scipy.misc
import os
import scipy.special
import cv2
from tqdm import tqdm
from os import listdir
from os.path import isfile,join
import sys
import time
import pandas as pd

values = ['0.0','1.0','2.0','3.0','4.0','5.0','6.0','7.0','8.0','9.0','10.0','11.0','12.0','13.0','14.0','15.0','16.0','17.0','18.0','19.0','20.0','21.0','22.0','23.0','24.0','25.0','26.0','27.0','28.0','29.0','30.0','31.0','32.0','33.0','34.0','35.0']

values2 = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

label = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

k = 0
dic = {}
dic2 = {}
for p in range(0,len(values)):
        dic[values[p]] = k
        dic2[k] = values2[p]
        k += 1

input_nodes = 625
output_nodes = 36
hidden_nodes1 = 200
hidden_nodes2 = 100
learning_rate = 0.01


f = open("full_train.csv","r")
f_list = f.readlines()[1:45500]
f.close()

mypath = sys.argv[1]
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#hidden_inp1 = int(sys.argv[2])
#hidden_inp2 = int(sys.argv[3])
#activation = sys.argv[4]

clf = MLPClassifier(solver='sgd', alpha=1e-5,hidden_layer_sizes=(400, 300), random_state=1, verbose = True)
epochs = 1

for i in range(epochs):
	loss=0.0
	an = pd.read_csv('full_train.csv')
	an = an.iloc[0:40000]
	label = an['0']
	an = an.drop(an.columns[[0]], axis=1)
	#an = an.drop(an.index[49950:])
	#an = an.replace([NaN],[0])
	an= an/255
	#y = np.zeros(output_nodes) + 0.01
	#print y
	#print dic
        #y[int(dic[all_values[0]])] = 0.99
	clf.fit(an,label)
	break
'''	for record in tqdm(f_list):
		all_values = record.split(',')
		print "type all_values and all_values[0]"		
		print type(all_values)
		print type(all_values[0])
		time.sleep(2)
                scaled_input = np.asfarray(all_values[1:])/255.0 * 0.99 + 0.01
                print type(scaled_input)
		print type(scaled_input[0])
		print len(scaled_input)
		
		time.sleep(5)
		y = np.zeros(output_nodes) + 0.01
                y[int(dic[all_values[0]])] = 0.99
		clf.fit(scaled_input, y)
		break
'''
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
                res = clf.predict(gray_img/255.0)
		#res = nn.classify(np.asfarray(gray_img)/255.0 * 0.99 + 0.01)

                arr = np.argmax(res)
                # print max(res), dic2[arr], result
                #print file_name
                if dic2[arr] == result:
                        score += 1.0
                else:
                        wrong_pred.append((result, dic2[arr]))

        final_score = float(float(score)/float(total_files))
        return final_score*100.0, wrong_pred


print "Predicting the result from file"
validation_accuracy, wrong_predictions = validate(files)
print "validation accuracy is " + str(validation_accuracy) +"\n wrong_predictions are " + str(wrong_predictions)

'''for file_name in files:
	file_name = mypath+"/"+file_name
	img = cv2.imread(file_name)
        result = file_name[14]
        # print result
        img = cv2.resize(img,(25,25))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(1,625)
        #print gray_img
	#print type(gray_img)
	res = clf.predict(gray_img/255.0)
	print file_name
	print res
	print dic2[(res[0])]
	#break
'''

