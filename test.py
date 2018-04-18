import os
import sys
from final_code_ann import *

file_name = sys.argv[1]
imglist = []
print "file_name " + file_name
for img in os.listdir(file_name):
	imglist.append(img)

imglist.sort()
print imglist
plate = wrapper_for_validate_only(imglist)
print plate
