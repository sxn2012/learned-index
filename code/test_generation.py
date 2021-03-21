import os
import codecs
def readfile(filename):
	current_path=os.path.abspath(os.curdir)
	file_path=os.path.join(current_path,filename)
	if not os.path.exists(file_path):
		print("error:file not found:"+filename)
		return ""
	f=codecs.open(file_path,"r","utf-8")
	s=f.read()
	f.close()
	return s
train_data=readfile("data.csv")
import random
list_train=train_data.split("\r\n")
list_test=[]
for i in range(0,len(list_train)//3):
  list_test.append(list_train[random.randint(0,len(list_train)-1)])
str=""
for i in range(0,len(list_test)):
  str=str+list_test[i]+"\r\n"
f=codecs.open("data_test.csv","w","utf-8")
f.write(str)
f.close()
