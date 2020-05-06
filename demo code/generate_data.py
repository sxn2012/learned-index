'''
author: Xinnan SHEN
date: 03/05/2020
'''
#This file is used for generating dataset (data.csv)
import os
import codecs
import random
def main():
	datalist=[]
	for i in range(0,3000):
		x=random.randint(1000,9999)
		datalist.append(x)
	for i in range(0,len(datalist)):
		temp=False
		for j in range(0,len(datalist)-i-1):
			if datalist[j]>datalist[j+1]:
				t=datalist[j]
				datalist[j]=datalist[j+1]
				datalist[j+1]=t
				temp=True
		if not temp:
			break
	current_path=os.path.abspath(os.curdir)
	f=codecs.open(os.path.join(current_path,"data.csv"), "w", "utf-8")
	for i in range(0,len(datalist)):
		f.write(str(datalist[i])+","+str(i)+"\n")
	f.close()
	return
if __name__ == '__main__':
	main()
