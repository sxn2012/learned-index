'''
author: Xinnan SHEN
date: 03/05/2020
'''
#This is a B-Tree model
#reference: https://www.jianshu.com/p/c625a009e488
from random import shuffle
import random
import os
import codecs
import numpy as np
from sklearn.model_selection import train_test_split
root_node = None
_M = 3
class Logger(object):
    @classmethod
    def tree(cls, node, child_name, dsc, depth):
        if depth == 0:
            head = "|   " * depth
            print(head + "+--" + dsc(node))
            depth = depth + 1
        for child in getattr(node, child_name):
            head = "|   " * depth
            print(head + "+--" + dsc(child))
            cls.tree(child, child_name, dsc, depth + 1)
class BKeyword(object):
    def __init__(self, key, loc):
        self.key = key
        self.loc = loc
class BNode(object):
    def __init__(self):
        self._parent: BNode = None
        self.keywords = []
        self.child_nodes = []
    # set parent node
    def set_parent(self, node):
        self._parent = node
        if node.get_parent() is None:
            global root_node
            root_node = node.get_parent()
    # get parent node
    def get_parent(self):
        return self._parent
    # add child node to right location
    def insert_child_node(self, index, add_node):
        add_node.set_parent(self)
        self.child_nodes.insert(index, add_node)
    # add child node
    def append_child_node(self, add_node):
        add_node.set_parent(self)
        self.child_nodes.append(add_node)
    # find right insertion location
    def find_add_index(self, add_word):
        if len(self.keywords) == 0:
            return 0
        index = 0
        while True:
            if index >= len(self.keywords):
                break
            key = self.keywords[index].key
            if add_word.key < key:
                break
            index = index + 1
        return index
	#find the location of given keyword
    def find_loc(self,word):
        if len(self.keywords) == 0:
            return -1
        index = 0
        key=-1
        while True:
            if index >= len(self.keywords):
                break
            key = self.keywords[index].key
            if word < key:
                break
            index = index + 1
        if index==0:
            index=1
        index=index-1
        #print(index)
        if index+1>=len(self.keywords):
            return int(self.keywords[index].loc)
        if self.keywords[index].key==word or abs(int(word)-int(self.keywords[index].key))<abs(int(word)-int(self.keywords[index+1].key)):
            return int(self.keywords[index].loc)
        else:
            return int(self.keywords[index+1].loc)

    # insert data to right location (regardless of M)
    def blind_add(self, word: BKeyword) -> int:
        index = self.find_add_index(word)
        self.keywords.insert(index, word)
    def split(self):
        # split node
        parent, center_keyword, left_node, right_node = self.split_to_piece()
        # add two new nodes as parent, build relationship
        parent_add_index = parent.find_add_index(center_keyword)
        parent.insert_child_node(parent_add_index, right_node)
        parent.insert_child_node(parent_add_index, left_node)
        # remove itself
        if self in parent.child_nodes:
            parent.child_nodes.remove(self)
        parent.add_word(center_keyword, force=True)
        # redefine root
        root = self
        while root.get_parent() is not None:
            root = root.get_parent()
        global root_node
        root_node = root
    def split_to_piece(self):
        center_keyword = self.keywords[int((_M-1)/2)]
        if self.get_parent() is None:
            self.set_parent(BNode())
        left_node = BNode()
        right_node = BNode()
        for keyword in self.keywords:
            if keyword.key < center_keyword.key:
                left_node.keywords.append(keyword)
            elif keyword.key > center_keyword.key:
                right_node.keywords.append(keyword)
        for i in range(len(self.child_nodes)):
            if i <= int((len(self.child_nodes) - 1)/2):
                left_node.append_child_node(self.child_nodes[i])
            else:
                right_node.append_child_node(self.child_nodes[i])
        return self.get_parent(), center_keyword, left_node, right_node
    def add_word(self, keyword, force=False):


        if len(self.child_nodes) == 0 or force:
            self.blind_add(keyword)
            if len(self.keywords) == _M:
                self.split()
        else:

            index = self.find_add_index(keyword)
            if index >= len(self.child_nodes):
                index = index - 1
            self.child_nodes[index].add_word(keyword)


def print_tree(node):

    print("\n******************************")

    def dsc(node):
        s = ''
        for keyword in node.keywords:
            s=s+'['+str(keyword.key)+":"+str(keyword.loc)+"]"+","

        s = s[:-1]
        return s
    Logger.tree(node, 'child_nodes', dsc,  0)
    print("******************************")


def prepare():
    array = []
    number = 0
    for i in range(1000,3000):
        number = i + random.randint(1, 4)
        # number = number + 1
        array.append(number)
    shuffle(array)
    print(array)
    return array
def main():
    '''
	current_path=os.path.abspath(os.curdir)
	f=codecs.open(os.path.join(current_path,"data.csv"), "r", "utf-8")
	strlist=f.read().split("\n")
	f.close()
	list=[]
	for ele in strlist:
		temp=ele.split(",")
		if len(temp)!=2:
			continue
		list.append(temp[0]+":"+temp[1])
	root_node = BNode()
	for i in range(0,len(list)):
		temp=list[i].split(":")
		keyword = BKeyword(temp[0], temp[1])
		root_node.add_word(keyword)
	#print_tree(root_node)
	while True:
		a=input("Input key:")
		print("location: "+str(root_node.find_loc(a))+"\n")
    '''
    current_path=os.path.abspath(os.curdir)
    f=codecs.open(os.path.join(current_path,"data.csv"), "r", "utf-8")
    strlist=f.read().split("\n")
    f.close()
    list_key=[]
    list_res=[]
    for ele in strlist:
        temp=ele.split(",")
        if len(temp)!=2:
            continue
		#list.append(temp[0]+":"+temp[1])
        list_key.append(temp[0])
        list_res.append(temp[1])
    keys=np.array(list_key)
    res=np.array(list_res)
    trainkeys,testkeys,trainres,testres=train_test_split(keys,res,test_size=0.3)
    trainkeys=list(trainkeys)
    testkeys=list(testkeys)
    trainres=list(trainres)
    testres=list(testres)
    root_node=BNode()
    for i in range(0,len(trainkeys)):
        keyword=BKeyword(trainkeys[i],trainres[i])
        root_node.add_word(keyword)
    testpre=[]
    for i in range(0,len(testkeys)):
        testpre.append(root_node.find_loc(testkeys[i]))
    '''
    f=codecs.open(os.path.join(current_path,"1.csv"), "w", "utf-8")
    for i in range(0,len(testkeys)):
        f.write(str(testkeys[i])+","+str(testpre[i])+","+str(testres[i])+"\n")
    f.close()
    '''
    #print(testpre)
    mse=0.0#mean squared error
    for i in range(0,len(testkeys)):
        mse=mse+(int(testpre[i])-int(testres[i]))**2
    mse=mse/len(testkeys)
    print("MSE: ",round(mse,3))
    return
if __name__ == '__main__':
	main()
