# -*- coding: utf-8 -*-
"""learned_index_1dint.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zAansNiJACI0g66cygIon60DZ8WCEhxm

# Read dataset

## read data
"""

import codecs
import os
minkey=1000
maxkey=9999
keynum=3000
current_path=os.path.abspath(os.curdir)
f=codecs.open(os.path.join(current_path,"data.csv"), "r", "utf-8")
strlist=f.read().split("\n")
f.close()
trainkeys=[]
trainres=[]
for ele in strlist:
    temp=ele.split(",")
    if len(temp)!=2:
        continue
    trainkeys.append(int(temp[0]))
    trainres.append(int(temp[1]))
# f=codecs.open(os.path.join(current_path,"data_dev.csv"), "r", "utf-8")
# strlist=f.read().split("\n")
# f.close()
# devkeys=[]
# devres=[]
# for ele in strlist:
#     temp=ele.split(",")
#     if len(temp)!=2:
#         continue
#     devkeys.append(int(temp[0]))
#     devres.append(int(temp[1]))
f=codecs.open(os.path.join(current_path,"data_test.csv"), "r", "utf-8")
strlist=f.read().split("\n")
f.close()
testkeys=[]
testres=[]
for ele in strlist:
    temp=ele.split(",")
    if len(temp)!=2:
        continue
    testkeys.append(int(temp[0]))
    testres.append(int(temp[1]))

# It is very time and space consuming to build models based on the entire dataset
# Instead, we divide the dataset into 3 parts (training, dev, testing)
# We build and train models based on training set, and give index predictions based on testing set

# trainkeys.extend(devkeys)
# trainres.extend(devres)
# trainkeys.extend(testkeys)
# trainres.extend(testres)

print("training data size:",len(trainkeys))
# print("development data size:",len(devkeys))
print("testing data size:",len(testkeys))

"""# Build Models"""

trainpages=[]
for ele in trainres:
  trainpages.append(ele//100)
testpages=[]
for ele in testres:
  testpages.append(ele//100)

import numpy as np
X_train=np.array(trainkeys).reshape(-1,1)
Y_train=np.array(trainres).reshape(-1,1)
Z_train=np.array(trainpages).reshape(-1,1)
X_test=np.array(testkeys).reshape(-1,1)
Y_test=np.array(testres).reshape(-1,1)
Z_test=np.array(testpages).reshape(-1,1)

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

"""## B-Tree"""

# #This is a B-Tree model
# #reference: https://www.jianshu.com/p/c625a009e488
# from random import shuffle
# import random
# import os
# import codecs
# import numpy as np
# import math
# import time
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error 
# mse_BTree=0.0
# root_node = None
# # trainkeys=[]
# # trainres=[]
# # devkeys=[]
# # devres=[]
# # testkeys=[]
# # testres=[]
# class Logger(object):
#     @classmethod
#     def tree(cls, node, child_name, dsc, depth):
#         if depth == 0:
#             head = "|   " * depth
#             print(head + "+--" + dsc(node))
#             depth = depth + 1
#         for child in getattr(node, child_name):
#             head = "|   " * depth
#             print(head + "+--" + dsc(child))
#             cls.tree(child, child_name, dsc, depth + 1)
# class BKeyword(object):
#     def __init__(self, key, loc):
#         self.key = key
#         self.loc = loc
# class BNode(object):
#     def __init__(self,M):
#         self._parent: BNode = None
#         self.keywords = []
#         self.child_nodes = []
#         self.M=M
#     # set parent node
#     def set_parent(self, node):
#         self._parent = node
#         if node.get_parent() is None:
#             global root_node
#             root_node = node.get_parent()
#     # get parent node
#     def get_parent(self):
#         return self._parent
#     # add child node to right location
#     def insert_child_node(self, index, add_node):
#         add_node.set_parent(self)
#         self.child_nodes.insert(index, add_node)
#     # add child node
#     def append_child_node(self, add_node):
#         add_node.set_parent(self)
#         self.child_nodes.append(add_node)
#     # find right insertion location
#     def find_add_index(self, add_word):
#         if len(self.keywords) == 0:
#             return 0
#         index = 0
#         while True:
#             if index >= len(self.keywords):
#                 break
#             key = self.keywords[index].key
#             if add_word.key < key:
#                 break
#             index = index + 1
#         return index
# 	#find the location of given keyword
#     def find_loc(self,word):
#         if len(self.keywords) == 0:
#             return -1
#         index = 0
#         key=-1
#         while True:
#             if index >= len(self.keywords):
#                 break
#             key = self.keywords[index].key
#             if word < key:
#                 break
#             index = index + 1
#         if index==0:
#             index=1
#         index=index-1
#         #print(index)
#         if index+1>=len(self.keywords):
#             return int(self.keywords[index].loc)
#         if self.keywords[index].key==word or abs(int(word)-int(self.keywords[index].key))<abs(int(word)-int(self.keywords[index+1].key)):
#             return int(self.keywords[index].loc)
#         else:
#             return int(self.keywords[index+1].loc)
#     # insert data to right location (regardless of M)
#     def blind_add(self, word: BKeyword) -> int:
#         index = self.find_add_index(word)
#         self.keywords.insert(index, word)
#     def split(self):
#         # split node
#         parent, center_keyword, left_node, right_node = self.split_to_piece()
#         # add two new nodes as parent, build relationship
#         parent_add_index = parent.find_add_index(center_keyword)
#         parent.insert_child_node(parent_add_index, right_node)
#         parent.insert_child_node(parent_add_index, left_node)
#         # remove itself
#         if self in parent.child_nodes:
#             parent.child_nodes.remove(self)
#         parent.add_word(center_keyword, force=True)
#         # redefine root
#         root = self
#         while root.get_parent() is not None:
#             root = root.get_parent()
#         global root_node
#         root_node = root
#     def split_to_piece(self):
#         center_keyword = self.keywords[int((self.M-1)/2)]
#         if self.get_parent() is None:
#             self.set_parent(BNode(self.M))
#         left_node = BNode(self.M)
#         right_node = BNode(self.M)
#         for keyword in self.keywords:
#             if keyword.key < center_keyword.key:
#                 left_node.keywords.append(keyword)
#             elif keyword.key > center_keyword.key:
#                 right_node.keywords.append(keyword)
#         for i in range(len(self.child_nodes)):
#             if i <= int((len(self.child_nodes) - 1)/2):
#                 left_node.append_child_node(self.child_nodes[i])
#             else:
#                 right_node.append_child_node(self.child_nodes[i])
#         return self.get_parent(), center_keyword, left_node, right_node
#     def add_word(self, keyword, force=False):
#         if len(self.child_nodes) == 0 or force:
#             self.blind_add(keyword)
#             if len(self.keywords) == self.M:
#                 self.split()
#         else:

#             index = self.find_add_index(keyword)
#             if index >= len(self.child_nodes):
#                 index = index - 1
#             self.child_nodes[index].add_word(keyword)
# def B_Tree_Model():
#     # print("Simple B-Tree Model")
#     t1=time.time()
#     root_node=BNode(4)
#     for i in range(0,len(trainkeys)):
#         keyword=BKeyword(trainkeys[i],trainres[i])
#         root_node.add_word(keyword)
#     t2=time.time()
#     time_interval=t2-t1
#     print("time interval for building model:"+str(time_interval*1000)+" ms")
#     # devpre=[]
#     # for i in range(0,len(devkeys)):
#     #     devpre.append(root_node.find_loc(devkeys[i]))
#     # global mse_BTree
#     # mse_BTree=mean_squared_error(devres,devpre)
#     # print("log MSE dev: ",round(math.log(1+mse_BTree,2),3))
#     t1=time.time()
#     testpre=[]
#     for i in range(0,len(testkeys)):
#         testpre.append(root_node.find_loc(testkeys[i]))
#     t2=time.time()
#     time_interval=t2-t1
#     print("time interval for indexing data :"+str(time_interval*1000)+" ms")
#     print("average time interval for indexing data :"+str(time_interval/len(testkeys)*1000)+" ms")
#     # print("log MSE test: ",round(math.log(1+mean_squared_error(testres,testpre),2),3))
#     return
# if __name__ == '__main__':
#     # f=codecs.open(os.path.join(current_path,"data_train.csv"), "r", "utf-8")
#     # strlist=f.read().split("\n")
#     # f.close()
#     # trainkeys=[]
#     # trainres=[]
#     # for ele in strlist:
#     #     temp=ele.split(",")
#     #     if len(temp)!=2:
#     #         continue
#     #     trainkeys.append(int(temp[0]))
#     #     trainres.append(int(temp[1]))
#     # f=codecs.open(os.path.join(current_path,"data_dev.csv"), "r", "utf-8")
#     # strlist=f.read().split("\n")
#     # f.close()
#     # devkeys=[]
#     # devres=[]
#     # for ele in strlist:
#     #     temp=ele.split(",")
#     #     if len(temp)!=2:
#     #         continue
#     #     devkeys.append(int(temp[0]))
#     #     devres.append(int(temp[1]))
#     # f=codecs.open(os.path.join(current_path,"data_test.csv"), "r", "utf-8")
#     # strlist=f.read().split("\n")
#     # f.close()
#     # testkeys=[]
#     # testres=[]
#     # for ele in strlist:
#     #     temp=ele.split(",")
#     #     if len(temp)!=2:
#     #         continue
#     #     testkeys.append(int(temp[0]))
#     #     testres.append(int(temp[1]))
#     B_Tree_Model()

import time
# ref: https://peefy.github.io/blog/2018/06/10/Python-BTree/
class BTreeNode:
    '''
    B树结点
    '''
    def __init__(self, n = 0, isleaf = True):
        '''
        B树结点

        Args
        ===
        `n` : 结点包含关键字的数量

        `isleaf` : 是否是叶子节点

        '''
        # 结点包含关键字的数量
        self.n = n
        # 关键字的值数组
        self.keys = []
        # 子结点数组
        self.children = []
        # 是否是叶子节点
        self.isleaf = isleaf

    def __str__(self):

        returnStr = 'keys:['
        for i in range(self.n):
            returnStr += str(self.keys[i]) + ' '
        returnStr += '];childrens:['
        for child in self.children:
            returnStr += str(child) + ';'
        returnStr += ']\r\n'
        return returnStr

    def diskread(self):
        '''
        磁盘读
        '''
        pass

    def diskwrite(self):
        '''
        磁盘写
        '''
        pass

    @classmethod
    def allocate_node(self, key_max):
        '''
        在O(1)时间内为一个新结点分配一个磁盘页

        假定由ALLOCATE-NODE所创建的结点无需做DISK-READ，因为磁盘上还没有关于该结点的有用信息

        Return
        ===
        `btreenode` : 分配的B树结点

        Example
        ===
        ```python
        btreenode = BTreeNode.allocate_node()
        ```
        '''
        node = BTreeNode()
        child_max = key_max + 1
        for i in range(key_max):
            node.keys.append(None)
        for i in range(child_max):
            node.children.append(None)
        return node

class BTree:
    '''
    B树
    '''
    def __init__(self, m = 3):
        '''
        B树的定义
        '''
        # B树的最小度数
        self.M = m
        # 节点包含关键字的最大个数
        self.KEY_MAX = 2 * self.M - 1
        # 非根结点包含关键字的最小个数
        self.KEY_MIN = self.M - 1
        # 子结点的最大个数
        self.CHILD_MAX = self.KEY_MAX + 1
        # 子结点的最小个数
        self.CHILD_MIN = self.KEY_MIN + 1
        # 根结点
        self.root: BTreeNode = None

    def __new_node(self):
        '''
        创建新的B树结点
        '''
        return BTreeNode.allocate_node(self.KEY_MAX)

    def insert(self, key):
        '''
        向B树中插入新结点`key`  
        '''
        # 检查关键字是否存在
        if self.contain(key) == True:
            return False
        else:
            # 检查是否为空树
            if self.root is None:
                node = self.__new_node()
                node.diskwrite()
                self.root = node    
            # 检查根结点是否已满      
            if self.root.n == self.KEY_MAX:
                # 创建新的根结点
                pNode = self.__new_node()
                pNode.isleaf = False
                pNode.children[0] = self.root
                self.__split_child(pNode, 0, self.root)
                # 更新结点指针
                self.root = pNode
            self.__insert_non_full(self.root, key)
            return True

    def remove(self, key): 
        '''
        从B中删除结点`key`
        '''      
        # 如果关键字不存在
        if not self.search(self.root, key):
            return False
        # 特殊情况处理
        if self.root.n == 1:
            if self.root.isleaf == True:
                self.clear()
            else:
                pChild1 = self.root.children[0]
                pChild2 = self.root.children[1]
                if pChild1.n == self.KEY_MIN and pChild2.n == self.KEY_MIN:
                    self.__merge_child(self.root, 0)
                    self.__delete_node(self.root)
                    self.root = pChild1
        self.__recursive_remove(self.root, key)
        return True
    
    def display(self):
        '''
        打印树的关键字  
        '''
        self.__display_in_concavo(self.root, self.KEY_MAX * 10)

    def contain(self, key):
        '''
        检查该`key`是否存在于B树中  
        '''
        self.__search(self.root, key)

    def clear(self):
        '''
        清空B树  
        '''
        self.__recursive_clear(self.root)
        self.root = None

    def __recursive_clear(self, pNode : BTreeNode):
        '''
        删除树  
        '''
        if pNode is not None:
            if not pNode.isleaf:
                for i in range(pNode.n):
                    self.__recursive_clear(pNode.children[i])
            self.__delete_node(pNode)

    def __delete_node(self, pNode : BTreeNode):
        '''
        删除节点 
        '''
        if pNode is not None:
            pNode = None
    
    def __search(self, pNode : BTreeNode, key):
        '''
        查找关键字  
        '''
        # 检测结点是否为空，或者该结点是否为叶子节点
        if pNode is None:
            return False
        else:
            i = 0
            # 找到使key < pNode.keys[i]成立的最小下标
            while i < pNode.n and key > pNode.keys[i]:
                i += 1
            if i < pNode.n and key == pNode.keys[i]:
                return True
            else:
                # 检查该结点是否为叶子节点
                if pNode.isleaf == True:
                    return False
                else:
                    return self.__search(pNode.children[i], key)

    def __split_child(self, pParent : BTreeNode, nChildIndex, pChild : BTreeNode):
        '''
        分裂子节点
        '''
        # 将pChild分裂成pLeftChild和pChild两个结点
        pRightNode = self.__new_node()  # 分裂后的右结点
        pRightNode.isleaf = pChild.isleaf
        pRightNode.n = self.KEY_MIN
        # 拷贝关键字的值
        for i in range(self.KEY_MIN):
            pRightNode.keys[i] = pChild.keys[i + self.CHILD_MIN]
        # 如果不是叶子结点，就拷贝孩子结点指针
        if not pChild.isleaf:
            for i in range(self.CHILD_MIN):
                pRightNode.children[i] = pChild.children[i + self.CHILD_MIN]
        # 更新左子树的关键字个数
        pChild.n = self.KEY_MIN
        # 将父结点中的pChildIndex后的所有关键字的值和子树指针向后移动一位
        for i in range(nChildIndex, pParent.n):
            j = pParent.n + nChildIndex - i
            pParent.children[j + 1] = pParent.children[j]
            pParent.keys[j] = pParent.keys[j - 1]
        # 更新父结点的关键字个数
        pParent.n += 1
        # 存储右子树指针
        pParent.children[nChildIndex + 1] = pRightNode
        # 把结点的中间值提到父结点
        pParent.keys[nChildIndex] = pChild.keys[self.KEY_MIN]
        pChild.diskwrite()
        pRightNode.diskwrite()
        pParent.diskwrite()
    
    def __insert_non_full(self, pNode: BTreeNode, key):
        '''
        在非满节点中插入关键字
        '''
        # 获取结点内关键字个数
        i = pNode.n
        # 如果pNode是叶子结点
        if pNode.isleaf == True:
            # 从后往前 查找关键字的插入位置
            while i > 0 and key < pNode.keys[i - 1]:
                # 向后移位
                pNode.keys[i] = pNode.keys[i - 1]
                i -= 1
            # 插入关键字的值
            pNode.keys[i] = key
            # 更新结点关键字的个数
            pNode.n += 1
            pNode.diskwrite()
        # pnode是内结点
        else:
            # 从后往前 查找关键字的插入的子树
            while i > 0 and key < pNode.keys[i - 1]:
                i -= 1
            # 目标子树结点指针
            pChild = pNode.children[i]
            pNode.children[i].diskread()
            # 子树结点已经满了
            if pChild.n == self.KEY_MAX:
                # 分裂子树结点
                self.__split_child(pNode, i, pChild)
                # 确定目标子树
                if key > pNode.keys[i]:
                    pChild = pNode.children[i + 1]
            # 插入关键字到目标子树结点
            self.__insert_non_full(pChild, key)

    def __display_in_concavo(self, pNode: BTreeNode, count):
        '''
        用括号打印树 
        '''
        if pNode is not None:
            i = 0
            j = 0
            for i in range(pNode.n):
                if not pNode.isleaf:
                    self.__display_in_concavo(pNode.children[i], count - 2)
                for j in range(-1, count):
                    k = count - j - 1
                    print('-', end='')
                print(pNode.keys[i])
            if not pNode.isleaf:
                self.__display_in_concavo(pNode.children[i], count - 2)

    def __merge_child(self, pParent: BTreeNode, index):
        '''
        合并两个子结点
        '''
        pChild1 = pParent.children[index]
        pChild2 = pParent.children[index + 1]
        # 将pChild2数据合并到pChild1
        pChild1.n = self.KEY_MAX
        # 将父结点index的值下移
        pChild1.keys[self.KEY_MIN] = pParent.keys[index]
        for i in range(self.KEY_MIN):
            pChild1.keys[i + self.KEY_MIN + 1] = pChild2.keys[i]
        if not pChild1.isleaf:
            for i in range(self.CHILD_MIN):
                pChild1.children[i + self.CHILD_MIN] = pChild2.children[i]
        # 父结点删除index的key，index后的往前移一位
        pParent.n -= 1
        for i in range(index, pParent.n):
            pParent.keys[i] = pParent.keys[i + 1]
            pParent.children[i + 1] = pParent.children[i + 2]
        # 删除pChild2
        self.__delete_node(pChild2)

    def __recursive_remove(self, pNode: BTreeNode, key):
        '''
        递归的删除关键字`key`  
        '''
        i = 0
        while i < pNode.n and key > pNode.keys[i]:
            i += 1
        # 关键字key在结点pNode
        if i < pNode.n and key == pNode.keys[i]:
            # pNode是个叶结点
            if pNode.isleaf == True:
                # 从pNode中删除k
                for j in range(i, pNode.n):
                    pNode.keys[j] = pNode.keys[j + 1]
                return
            # pNode是个内结点
            else:
                # 结点pNode中前于key的子结点
                pChildPrev = pNode.children[i]
                # 结点pNode中后于key的子结点
                pChildNext = pNode.children[i + 1]
                if pChildPrev.n >= self.CHILD_MIN:
                    # 获取key的前驱关键字
                    prevKey = self.predecessor(pChildPrev)
                    self.__recursive_remove(pChildPrev, prevKey)
                    # 替换成key的前驱关键字
                    pNode.keys[i] = prevKey
                    return
                # 结点pChildNext中至少包含CHILD_MIN个关键字
                elif pChildNext.n >= self.CHILD_MIN:
                    # 获取key的后继关键字
                    nextKey = self.successor(pChildNext)
                    self.__recursive_remove(pChildNext, nextKey)
                    # 替换成key的后继关键字
                    pNode.keys[i] = nextKey
                    return
                # 结点pChildPrev和pChildNext中都只包含CHILD_MIN-1个关键字
                else:
                    self.__merge_child(pNode, i)
                    self.__recursive_remove(pChildPrev, key)
        # 关键字key不在结点pNode中
        else:
            # 包含key的子树根结点
            pChildNode = pNode.children[i]
            # 只有t-1个关键字
            if pChildNode.n == self.KEY_MAX:
                # 左兄弟结点
                pLeft = None
                # 右兄弟结点
                pRight = None
                # 左兄弟结点
                if i > 0:
                    pLeft = pNode.children[i - 1]
                # 右兄弟结点
                if i < pNode.n:
                    pRight = pNode.children[i + 1]
                j = 0
                if pLeft is not None and pLeft.n >= self.CHILD_MIN:
                    # 父结点中i-1的关键字下移至pChildNode中
                    for j in range(pChildNode.n):
                        k = pChildNode.n - j
                        pChildNode.keys[k] = pChildNode.keys[k - 1]
                    pChildNode.keys[0] = pNode.keys[i - 1]
                    if not pLeft.isleaf:
                        # pLeft结点中合适的子女指针移到pChildNode中
                        for j in range(pChildNode.n + 1):
                            k = pChildNode.n + 1 - j
                            pChildNode.children[k] = pChildNode.children[k - 1]
                        pChildNode.children[0] = pLeft.children[pLeft.n]
                    pChildNode.n += 1
                    pNode.keys[i] = pLeft.keys[pLeft.n - 1]
                    pLeft.n -= 1
                # 右兄弟结点至少有CHILD_MIN个关键字
                elif pRight is not None and pRight.n >= self.CHILD_MIN:
                    # 父结点中i的关键字下移至pChildNode中
                    pChildNode.keys[pChildNode.n] = pNode.keys[i]
                    pChildNode.n += 1
                    # pRight结点中的最小关键字上升到pNode中
                    pNode.keys[i] = pRight.keys[0]
                    pRight.n -= 1
                    for j in range(pRight.n):
                        pRight.keys[j] = pRight.keys[j + 1]
                    if not pRight.isleaf:
                        # pRight结点中合适的子女指针移动到pChildNode中
                        pChildNode.children[pChildNode.n] = pRight.children[0]
                        for j in range(pRight.n):
                            pRight.children[j] = pRight.children[j + 1]
                # 左右兄弟结点都只包含CHILD_MIN-1个结点
                elif pLeft is not None:
                    self.__merge_child(pNode, i - 1)
                    pChildNode = pLeft
                # 与右兄弟合并
                elif pRight is not None:
                    self.__merge_child(pNode, i)
            self.__recursive_remove(pChildNode, key)

    def predecessor(self, pNode: BTreeNode):
        '''
        前驱关键字
        '''
        while not pNode.isleaf:
            pNode = pNode.children[pNode.n]
        return pNode.keys[pNode.n - 1]

    def successor(self, pNode: BTreeNode):
        '''
        后继关键字
        '''
        while not pNode.isleaf:
            pNode = pNode.children[0]
        return pNode.keys[0]

def test():
    '''
    test class `BTree` and class `BTreeNode`
    '''
    tree = BTree(3)
    # tree.insert(11)
    # tree.insert(3)
    # tree.insert(1)
    # tree.insert(4)
    # tree.insert(33)
    # tree.insert(13)
    # tree.insert(63)
    # tree.insert(43)
    # tree.insert(2)
    # print(tree.root)
    # tree.display()
    # tree.clear()
    # tree = BTree(2)
    # tree.insert(11)
    # tree.insert(3)
    # tree.insert(1)
    # tree.insert(4)
    # tree.insert(33)
    # tree.insert(13)
    # tree.insert(63)
    # tree.insert(43)
    # tree.insert(2)
    # print(tree.root)
    # tree.display()
    t1=time.time()
    for i in range(0,len(trainkeys)):
        tree.insert(trainkeys[i])
    t2=time.time()
    time_interval=t2-t1
    print("time interval for building model:"+str(time_interval*1000)+" ms")
    ret1=time_interval*1000
    t1=time.time()
    # testpre=[]
    for i in range(0,len(testkeys)):
        tree.contain(testkeys[i])
    t2=time.time()
    time_interval=t2-t1
    print("time interval for indexing data :"+str(time_interval*1000)+" ms")
    print("average time interval for indexing data :"+str(time_interval/len(testkeys)*1000)+" ms")
    ret2=time_interval*1000
    ret3=time_interval/len(testkeys)*1000
    return (ret1,ret2,ret3)

if __name__ == '__main__':
    avg_a=0.0
    avg_b=0.0
    avg_c=0.0
    counting=20
    for i in range(0,20):
        (a,b,c)=test()
        avg_a+=a
        avg_b+=b
        avg_c+=c
    avg_a=avg_a/counting
    avg_b=avg_b/counting
    avg_c=avg_c/counting
    print("average times (ms):",avg_a,avg_b,avg_c)
else:
    pass

"""## Linear Regression"""

from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error 
import math
import time
# print("Linear Regression Model")
def test():
  t1=time.time()
  reg = LinearRegression()
  reg.fit(X_train,Y_train)
  t2=time.time()
  time_interval=t2-t1
  print("time interval for building model:"+str(time_interval*1000)+" ms")
  ret1=time_interval*1000
  # devpre=reg.predict(np.array(devkeys).reshape(-1,1)).reshape(1,-1).tolist()[0]
  # for i in range(0,len(devpre)):
  #     devpre[i]=abs(int(devpre[i]))
  # mse_LR=mean_squared_error(devres,devpre)
  # print("MSE dev: ",mse_LR)
  t1=time.time()
  testpre=reg.predict(X_test).reshape(1,-1).tolist()[0]
  for i in range(0,len(testpre)):
    testpre[i]=abs(int(testpre[i]))
  t2=time.time()
  time_interval=t2-t1
  print("time interval for indexing data :"+str(time_interval*1000)+" ms")
  print("average time interval for indexing data :"+str(time_interval/len(testkeys)*1000)+" ms")
  ret2=time_interval*1000
  ret3=time_interval/len(testkeys)*1000
  # print("log MSE test: ",round(math.log(1+mean_squared_error(testres,testpre),2),3))
  t1=time.time()
  count_error=0
  for i in range(0,len(testpre)):
    estimated_loc=testpre[i]
    correct_res=testkeys[i]
    if estimated_loc>=0 and estimated_loc<len(trainkeys):
      finding_res=trainkeys[estimated_loc]
    elif estimated_loc<0:
      finding_res=trainkeys[0]
    else:
      finding_res=trainkeys[len(trainkeys)-1]
    if finding_res!=correct_res:
      count_error+=1
    begin=0
    end=len(trainkeys)-1
    while finding_res!=correct_res:
      
      # # print(finding_res,correct_res)
      # if count_error>30:
      #   return
      if finding_res<correct_res:
        begin=estimated_loc
        # end=len(trainkeys)-1
        estimated_loc=(begin+end)//2
        if estimated_loc>=0 and estimated_loc<len(trainkeys):
          finding_res=trainkeys[estimated_loc]
        elif estimated_loc<0:
          finding_res=trainkeys[0]
        else:
          finding_res=trainkeys[len(trainkeys)-1]
      else:
        # begin=0
        end=estimated_loc
        estimated_loc=(begin+end)//2
        if estimated_loc>=0 and estimated_loc<len(trainkeys):
          finding_res=trainkeys[estimated_loc]
        elif estimated_loc<0:
          finding_res=trainkeys[0]
        else:
          finding_res=trainkeys[len(trainkeys)-1]
  t2=time.time()
  time_interval=t2-t1
  print("time interval for error correction :"+str(time_interval*1000)+" ms")
  print("average time interval for error correction :"+str(time_interval/count_error*1000)+" ms")
  ret4=time_interval*1000
  ret5=time_interval/len(testkeys)*1000
  return (ret1,ret2,ret3,ret4,ret5)
avg_a=0.0
avg_b=0.0
avg_c=0.0
avg_d=0.0
avg_e=0.0
counting=20
for i in range(0,20):
  (a,b,c,d,e)=test()
  avg_a+=a
  avg_b+=b
  avg_c+=c
  avg_d+=d
  avg_e+=e
avg_a=avg_a/counting
avg_b=avg_b/counting
avg_c=avg_c/counting
avg_d=avg_d/counting
avg_e=avg_e/counting
print("average times (ms):",avg_a,avg_b,avg_c,avg_d,avg_e)

"""## Ridge Regression"""

from sklearn.linear_model import Ridge
import numpy as np
from sklearn.metrics import mean_squared_error 
import math
import time
def test():
  t1=time.time()
  reg = Ridge(alpha=1.0)
  reg.fit(X_train,Y_train)
  t2=time.time()
  time_interval=t2-t1
  print("time interval for building model:"+str(time_interval*1000)+" ms")
  ret1=time_interval*1000
  # devpre=reg.predict(np.array(devkeys).reshape(-1,1)).reshape(1,-1).tolist()[0]
  # for i in range(0,len(devpre)):
  #     devpre[i]=abs(int(devpre[i]))
  # mse_LR=mean_squared_error(devres,devpre)
  # print("MSE dev: ",mse_LR)
  t1=time.time()
  testpre=reg.predict(X_test).reshape(1,-1).tolist()[0]
  for i in range(0,len(testpre)):
    testpre[i]=abs(int(testpre[i]))
  t2=time.time()
  time_interval=t2-t1
  print("time interval for indexing data :"+str(time_interval*1000)+" ms")
  print("average time interval for indexing data :"+str(time_interval/len(testkeys)*1000)+" ms")
  ret2=time_interval*1000
  ret3=time_interval/len(testkeys)*1000
  # print("log MSE test: ",round(math.log(1+mean_squared_error(testres,testpre),2),3))
  t1=time.time()
  count_error=0
  for i in range(0,len(testpre)):
    estimated_loc=testpre[i]
    correct_res=testkeys[i]
    if estimated_loc>=0 and estimated_loc<len(trainkeys):
      finding_res=trainkeys[estimated_loc]
    elif estimated_loc<0:
      finding_res=trainkeys[0]
    else:
      finding_res=trainkeys[len(trainkeys)-1]
    if finding_res!=correct_res:
      count_error+=1
    begin=0
    end=len(trainkeys)-1
    while finding_res!=correct_res:
      
      # # print(finding_res,correct_res)
      # if count_error>30:
      #   return
      if finding_res<correct_res:
        begin=estimated_loc
        # end=len(trainkeys)-1
        estimated_loc=(begin+end)//2
        if estimated_loc>=0 and estimated_loc<len(trainkeys):
          finding_res=trainkeys[estimated_loc]
        elif estimated_loc<0:
          finding_res=trainkeys[0]
        else:
          finding_res=trainkeys[len(trainkeys)-1]
      else:
        # begin=0
        end=estimated_loc
        estimated_loc=(begin+end)//2
        if estimated_loc>=0 and estimated_loc<len(trainkeys):
          finding_res=trainkeys[estimated_loc]
        elif estimated_loc<0:
          finding_res=trainkeys[0]
        else:
          finding_res=trainkeys[len(trainkeys)-1]
  t2=time.time()
  time_interval=t2-t1
  print("time interval for error correction :"+str(time_interval*1000)+" ms")
  print("average time interval for error correction :"+str(time_interval/count_error*1000)+" ms")
  ret4=time_interval*1000
  ret5=time_interval/count_error*1000
  return (ret1,ret2,ret3,ret4,ret5)
avg_a=0.0
avg_b=0.0
avg_c=0.0
avg_d=0.0
avg_e=0.0
counting=20
for i in range(0,20):
  (a,b,c,d,e)=test()
  avg_a+=a
  avg_b+=b
  avg_c+=c
  avg_d+=d
  avg_e+=e
avg_a=avg_a/counting
avg_b=avg_b/counting
avg_c=avg_c/counting
avg_d=avg_d/counting
avg_e=avg_e/counting
print("average times (ms):",avg_a,avg_b,avg_c,avg_d,avg_e)

"""## KNN"""

from sklearn.neighbors import KNeighborsClassifier
import time
import numpy as np
from sklearn.metrics import classification_report
def test():
  t1=time.time()
  neigh = KNeighborsClassifier(n_neighbors=9)
  neigh.fit(X_train,Z_train)
  t2=time.time()
  time_interval=t2-t1
  # devpre=neigh.predict(X_dev)#.reshape(1,-1).tolist()[0]
  # print(classification_report(Y_dev,devpre))
  print("time interval for building model:"+str(time_interval*1000)+" ms")
  ret1=time_interval*1000
  t1=time.time()
  testpre=neigh.predict(X_test).reshape(1,-1).tolist()[0]
  t2=time.time()
  time_interval=t2-t1
  print("time interval for indexing data :"+str(time_interval*1000)+" ms")
  print("average time interval for indexing data :"+str(time_interval/len(testkeys)*1000)+" ms")
  ret2=time_interval*1000
  ret3=time_interval/len(testkeys)*1000
  t1=time.time()
  for i in range(0,len(testpre)):
    estimated_page=testpre[i]
    correct_res=testkeys[i]
    if correct_res in range(estimated_page*100,estimated_page*100+100):
      pass
    else:
      estimated_page=correct_res//100
    begin=estimated_page*100
    end=estimated_page*100+100
    while begin<end:
      middle=(begin+end)//2
      if middle==correct_res:
        estimated_loc=middle
        break
      elif middle<correct_res:
        begin=middle
      else:
        end=middle
  t2=time.time()
  time_interval=t2-t1
  print("time interval for error correction :"+str(time_interval*1000)+" ms")
  print("average time interval for error correction :"+str(time_interval/len(testkeys)*1000)+" ms")
  ret4=time_interval*1000
  ret5=time_interval/len(testkeys)*1000
  return (ret1,ret2,ret3,ret4,ret5)
avg_a=0.0
avg_b=0.0
avg_c=0.0
avg_d=0.0
avg_e=0.0
counting=20
for i in range(0,20):
  (a,b,c,d,e)=test()
  avg_a+=a
  avg_b+=b
  avg_c+=c
  avg_d+=d
  avg_e+=e
avg_a=avg_a/counting
avg_b=avg_b/counting
avg_c=avg_c/counting
avg_d=avg_d/counting
avg_e=avg_e/counting
print("average times (ms):",avg_a,avg_b,avg_c,avg_d,avg_e)

"""## Naive Bayes"""

from sklearn.naive_bayes import GaussianNB
import time
import numpy as np
from sklearn.metrics import classification_report
def test():
  t1=time.time()
  NB = GaussianNB()
  NB.fit(X_train,Z_train)
  t2=time.time()
  time_interval=t2-t1
  # devpre=NB.predict(X_dev)#.reshape(1,-1).tolist()[0]
  # print(classification_report(Y_dev,devpre))
  print("time interval for building model:"+str(time_interval*1000)+" ms")
  ret1=time_interval*1000
  t1=time.time()
  testpre=NB.predict(X_test).reshape(1,-1).tolist()[0]
  t2=time.time()
  time_interval=t2-t1
  print("time interval for indexing data :"+str(time_interval*1000)+" ms")
  print("average time interval for indexing data :"+str(time_interval/len(testkeys)*1000)+" ms")
  ret2=time_interval*1000
  ret3=time_interval/len(testkeys)*1000
  t1=time.time()
  for i in range(0,len(testpre)):
    estimated_page=testpre[i]
    correct_res=testkeys[i]
    if correct_res in range(estimated_page*100,estimated_page*100+100):
      pass
    else:
      estimated_page=correct_res//100
    begin=estimated_page*100
    end=estimated_page*100+100
    while begin<end:
      middle=(begin+end)//2
      if middle==correct_res:
        estimated_loc=middle
        break
      elif middle<correct_res:
        begin=middle
      else:
        end=middle
  t2=time.time()
  time_interval=t2-t1
  print("time interval for error correction :"+str(time_interval*1000)+" ms")
  print("average time interval for error correction :"+str(time_interval/len(testkeys)*1000)+" ms")
  ret4=time_interval*1000
  ret5=time_interval/len(testkeys)*1000
  return (ret1,ret2,ret3,ret4,ret5)
avg_a=0.0
avg_b=0.0
avg_c=0.0
avg_d=0.0
avg_e=0.0
counting=20
for i in range(0,20):
  (a,b,c,d,e)=test()
  avg_a+=a
  avg_b+=b
  avg_c+=c
  avg_d+=d
  avg_e+=e
avg_a=avg_a/counting
avg_b=avg_b/counting
avg_c=avg_c/counting
avg_d=avg_d/counting
avg_e=avg_e/counting
print("average times (ms):",avg_a,avg_b,avg_c,avg_d,avg_e)

"""## Decision Tree"""

from sklearn import tree
import time
import numpy as np
from sklearn.metrics import classification_report
def test():
  t1=time.time()
  dtree = tree.DecisionTreeClassifier()
  dtree.fit(X_train,Z_train)
  t2=time.time()
  time_interval=t2-t1
  # devpre=tree.predict(X_dev)#.reshape(1,-1).tolist()[0]
  # print(classification_report(Y_dev,devpre))
  print("time interval for building model:"+str(time_interval*1000)+" ms")
  ret1=time_interval*1000
  t1=time.time()
  testpre=dtree.predict(X_test).reshape(1,-1).tolist()[0]
  t2=time.time()
  time_interval=t2-t1
  print("time interval for indexing data :"+str(time_interval*1000)+" ms")
  print("average time interval for indexing data :"+str(time_interval/len(testkeys)*1000)+" ms")
  ret2=time_interval*1000
  ret3=time_interval/len(testkeys)*1000
  t1=time.time()
  for i in range(0,len(testpre)):
    estimated_page=testpre[i]
    correct_res=testkeys[i]
    if correct_res in range(estimated_page*100,estimated_page*100+100):
      pass
    else:
      estimated_page=correct_res//100
    begin=estimated_page*100
    end=estimated_page*100+100
    while begin<end:
      middle=(begin+end)//2
      if middle==correct_res:
        estimated_loc=middle
        break
      elif middle<correct_res:
        begin=middle
      else:
        end=middle
  t2=time.time()
  time_interval=t2-t1
  print("time interval for error correction :"+str(time_interval*1000)+" ms")
  print("average time interval for error correction :"+str(time_interval/len(testkeys)*1000)+" ms")
  ret4=time_interval*1000
  ret5=time_interval/len(testkeys)*1000
  return (ret1,ret2,ret3,ret4,ret5)
avg_a=0.0
avg_b=0.0
avg_c=0.0
avg_d=0.0
avg_e=0.0
counting=20
for i in range(0,20):
  (a,b,c,d,e)=test()
  avg_a+=a
  avg_b+=b
  avg_c+=c
  avg_d+=d
  avg_e+=e
avg_a=avg_a/counting
avg_b=avg_b/counting
avg_c=avg_c/counting
avg_d=avg_d/counting
avg_e=avg_e/counting
print("average times (ms):",avg_a,avg_b,avg_c,avg_d,avg_e)