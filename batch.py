#! c:/Python26/python.exe
# -*- coding: utf-8 -*-

import os
import sys
import time
from stat import *
import pickle
import math
import random
import numpy as np
import csv

class Batch:
    def __init__(self, data_size, type=0, class_num=10):
        # type
        # 0 : classification
        # 1 : regression
        self.type = type
        self.class_num = class_num
        self.data_size = data_size
        self.quantize = False

    def load(self, path):
        data = None
        size = 0
        try:
            with open(path, mode='rb') as f:
                data = pickle.load(f)
            #
        except:
            print("load error", path)
            return None, 0
        #
        if type(data)==list:
            size = len(data)
        elif type(data)==np.ndarray:
            size = data.shape[0]
        #
        print(type(data), size)
        return data, size
        
    def load_data(self, path): # array
        self.data_path = path
        self.data_array, self.batch_size = self.load(self.data_path)
        print(self.batch_size)
            
    def load_label(self, path): # list
        self.label_path = path
        self.label_list, self.batch_size = self.load(self.label_path)
        print(self.batch_size)
        
    def prepare_batch(self, scale=True):#, quantize=False):
        self.label_array = np.zeros((self.batch_size, self.class_num), dtype=np.float32)
        if self.quantize:
            self.data_array_q = np.zeros((self.batch_size, self.data_size), dtype=np.uint8)
        #
        
        for i in range(self.batch_size):
            if scale==True:
                # scale : 0.0 - 1.0
                self.data_array[i] = self.data_array[i] / 255.0
            #
            if self.quantize:
                size = self.data_array[i].shape[0]
                for j in range(size):
                    q = self.data_array[i][j]
                    if q>=0.0 and q<=0.0625:
                        q = 4 # 0.0
                    elif q>=0.0625 and q<=0.1875:
                        q = 5 # 0.125
                    elif q>0.1875 and q<=0.375:
                        q = 6 # 0.25
                    elif q>0.375 and q<=0.75:
                        q =7 # 0.5
                    elif q>0.75 and q<=1.0:
                        q = 8 #1.0
                    #
                    self.data_array_q[i][j] = q
                # for j
            #
            k = int(self.label_list[i])
            self.label_array[i][k] = 1.0
        #
        if self.quantize:
            print(self.data_array_q[0][0])
        #
    
    def prepare_mini_batch(self, size):
        self.mini_batch_size = size
        self.mini_batch_num = int(self.batch_size / self.mini_batch_size)
        self.mini_batch_idx_list = []
        for i in range(self.batch_size):
            self.mini_batch_idx_list.append(i)
        #
        print("self.mini_batch_num", self.mini_batch_num)
        #random.shuffle(self.mini_batch_idx_list)
        
    def shuffle_mini_batch(self):
        random.shuffle(self.mini_batch_idx_list)
    
    def get_batch(self, size, offset=0):
        data_array = np.zeros((size, self.data_size), dtype=np.float32)
        label_array = np.zeros((size, self.class_num), dtype=np.float32)
        for i in range(size):
            data_array[i] = self.data_array[offset + i]
            label_array[i] = self.label_array[offset + i]
        #
        return data_array, label_array
    
    def get_mini_batch(self, offset, quantize=False):
        #print("batch::get_mini_batch(), offset", offset)
        #print("self.mini_batch_size", self.mini_batch_size)
        #print("self.mini_batch_idx_list", len(self.mini_batch_idx_list))
        if self.quantize:
            data_array = np.zeros((self.mini_batch_size, self.data_size), dtype=np.uint8)
        else:
            data_array = np.zeros((self.mini_batch_size, self.data_size), dtype=np.float32)
        #
        label_array = np.zeros((self.mini_batch_size, self.class_num), dtype=np.float32)
        for i in range(self.mini_batch_size):
            idx = self.mini_batch_idx_list[offset + i]
            if self.quantize:
                data_array[i] = self.data_array_q[idx]
            else:
                data_array[i] = self.data_array[idx]
            #
            label_array[i] = self.label_array[idx]
        #
        return data_array, label_array
        
def main():
    argvs = sys.argv
    argc = len(argvs)
    print(argc)
    #
    print("test")

if __name__=='__main__':
    print(">> start")
    sts = main()
    print(">> end")
    print("\007")
    sys.exit(sts)
