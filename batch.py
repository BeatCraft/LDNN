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
        # 0 : classification, 1 : regression
        self.type = type
        self.class_num = class_num
        #self.data_path = ""
        #self.label_path = ""
        #self.batch_size = 0
        #self.mini_batch_size = 0
        #self.mini_batch_num = 0
        
        self.data_size = data_size

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
    
    def prepare_batch(self, scale=True):
        self.label_array = np.zeros((self.batch_size, self.class_num), dtype=np.float32)
        for i in range(self.batch_size):
            # scale to 0.0 - 1.0
            if scale==True:
                self.data_array[i] = self.data_array[i] / 255.0
            #
            k = int(self.label_list[i])
            self.label_array[i][k] = 1.0
        #
    
    def prepare_mini_batch(self, size):
        self.mini_batch_size = size
        self.mini_batch_num = int(self.batch_size / self.mini_batch_size)
        self.mini_batch_idx_list = []
        for i in range(self.batch_size):
            self.mini_batch_idx_list.append(i)
        #
        random.shuffle(self.mini_batch_idx_list)
        
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
    
    def get_mini_batch(self, offset):
        data_array = np.zeros((self.mini_batch_size, self.data_size), dtype=np.float32)
        label_array = np.zeros((self.mini_batch_size, self.class_num), dtype=np.float32)
        
        #print("mini", data_array.shape, self.mini_batch_size)
        
        idx_offset = self.mini_batch_size * offset
        #print("offset", offset, idx_offset)
        for i in range(self.mini_batch_size):
            idx = self.mini_batch_idx_list[idx_offset + i]
            #print(idx)
            data_array[i] = self.data_array[idx]
            label_array[i] = self.label_array[idx]
        #
        return data_array, label_array
        
    def load_data(self, path): # array
        self.data_path = path
        self.data_array, self.batch_size = self.load(self.data_path)
        print(self.batch_size)
            
    def load_label(self, path): # list
        self.label_path = path
        self.label_list, self.batch_size = self.load(self.label_path)
        print(self.batch_size)
    
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
