#! c:/Python26/python.exe
# -*- coding: utf-8 -*-

import os, sys, time, math
from stat import *
import random
import copy
import struct

import pickle
import numpy as np
import cPickle

from PIL import Image
from PIL import ImageFile
from PIL import JpegImagePlugin
from PIL import ImageFile
from PIL import PngImagePlugin
import zlib

import csv

import core
#import gpu
#
# constant values
#
C_PURPLE = (255, 0, 255)
C_RED = (255, 0, 0)
C_ORANGE = (255, 128, 0)
C_YELLOW = (255, 255, 0)
C_RIGHT_GREEN = (128, 255, 128)
C_GREEN = (0, 255, 0)
C_RIGHT_BLUE = (128, 128, 255)
C_BLUE = (0, 0, 255)
COLOR_PALLET = [C_BLUE, C_RIGHT_BLUE, C_GREEN, C_RIGHT_GREEN,
                C_YELLOW, C_ORANGE, C_RED, C_PURPLE]
#
#
#
def cross_emtropy_error(y, y_len, t, t_len):
    if y_len != t_len:
        return None

    s = 0.0
    for i in range(y_len):
        s += (np.log(y[i])*t[i])

    s = s * -1.0
    return s

def cross_emtropy_error_2(y, y_len, t, t_len):
    delta = 1e-7
    #print np.log(y + delta)
    return -np.sum(t * np.log(y + delta))
#
#def cross_emtropy_error_fast(y, t, t_class):
#    return np.log(y[t_class]) * t[t_class] * -1
#
def cross_emtropy_error_fast(y, t_class):
    delta = 1e-7
    return -np.log(y[t_class]+delta)
#
#
#
def mean_squared_error_np(y, y_len, t, t_len):
    return np.sum( (y-t)**2.0 )/float(y_len)
#
#
#
def mean_squared_error(y, y_len, t, t_len):
    if y_len != t_len:
        return None
    
    s = 0.0
    for i in range(y_len):
        s += (y[i]-t[i])**2.0
    
    s = s / float(y_len)
    return s
#
#
#
def mean_squared_error_B(y, y_len, t, t_len):
    if y_len != t_len:
        return None
    
    s = 0.0
    for i in range(y_len):
        s += abs(y[i]-t[i])

    return s
#
# mostly, mean_absolute_error is used
#
def mean_absolute_error(y, y_len, t, t_len):
    if y_len != t_len:
        return None
    
    s = 0.0
    #sum = 0.0
    
    for i in range(y_len):
        #sum += y[i]
        s += abs(y[i]-t[i])
    
    #if sum<=0:
    #    print "FUCK(%f)" % sum
    
    return s/float(y_len)
#    if sum>0.0:
#       return s/y_len
#    return 100.0
#
#
#
def img2List(img):
    pix = img.load()
    w = img.size[0]
    h = img.size[1]
    
    ret = []
    for y in range(h):
        for x in range(w):
            k = pix[x, y] #/ 255.0
            ret.append(k)
    return ret
#
#
#
def loadData(path):
    img = Image.open(path)
    img = img.convert("L")
    #img = img.resize((14,14))
    data = img2List(img)
    return data
#
#
#
def pickle_save(path, data):
    with open(path, mode='wb') as f:
        pickle.dump(data, f)
#
#
#
def pickle_load(path):
    try:
        with open(path, mode='rb') as f:
            data = pickle.load(f)
            return data
    except:
        return None
#
#
#
def exportPng(r, num_of_processed):
    connections = r.getConnections()
    wlist = []
    for con in connections:
        w = con.getWeightIndex()
        wlist.append(w)
        
        lc = r.countLayers()
        bc = lc - 1
        width = 0
        height = 0
    
    for i in range(bc):
        left = r.getLayerAt(i)
        left_nc = left.countNodes()
        right = r.getLayerAt(i+1)
        right_nc = right.countNodes()
        width = width + right_nc*4
        if left_nc + right_nc > height:
            height = left_nc*4 + right_nc*4
    
    img = Image.new("RGB", (width+100, height+10))
    pix = img.load()
    windex = 0
    w = 0
    h = 0
    
    for i in range(bc):
        left = r.getLayerAt(i)
        left_nc = left.countNodes()
        right = r.getLayerAt(i+1)
        right_nc = right.countNodes()
        
        start_w = left_nc*4*i + 10*i
        start_h = 0
        
        for x in range(right_nc):
            w = start_w + x*4
            for y in range(left_nc):
                h = start_h + y*4
                #print "[%d]%d, %d : %d" % (i, w, h, windex)
                wv = wlist[windex]
                v = COLOR_PALLET[wv]
                pix[w,   h  ] = v
                pix[w,   h+1] = v
                pix[w,   h+2] = v
                pix[w+1, h  ] = v
                pix[w+1, h+1] = v
                pix[w+1, h+2] = v
                pix[w+2, h  ] = v
                pix[w+2, h+1] = v
                pix[w+2, h+2] = v
                windex = windex + 1

    start_h = 0

    save_name = "./%05d.png" % (num_of_processed)
    img.save(save_name)
#
#
#
def list_to_csv(path, data_list):
    with open(path, 'wb') as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)
        wr.writerow(data_list)
#
#
#
def csv_to_list(path):
    print "csv_to_list()"
    data_list = []
    try:
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                for cell in row:
                    data_list.append(cell)

        return data_list

    except:
        return data_list

    return data_list
#
#
#
def get_key_input(prompt):
    try:
        c = input(prompt)
    except:
        c = -1
        
    return c
#
#
#
TRAIN_IMAGE_PATH = ["../ldnn_package/MNIST/train-images-idx3-ubyte",
                    "../ldnn_package/cifar-10-batches-py/data_batch_1"]
TRAIN_LABEL_PATH = ["../ldnn_package/MNIST/train-labels-idx1-ubyte",
                    "../ldnn_config/cifar-10-batches-py/train_label_batch.pickle"]
TRAIN_IMAGE_BATCH_PATH = ["../ldnn_config/MNIST/train_image_batch.pickle",
                          "../ldnn_config/cifar-10-batches-py/train_image_batch.pickle"]
TRAIN_LABEL_BATCH_PATH = ["../ldnn_config/MNIST/train_label_batch.pickle",
                          "../ldnn_config/cifar-10-batches-py/train_label_batch.pickle"]
TRAIN_BATCH_SIZE = [60000, 10000]
TEST_BATCH_SIZE = [10000, 10000]
TEST_IMAGE_PATH = ["../ldnn_package/MNIST/t10k-images-idx3-ubyte",
                   "./"]
TEST_LABEL_PATH = ["../ldnn_package/MNIST/t10k-labels-idx1-ubyte",
                   "./"]
TEST_IMAGE_BATCH_PATH = ["../ldnn_config/MNIST/test_image_batch.pickle",
                         "../ldnn_config/cifar-10-batches-py/test_image_batch.pickle"]
TEST_LABEL_BATCH_PATH = ["../ldnn_config/MNIST/test_label_batch.pickle",
                         "../ldnn_config/cifar-10-batches-py/test_label_batch.pickle"]

PACKAGE_NAME = ["MNIST", "cifar-10-batches-py"]
PACKAGE_IMAGE_WIDTH = [28, 32]
PACKAGE_IMAGE_HEIGHT = [28, 32]
PACKAGE_IMAGE_SIZE = [784, 1024]
PACKAGE_NUM_CLASS = [10, 10]

class Package:
    def __init__(self, package_id=0):
        self._package_id = package_id
        self._image_size = PACKAGE_IMAGE_SIZE[package_id]
        self._num_class = PACKAGE_NUM_CLASS[package_id]
        self._train_batch_size = TRAIN_BATCH_SIZE[package_id]
        self._test_batch_size = TEST_BATCH_SIZE[package_id]
        #
        self._name = PACKAGE_NAME[package_id]
        self._wi_csv_path = "../ldnn_config/%s/wi.csv" % (PACKAGE_NAME[package_id])
        #
        self._train_image_path = TRAIN_IMAGE_PATH[package_id]
        self._train_label_path = TRAIN_LABEL_PATH[package_id]
        self._test_image_path = TEST_IMAGE_PATH[package_id]
        self._test_label_path = TEST_LABEL_PATH[package_id]
        #
        self._train_image_batch_path = TRAIN_IMAGE_BATCH_PATH[package_id]
        self._train_label_batch_path = TRAIN_LABEL_BATCH_PATH[package_id]
        self._test_image_batch_path = TEST_IMAGE_BATCH_PATH[package_id]
        self._test_label_batch_path = TEST_LABEL_BATCH_PATH[package_id]
        #
        
    def load_batch(self):
        if os.path.isfile(self._train_image_batch_path):
            #print "restore train image batch"
            self._train_image_batch = pickle_load(self._train_image_batch_path)
        else:
            print "fatal error : no train image batch"
        #
        if os.path.isfile(self._train_label_batch_path):
            #print "restore train label batch"
            self._train_label_batch = pickle_load(self._train_label_batch_path)
        else:
            print "fatal error : no train label batch"
        #
        if os.path.isfile(self._test_image_batch_path):
            #print "restore test image batch"
            self._test_image_batch = pickle_load(self._test_image_batch_path)
        else:
            print "fatal error : no test image batch"
        #
        if os.path.isfile(self._test_label_batch_path) :
            #print "restore test label batch"
            self._test_label_batch = pickle_load(self._test_label_batch_path)
        else:
            print "fatal error : no test label batch"
            print self._test_label_batch_path
            
    def setup_dnn(self, my_gpu):
        r = core.Roster()
        r.set_gpu(my_gpu)
        #
        if self._package_id==0: # MNIST
            r.add_layer(core.LAYER_TYPE_INPUT, self._image_size, self._image_size)  # input
            c = r.countLayers()
            layer = core.ConvLayer(c,  28, 28, 3, my_gpu) # conv for MNIST
            r.layers.append(layer)
            #print layer._kernel_num
            r.add_layer(core.LAYER_TYPE_HIDDEN, layer._kernel_num, 64) # hidden 1
            #r.add_layer(1, self._image_size, 64) # hidden 1
            #r.add_layer(1, 64, 64) # hidden 2
            r.add_layer(core.LAYER_TYPE_OUTPUT, 64, self._num_class) # out
        elif self._package_id==1: # cifa-10
            input_layer = r.add_layer(0, self._image_size, self._image_size)
            hidden_layer_1 = r.add_layer(1, self._image_size, 64)
            hidden_layer_2 = r.add_layer(1, 64, 64)
            output_layer = r.add_layer(2, 64, self._num_class)
        else:
            print "package error"
            return None
        #
        if os.path.isfile(self._wi_csv_path):
            #print "restore weight index"
            r.import_weight_index(self._wi_csv_path)
        else:
            #print "init weight index"
            r.init_weight()
            r.export_weight_index(self._wi_csv_path)
        #
        if my_gpu:
            r.update_weight()
        #
        return r

def echo(data):
    print data
