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
MNIST_TRAIN_BATCH_SIZE  = 60000
MNIST_TEST_BATCH_SIZE   = 10000
MNIST_IMAGE_HEADER_SIZE = 16
MNIST_LABEL_HEADER_SIZE  = 8
MNIST_IMAGE_WIDTH = 28
MNIST_IMAGE_HEIGHT = 28
MNIST_NUM_CLASS = 10
MNIST_IMAGE_SIZE = MNIST_IMAGE_WIDTH*MNIST_IMAGE_HEIGHT
MNIST_TRAIN_IMAGE_PATH = "./MNIST/train-images-idx3-ubyte"
MNIST_TRAIN_LABEL_PATH = "./MNIST/train-labels-idx1-ubyte"
MNIST_TRAIN_IMAGE_BATCH_PATH = "./MNIST/train_image_batch.pickle"
MNIST_TRAIN_LABEL_BATCH_PATH = "./MNIST/train_label_batch.pickle"
MNIST_TEST_IMAGE_PATH = "./MNIST/t10k-images-idx3-ubyte"
MNIST_TEST_LABEL_PATH = "./MNIST/t10k-labels-idx1-ubyte"
MNIST_TEST_IMAGE_BATCH_PATH = "./MNIST/test_image_batch.pickle"
MNIST_TEST_LABEL_BATCH_PATH = "./MNIST/test_label_batch.pickle"

class Mnist:
    def __init__(self, mode=0): # 0 : train, 1 : test, 2 : self-test
        print "mnist"
        if mode==0 or mode==2:
            self._mode = 0 # 0 : train, 1 : test
            if os.path.isfile(MNIST_TRAIN_IMAGE_BATCH_PATH) and os.path.isfile(MNIST_TRAIN_LABEL_BATCH_PATH):
                print "restore train batch"
                self._data = pickle_load(MNIST_TRAIN_IMAGE_BATCH_PATH)
                self._labels = pickle_load(MNIST_TRAIN_LABEL_BATCH_PATH)
            else:
                print "make train batch"
                self.make_batch(MNIST_TRAIN_IMAGE_PATH, MNIST_TRAIN_LABEL_PATH, MNIST_TRAIN_BATCH_SIZE)
                pickle_save(MNIST_TRAIN_LABEL_BATCH_PATH, self._labels)
                pickle_save(MNIST_TRAIN_IMAGE_BATCH_PATH, self._data)

        else:
            self._mode = 1 # 0 : train, 1 : test
            if os.path.isfile(MNIST_TEST_LABEL_BATCH_PATH) and os.path.isfile(MNIST_TEST_IMAGE_BATCH_PATH) :
                print "restore test batch"
                self._data = pickle_load(MNIST_TEST_IMAGE_BATCH_PATH)
                self._labels = pickle_load(MNIST_TEST_LABEL_BATCH_PATH)
                #print self._labels
                #print self._data[0]
            else:
                print "make test batch"
                self.make_batch(MNIST_TEST_IMAGE_PATH, MNIST_TEST_LABEL_PATH, MNIST_TEST_BATCH_SIZE)
                pickle_save(MNIST_TEST_LABEL_BATCH_PATH, self._labels)
                pickle_save(MNIST_TEST_IMAGE_BATCH_PATH, self._data)
                
    
    def make_batch(self, image_path, labels_path, batch_size):
        file_in = open(labels_path)
        header = file_in.read(MNIST_LABEL_HEADER_SIZE)
        data = file_in.read()
        #
        self._labels = [0 for i in range(batch_size)]
        #
        for i in range(batch_size):
            label = struct.unpack('>B', data[i])
            self._labels[i] = label[0]
        #
        file_in = open(image_path)
        header = file_in.read(MNIST_IMAGE_HEADER_SIZE)
        #
        self._data = np.zeros((batch_size, (MNIST_IMAGE_SIZE)), dtype=np.float32)
        #
        for i in range(batch_size):
            data = file_in.read(MNIST_IMAGE_SIZE)
            da = np.frombuffer(data, dtype=np.uint8)
            a_float = da.astype(np.float32) # convert from uint8 to float32
            self._data[i] = a_float
        #

    def setup_dnn(self, my_gpu):
        if my_gpu:
            pass
        else:
            return None
        
        r = core.Roster()
        r.set_gpu(my_gpu)

        input_layer = r.add_layer(0, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)
        hidden_layer_1 = r.add_layer(1, MNIST_IMAGE_SIZE, 32)
        hidden_layer_2 = r.add_layer(1, 32, 32)
        hidden_layer_3 = r.add_layer(1, 32, 32)
        hidden_layer_4 = r.add_layer(1, 32, 32)
        output_layer = r.add_layer(2, 32, 10)
    
        return r
#
#
#
CIFAR10_TRAIN_BATCH_SIZE  = 10000
CIFAR10_TEST_BATCH_SIZE   = 10000
CIFAR10_IMAGE_WIDTH = 32
CIFAR10_IMAGE_HEIGHT = 32
CIFAR10_NUM_CLASS = 10
CIFAR10_IMAGE_SIZE = CIFAR10_IMAGE_WIDTH*CIFAR10_IMAGE_HEIGHT*3
CIFAR10_IMAGE_Y_SIZE = CIFAR10_IMAGE_WIDTH*CIFAR10_IMAGE_HEIGHT

CIFAR10_TRAIN_DATA_PATH = "./cifar-10-batches-py/data_batch_1"
CIFAR10_TRAIN_IMAGE_BATCH_PATH = "./cifar-10-batches-py/train_image_batch.pickle"
CIFAR10_TRAIN_LABEL_BATCH_PATH = "./cifar-10-batches-py/train_label_batch.pickle"

CIFAR10_TEST_DATA_PATH = "./cifar-10-batches-py/test_batch"
CIFAR10_TEST_IMAGE_BATCH_PATH = "./cifar-10-batches-py/test_image_batch.pickle"
CIFAR10_TEST_LABEL_BATCH_PATH = "./cifar-10-batches-py/test_label_batch.pickle"
#
class Cifar10:
    def __init__(self, mode=0): # 0 : train, 1 : test, 2 : self-test
        print "cifar10(%d)" % (mode)
        if mode==0 or mode==2:
            self._mode = 0 # train
            if os.path.isfile(CIFAR10_TRAIN_IMAGE_BATCH_PATH) and os.path.isfile(CIFAR10_TRAIN_LABEL_BATCH_PATH):
                print "restore train batch"
                self._data = pickle_load(CIFAR10_TRAIN_IMAGE_BATCH_PATH)
                self._labels = pickle_load(CIFAR10_TRAIN_LABEL_BATCH_PATH)
            else:
                print "make train batch"
                self.make_batch(CIFAR10_TRAIN_DATA_PATH, CIFAR10_TRAIN_BATCH_SIZE)
                pickle_save(CIFAR10_TRAIN_IMAGE_BATCH_PATH, self._data)
                pickle_save(CIFAR10_TRAIN_LABEL_BATCH_PATH, self._labels)
        else:
            self._mode = 1 # test
            if os.path.isfile(CIFAR10_TEST_IMAGE_BATCH_PATH) and os.path.isfile(CIFAR10_TEST_LABEL_BATCH_PATH) :
                print "restore test batch"
                self._data = pickle_load(CIFAR10_TEST_IMAGE_BATCH_PATH)
                self._labels = pickle_load(CIFAR10_TEST_LABEL_BATCH_PATH)
            else:
                print "make test batch"
                self.make_batch(CIFAR10_TEST_DATA_PATH, CIFAR10_TEST_BATCH_SIZE)
                pickle_save(CIFAR10_TEST_IMAGE_BATCH_PATH, self._data)
                pickle_save(CIFAR10_TEST_LABEL_BATCH_PATH, self._labels)
        #
    def make_batch(self, path, batch_size):
        with open(path, 'rb') as fo:
             dict = cPickle.load(fo)
        #
        self._labels = dict["labels"]
        #
        image_rgb = dict["data"]
        images_rgb = image_rgb.astype(np.float32)
        self._data = np.zeros((batch_size, (CIFAR10_IMAGE_Y_SIZE)), dtype=np.float32)
        for j in range(batch_size):
            for i in range(CIFAR10_IMAGE_Y_SIZE):
                red = image_rgb[j][i]
                green = image_rgb[j][CIFAR10_IMAGE_Y_SIZE+i]
                blue = image_rgb[j][CIFAR10_IMAGE_Y_SIZE*2+i]
                self._data[j][i] = 0.299*red + 0.587*green + 0.114*blue
        #

    def setup_dnn(self, my_gpu):
        if my_gpu:
            pass
        else:
            return None
        #
        r = core.Roster()
        r.set_gpu(my_gpu)
        #
        input_layer = r.add_layer(0, CIFAR10_IMAGE_Y_SIZE, CIFAR10_IMAGE_Y_SIZE)
        hidden_layer_1 = r.add_layer(1, CIFAR10_IMAGE_Y_SIZE, 32)
        hidden_layer_2 = r.add_layer(1, 32, 32)
        output_layer = r.add_layer(2, 32, 10)
        #
        return r
