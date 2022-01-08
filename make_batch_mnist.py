#! /usr/bin/python
# -*- coding: utf-8 -*-
#

#
# LDNN : lesser Deep Neural Network
#
PACKAGE_BASE_PATH = "../ldnn_package/"
#
#
#
import os, sys, time, math
from stat import *
import random
import copy
import math
import multiprocessing as mp
import numpy as np
import struct
import pickle
import csv
from PIL import Image
#
#
# LDNN Modules
import core
import util
import package
import gpu
#
#
#
sys.setrecursionlimit(10000)
#
#
#
MNIST_IMAGE_HEADER_SIZE = 16
MNIST_LABEL_HEADER_SIZE  = 8
MNIST_IMAGE_SIZE = 784
#
def make_mnist_train_batch(package):
    image_path = package._train_image_path
    labels_path = package._train_label_path
    batch_size = package._train_batch_size
    #
    file_in = open(labels_path, 'rb')#encoding='latin1') 
    header = file_in.read(MNIST_LABEL_HEADER_SIZE)
    data = file_in.read()
    #
    labels = [0 for i in range(batch_size)] # list
    #
    for i in range(batch_size):
        if i<10:
            print(data[i])
        #
	#label = struct.unpack('>B', data[i])
        labels[i] = int(data[i])
    #
    file_in = open(image_path, 'rb')
    header = file_in.read(MNIST_IMAGE_HEADER_SIZE)
    #
    images = np.zeros((batch_size, (MNIST_IMAGE_SIZE)), dtype=np.float32)
    #
    for i in range(batch_size):
        image = file_in.read(MNIST_IMAGE_SIZE)
        da = np.frombuffer(image, dtype=np.uint8)
        a_float = da.astype(np.float32) # convert from uint8 to float32
        images[i] = a_float
    #
    print((len(labels)))
    print((images.shape[0]))
    
    util.pickle_save(package._train_image_batch_path, images)
    util.pickle_save(package._train_label_batch_path, labels)
#
#
#
def make_mnist_test_batch(package):
    image_path = package._test_image_path
    labels_path = package._test_label_path
    batch_size = package._test_batch_size
    #
    file_in = open(labels_path, 'rb')
    header = file_in.read(MNIST_LABEL_HEADER_SIZE)
    data = file_in.read()
    #
    labels = [0 for i in range(batch_size)] # list
    #
    for i in range(batch_size):
        #label = struct.unpack('>B', data[i])
        labels[i] = int(data[i])#label[0]
    #
    file_in = open(image_path, 'rb')
    header = file_in.read(MNIST_IMAGE_HEADER_SIZE)
    #
    images = np.zeros((batch_size, (MNIST_IMAGE_SIZE)), dtype=np.float32)
    #
    for i in range(batch_size):
        image = file_in.read(MNIST_IMAGE_SIZE)
        da = np.frombuffer(image, dtype=np.uint8)
        a_float = da.astype(np.float32) # convert from uint8 to float32
        images[i] = a_float
    #
    print((len(labels)))
    print((images.shape[0]))
    util.pickle_save(package._test_image_batch_path, images)
    util.pickle_save(package._test_label_batch_path, labels)
#
#
#
def narray2png(p_1d, x, y):
    image = np.array(p_1d, dtype='uint8')
    image = image.reshape([x, y])
    return Image.fromarray(image)
    #return Image.fromarray(np.uint8(image))
#
#
#
def main():
    argvs = sys.argv
    argc = len(argvs)
    #
    # 0 : MNIST
    package_id = 0
    pack = package.Package(package_id)
    #
    make_mnist_train_batch(pack)
    make_mnist_test_batch(pack)
    #
    return 0
#
#
#
if __name__=='__main__':
    print(">> start")
    sts = main()
    print(">> end")
    print("\007")
    sys.exit(sts)
#
#
#
