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
import cPickle
import csv
from PIL import Image
from PIL import ImageFile
from PIL import JpegImagePlugin
from PIL import ImageFile
from PIL import PngImagePlugin
import zlib
#
#
# LDNN Modules
import core
import util
import gpu
#
#
#
sys.setrecursionlimit(10000)
#
#
#
CIFAR10_TRAIN_BATCH_NUM  = 5
CIFAR10_TRAIN_BATCH_SIZE  = 10000
CIFAR10_TEST_BATCH_SIZE   = 10000
CIFAR10_NUM_CLASS = 10
CIFAR10_IMAGE_WIDTH = 32
CIFAR10_IMAGE_HEIGHT = 32
CIFAR10_IMAGE_SIZE = CIFAR10_IMAGE_WIDTH*CIFAR10_IMAGE_HEIGHT*3
#
CIFAR10_TRAIN_DATA_PATH =[ "../ldnn_package/cifar-10-batches-py/data_batch_1",
                           "../ldnn_package/cifar-10-batches-py/data_batch_2",
                           "../ldnn_package/cifar-10-batches-py/data_batch_3",
                           "../ldnn_package/cifar-10-batches-py/data_batch_4",
                           "../ldnn_package/cifar-10-batches-py/data_batch_5"]
CIFAR10_TEST_DATA_PATH = "../ldnn_package/cifar-10-batches-py/test_batch"
#
#
#
def make_cifa10_train_batch(package):
    batch_size = CIFAR10_TRAIN_BATCH_SIZE * CIFAR10_TRAIN_BATCH_NUM
    data_array = np.zeros((batch_size, CIFAR10_IMAGE_SIZE), dtype=np.float32)
    label_array = np.zeros((batch_size), dtype=np.int32)
    #
    for i in range(CIFAR10_TRAIN_BATCH_NUM):
        path = CIFAR10_TRAIN_DATA_PATH[i]
        with open(path, 'rb') as fo:
            dict = cPickle.load(fo)
            labels = dict["labels"]
            images = dict["data"]
            offset = CIFAR10_TRAIN_BATCH_SIZE*i
            for j in range(CIFAR10_TRAIN_BATCH_SIZE):
                data_array[offset+j] = np.array(images[j])
                label_array[offset+j] = labels[j]
                print "(%d, %d)=%d" % (i, j, labels[j])
            #
        #
    #
    util.pickle_save(package._train_image_batch_path, data_array)
    util.pickle_save(package._train_label_batch_path, label_array)
    return 0
#
#
#
def make_cifa10_test_batch(package):
    batch_size = CIFAR10_TRAIN_BATCH_SIZE
    data_array = np.zeros((batch_size, CIFAR10_IMAGE_SIZE), dtype=np.float32)
    label_array = np.zeros((batch_size), dtype=np.int32)
    #
    path = CIFAR10_TEST_DATA_PATH
    #
    with open(path, 'rb') as fo:
        dict = cPickle.load(fo)
        labels = dict["labels"]
        images = dict["data"]
        for j in range(CIFAR10_TRAIN_BATCH_SIZE):
            data_array[j] = np.array(images[j])
            label_array[j] = labels[j]
            print "(%d)=%d" % (j, labels[j])
        #
        #print images[0].shape
    #
    #print data_array[0]
    util.pickle_save(package._test_image_batch_path, data_array)
    util.pickle_save(package._test_label_batch_path, label_array)
    return 0
#
#
#
def main():
    argvs = sys.argv
    argc = len(argvs)
    #
    # cifa-10
    #
    package_id = 1
    package = util.Package(package_id)
    package.load_batch()
    #
    make_cifa10_train_batch(package)
    make_cifa10_test_batch(package)
    #
    return 0
#
#
#
if __name__=='__main__':
    print ">> start"
    sts = main()
    print ">> end"
    print("\007")
    sys.exit(sts)
#
#
#
