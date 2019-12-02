#! /usr/bin/python
# -*- coding: utf-8 -*-
#

#
# LDNN : lesser Deep Neural Network
#

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
MNIST_IMAGE_HEADER_SIZE = 16
MNIST_LABEL_HEADER_SIZE  = 8
MNIST_IMAGE_SIZE = 784
#
#
#
def make_mnist_batch(package):
    image_path = package._train_image_path
    labels_path = package._train_label_path
    batch_size = package._train_batch_size
    #
    file_in = open(labels_path)
    
    header = file_in.read(MNIST_LABEL_HEADER_SIZE)
    data = file_in.read()
    #
    labels = [0 for i in range(batch_size)] # list
    #
    for i in range(batch_size):
        label = struct.unpack('>B', data[i])
        labels[i] = label[0]
    #
    file_in = open(image_path)
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
    print len(labels)
    print images.shape[0]
    
    util.pickle_save(package._train_image_batch_path, images)
    util.pickle_save(package._train_label_batch_path, labels)
#
#
#
def make_mnist2_batch():
    cat = 0
    clut = 0
    data_size = 28*28
    batch_size = 6000
    batch_data = np.zeros((batch_size, data_size), dtype=np.float32)
    batch_label = np.zeros(batch_size, dtype=np.int32)
    divider = 10
    total = 0

    src_index = 0
    dst_index = 0
    for cat in range(10):
        for clut in range(10):
            path = "./package/MNIST2/clustered_data/class_%d/cluster_%d/image.csv" % (cat, clut)
            print path
            #
            with open(path, "r") as f:
                reader = csv.reader(f)
                n = 0
                for row in reader:
                    left = src_index%divider
                    if left==0:
                        i = 0
                        for i in range(data_size):
                            cell = row[i]
                            tmp = int(cell)
                            batch_data[dst_index][i] = float(tmp)
                            i = i + 1
                        #
                        batch_label[dst_index] = cat
                        dst_index = dst_index+1
                        n = n + 1
                    #
                    src_index = src_index+1
                # for
                total = total + n
                print n
            # with
        #
    #
    util.pickle_save("./package/MNIST2/train_image_batch.pickle", batch_data)
    util.pickle_save("./package/MNIST2/train_label_batch.pickle", batch_label)
    print total
    #
    print batch_label
#
#
#
def main():
    argvs = sys.argv
    argc = len(argvs)
    #
    # 0 : MNIST
    # 1 : MNIST2 (clustered data set)
    # 2 : CIFAR-10
    package_id = 0
    package = util.Package(package_id)
    if package_id==0:
        make_mnist_batch(package)
    elif package_id==1:
        pass
    elif package_id==2:
        pass
    else:
        pass
    #
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
