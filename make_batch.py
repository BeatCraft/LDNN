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
#from PIL import ImageFile
#from PIL import JpegImagePlugin
#from PIL import ImageFile
#from PIL import PngImagePlugin
#import zlib
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
MNIST_IMAGE_HEADER_SIZE = 16
MNIST_LABEL_HEADER_SIZE  = 8
MNIST_IMAGE_SIZE = 784
#
def make_mnist_train_batch(package):
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
        #print label
    #
    #return 0
    
    file_in = open(image_path)
    header = file_in.read(MNIST_IMAGE_HEADER_SIZE)
    #
    images = np.zeros((batch_size, (MNIST_IMAGE_SIZE)), dtype=np.float32)
    #
    for i in range(batch_size):
        image = file_in.read(MNIST_IMAGE_SIZE)
        da = np.frombuffer(image, dtype=np.uint8)
        #
        if i<10:
            img = Image.new("L", (28, 28), 0)
            pix = img.load()
            for y in range(28):
                for x in range(28):
                    v = da[28*y + x]
                    pix[x, y] = int(v)
                #
            #
            img.save("./debug/mnist/%05d-%d.png" %(i, labels[i]))
        #
        #
        a_float = da.astype(np.float32) # convert from uint8 to float32
        images[i] = a_float
    #
    print(len(labels))
    print(images.shape[0])
    
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
    print(len(labels))
    print(images.shape[0])
    util.pickle_save(package._test_image_batch_path, images)
    util.pickle_save(package._test_label_batch_path, labels)
#
#
#
CIFAR10_TRAIN_BATCH_NUM  = 5
CIFAR10_TRAIN_BATCH_SIZE  = 10000
#CIFAR10_TEST_BATCH_NUM = 1
CIFAR10_TEST_BATCH_SIZE   = 10000
CIFAR10_NUM_CLASS = 10
CIFAR10_IMAGE_WIDTH = 32
CIFAR10_IMAGE_HEIGHT = 32
CIFAR10_IMAGE_SIZE = CIFAR10_IMAGE_WIDTH*CIFAR10_IMAGE_HEIGHT*3
CIFAR10_IMAGE_Y_SIZE = CIFAR10_IMAGE_WIDTH*CIFAR10_IMAGE_HEIGHT
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
    label_list = []
    batch_size = CIFAR10_TRAIN_BATCH_SIZE*CIFAR10_TRAIN_BATCH_NUM
    data_array = np.zeros((batch_size, CIFAR10_IMAGE_Y_SIZE), dtype=np.float32)
    pix = np.zeros((1024,3), dtype=np.uint8)
    #
    for i in range(CIFAR10_TRAIN_BATCH_NUM):
        path = CIFAR10_TRAIN_DATA_PATH[i]
        with open(path, 'rb') as fo:
            dict = pickle.load(fo)
            label_list.extend(dict["labels"])
            images_rgb = dict["data"]
            offset = CIFAR10_TRAIN_BATCH_SIZE*i
            for j in range(CIFAR10_TRAIN_BATCH_SIZE):
                print("(%d, %d)" % (i, j))
                image = images_rgb[j]
                image = image.reshape([3,1024])
                red = image[0]
                green = image[1]
                blue = image[2]
                for m in range(1024):
                    r = red[m]
                    g = green[m]
                    b = blue[m]
                    pix[m][0] = r
                    pix[m][1] = g
                    pix[m][2] = b
                    y = 0.299*float(r) + 0.587*float(g) + 0.114*float(b)
                    data_array[offset+j][m] = y
                # for m
                # save rgb
                array_rgb = pix.reshape([32, 32, 3])
                img = Image.fromarray(array_rgb)
                label = label_list[offset+j]
                save_path = "../ldnn_package/cifar-10-batches-py/train/rgb/%d/%d/%05d.png" % (i, label, offset+j)
                img.save(save_path)
                # save y
                array_y = data_array[offset+j].astype(np.uint8)
                array_y = array_y.reshape([32,32])
                img = Image.fromarray(array_y)
                save_path = "../ldnn_package/cifar-10-batches-py/train/y/%d/%d/%05d.png" % (i, label, offset+j)
                img.save(save_path)
                #
            # for j
        # with
    # for i
    util.pickle_save(package._train_image_batch_path, data_array)
    util.pickle_save(package._train_label_batch_path, label_list)
    return 0
#
#
#
def make_cifa10_test_batch(package):
    label_list = []
    batch_size = CIFAR10_TEST_BATCH_SIZE
    data_array = np.zeros((batch_size, CIFAR10_IMAGE_Y_SIZE), dtype=np.float32)
    pix = np.zeros((1024,3), dtype=np.uint8)
    path = CIFAR10_TEST_DATA_PATH
    #
    with open(path, 'rb') as fo:
        dict = pickle.load(fo)
        label_list.extend(dict["labels"])
        images_rgb = dict["data"]
        for j in range(CIFAR10_TRAIN_BATCH_SIZE):
            print("(%d)" % (j))
            image = images_rgb[j]
            image = image.reshape([3,1024])
            red = image[0]
            green = image[1]
            blue = image[2]
            for m in range(1024):
                r = red[m]
                g = green[m]
                b = blue[m]
                pix[m][0] = r
                pix[m][1] = g
                pix[m][2] = b
                y = 0.299*float(r) + 0.587*float(g) + 0.114*float(b)
                data_array[j][m] = y
            # for m
            #
            # save rgb
            array_rgb = pix.reshape([32, 32, 3])
            img = Image.fromarray(array_rgb)
            label = label_list[j]
            save_path = "../ldnn_package/cifar-10-batches-py/test/rgb/%d/%05d.png" % (label, j)
            img.save(save_path)
            # save y
            array_y = data_array[j].astype(np.uint8)
            array_y = array_y.reshape([32,32])
            img = Image.fromarray(array_y)
            save_path = "../ldnn_package/cifar-10-batches-py/test/y/%d/%05d.png" % (label, j)
            img.save(save_path)
        # for j
    # with
    #
    util.pickle_save(package._test_image_batch_path, data_array)
    util.pickle_save(package._test_label_batch_path, label_list)
    return 0
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
    # 1 : MNIST2 (smaller network)
    # 2 : CIFAR-10
    package_id = 0
    package = util.Package(package_id)
    #
#    package.load_batch()
#    images = package._train_image_batch
#    print type(images)
#    print len(images)
#    labels = package._train_label_batch
#    print len(labels)
#    return 0
    #
    if package_id==0:
        make_mnist_train_batch(package)
        #make_mnist_test_batch(package)
    elif package_id==1:
        pass
    elif package_id==2:
        #make_cifa10_train_batch(package)
        make_cifa10_test_batch(package)
    else:
        pass
    #
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
