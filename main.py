#! /usr/bin/python
# -*- coding: utf-8 -*-
#

#
# LDNN : lesser's Deep Neural Network
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
import pyopencl as cl
#
# LDNN Modules
#
import core
import util
import gpu
import train
import test
#
#
#
sys.setrecursionlimit(10000)
#
# constant values
#
WEIGHT_INDEX_CSV_PATH = "./wi.csv"
#
#
#
def check_weight_distribution():
    w_list = util.csv_to_list(WEIGHT_INDEX_CSV_PATH)
    w_total = len(w_list)
    print w_total
    if w_total<=0:
        print "error"
        return 0

    v_list = []
    num_list = []
    total = 0.0
    for i in range(core.WEIGHT_INDEX_SIZE):
        key = str(i)
        num =  w_list.count(key)
        num_list.append(num)
        v = float(num)/w_total*100.0
        print "[%02d] %d : %f" % (i, num, v)
        v_list.append(v)
        total = total + v

    ave = total / float(len(v_list))
    print "average : %f" % (ave)

    for i in range(core.WEIGHT_INDEX_SIZE):
        dif = v_list[i] - ave
        print "[%02d] %f" % (i, dif)

    for i in range(core.WEIGHT_INDEX_SIZE):
        print num_list[i]
    
    return 0
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
def main():
    argvs = sys.argv
    argc = len(argvs)
    #
    debug = 1
    it = 20*20
    batch_size = 1000
    data_size = 784
    num_class = 10
    mini_batch_size = 1000
    #
    # GPU
    #
    platform_id = 0
    print "- Select an GPU -"
    for platform in cl.get_platforms():
        d = 0
        for device in platform.get_devices():
            print("%d : %s" % (d, device.name))
            d = d + 1
        #
    #
    menu = get_key_input("input command >")
    if menu==0:
        device_id = 0
    elif menu==1:
        device_id = 1
    elif menu==2:
        device_id = 2
    else:
        device_id = 1 # my MBP
    #
    my_gpu = gpu.Gpu(platform_id, device_id)
    my_gpu.set_kernel_code()
    #
    #
    #
    package_id = 0
    print "- Select a package -"
    print "0 : MNIST"
    print "1 : CIFAR-10"
    menu = get_key_input("input command >")
    if menu==0:
        package_id = 0
    elif menu==1:
        package_id = 1
    else:
        package_id = 0
    #
    #
    #
    print "0 : train"
    print "1 : test (batch)"
    print "2 : test (single)"
    print "3 : train (mini-batch)"
    print "3 : train (mini-batch pre)"
    menu = get_key_input("input command >")
    if menu==0:
        mode = 0
    elif menu==1:
        mode = 1
    elif menu==2:
        mode = 2
    elif menu==3:
        mode = 3
    elif menu==4:
        mode = 4
    else:
        mode = 1
    #
    package = util.Package(package_id)
    r = package.setup_dnn(my_gpu)
    if r is None:
        print "fatal DNN error"
        return 0
    #
    if mode==0: # train
        package.load_batch()
        r.set_batch(package._train_image_batch, package._train_label_batch, 0, batch_size, package._image_size, package._num_class, 0)
        train.loop(it, r, package, debug)
    elif mode==1: # test (batch)
        test.test(r, package)
    elif mode==2: # test (single)
        test.test_single(r, package)
    elif mode==3: # train (mini-batch)
        #mini_batch_size = 150
        #num = package._train_batch_size / mini_batch_size
        #epoc = 4
        t = train.Train(package, r)
        t.set_limit(0.000001)
        t.set_mini_batch_size(150)
        t.set_divider(64)
        t.set_iteration(package._train_batch_size/150)
        t.set_epoc(4)
        t.set_layer_direction(1) # output to input
        t.loop()
        #train.train_minibatch(r, package, mini_batch_size, num, epoc)
    elif mode==4: # train (mini-batch pre)
        mini_batch_size = 2000
        num = 50
        epoc = 5
        train.train_minibatch_preset(r, package, mini_batch_size, num, epoc)
    else:
        print "input error"
        pass
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
