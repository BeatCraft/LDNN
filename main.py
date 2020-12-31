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
import pickle
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
#import logging
#logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)
#get_handler = logging.FileHandler("log/log.txt")
#logger.addHandler(get_handler)
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
    print(w_total)
    if w_total<=0:
        print("error")
        return 0

    v_list = []
    num_list = []
    total = 0.0
    for i in range(core.WEIGHT_INDEX_SIZE):
        key = str(i)
        num =  w_list.count(key)
        num_list.append(num)
        v = float(num)/w_total*100.0
        print("[%02d] %d : %f" % (i, num, v))
        v_list.append(v)
        total = total + v

    ave = total / float(len(v_list))
    print("average : %f" % (ave))

    for i in range(core.WEIGHT_INDEX_SIZE):
        dif = v_list[i] - ave
        print("[%02d] %f" % (i, dif))

    for i in range(core.WEIGHT_INDEX_SIZE):
        print(num_list[i])
    
    return 0
#
#
#
def get_key_input(prompt):
    c = -1
    try:
        if sys.version_info[0]==2:
            c = input(prompt) # Python2.7
        else:
            c = eval(input(prompt)) # Python3.x
        #
    except:
        c = -1
    #
    return c
#
#
#
def main():
    argvs = sys.argv
    argc = len(argvs)
    #
    # GPU
    #
    platform_id = 0
    device_id = 0
    package_id = 0
    config = 0
    mode = 0
    
    print("- Select an OpenCL platform -")
    d = 0
    for platform in cl.get_platforms():
        print("%d : %s" % (d, platform.name))
        d = d + 1
    #
    menu = get_key_input("input command >")
    if menu==0:
        platform_id = 0
    elif menu==1:
        platform_id = 1
    elif menu==2:
        platform_id = 2
    else:
        platform_id = 1 # my MBP
    #

    print("- Select a device -")
    d = 0
    for device in platform.get_devices():
        print("%d : %s" % (d, device.name))
        d = d + 1
    #
    menu = get_key_input("input command >")
    print(menu)
    if menu==0:
        device_id = 0
    elif menu==1:
        device_id = 1
    elif menu==2:
        device_id = 2
    else:
        device_id = 0
    #
    print("%d : %d" %(platform_id, device_id))
    
    my_gpu = gpu.Gpu(platform_id, device_id)
    my_gpu.set_kernel_code()
    #
    #
    #
    package_id = 0
    print("- Select a package -")
    print("0 : MNIST")
    print("1 : CIFAR-10")
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
    print("- Select a config-")
    print("0 : FC for MNIST")
    print("1 : CNN for MNIST")
    menu = get_key_input("input command >")
    if menu==0:
        config = 0
    elif menu==1:
        config = 1
    else:
        config = 1
    #
    
    #
    #
    #
    print("- Select a command-")
    print("0 : train (mini-batch)")
    print("1 : test (500)")
    print("2 : ? (unit test)")
    print("3 : CNN Test")
    menu = get_key_input("input command >")
    if menu==0:
        mode = 0
    elif menu==1:
        mode = 1
    elif menu==2:
        mode = 2
    elif menu==3:
        mode = 3
    else:
        mode = 1
    #
    package = util.Package(package_id)
    r = package.setup_dnn(my_gpu, config)
    if r is None:
        print("fatal DNN error")
        return 0
    #
    if mode==0: # train
        epoc = 4
        mini_batch_size = 500
        loop = 1#16 # 1 2 4 8
        print("package._train_batch_size=%d" % (package._train_batch_size))
        it = int(package._train_batch_size/mini_batch_size)
        print("it = %d" % (it))
        #
        t = train.Train(package, r)
        t.set_limit(0.000001)
        t.set_mini_batch_size(mini_batch_size)
        t.set_iteration(it)
        t.set_epoc(epoc)
        t.set_loop(loop)
        t.set_layer_direction(1) # 0 : in to out, 1 : out to in
        #t.disable_mini_batch()
        #
        t.loop()
    elif mode==1: # test (batch)
        test.test_n(r, package, 500)
    elif mode==2: #
        test.unit_test(r, package)
    elif mode==3: #
        test.cnn_test(r, package)
    else:
        print("input error")
        pass
    #
    return 0
#
#
#
if __name__=='__main__':
    print(">> start")
#    logger.debug('hello')
    sts = main()
    print(">> end")
    print("\007")
    sys.exit(sts)
#
#
#
