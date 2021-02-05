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
    print(argc)
    #
    if argc==6:
        pass
    else:
        print("error in sh")
        return 0
    #
    platform_id = int(argvs[1])
    device_id = int(argvs[2])
    package_id = int(argvs[3])
    config = int(argvs[4])
    mode = int(argvs[5])
    print("platform_id=%d" % (platform_id))
    print("device_id=%d" % (device_id))
    print("package_id=%d" % (package_id))
    print("config=%d" % (config))
    print("mode=%d" % (mode))
    #
    my_gpu = gpu.Gpu(platform_id, device_id)
    my_gpu.set_kernel_code()
    package = util.Package(package_id)
    r = package.setup_dnn(my_gpu, config)
    #
    if mode==0: # train
        mini_batch_size = 500
        print("package._train_batch_size=%d" % (package._train_batch_size))
        t = train.Train(package, r)
        t.set_mini_batch_size(mini_batch_size)
        t.simple_loop()
    elif mode==1: # test
        test.test_n(r, package, 500)
    elif mode==2: #
        test.unit_test(r, package)
    elif mode==3: #
        test.cnn_test(r, package)
    else:
        print("mode error : %d" % (mode))
        return 0
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
