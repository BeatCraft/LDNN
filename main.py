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
#
# LDNN Modules
#
import core
import util
import gpu
import train
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
def test(r):
    batch_size = r._batch_size
    print ">>batch test mode (%d)" % (batch_size)
    dist = [0,0,0,0,0,0,0,0,0,0] # data_class
    rets = [0,0,0,0,0,0,0,0,0,0] # result of infs
    oks  = [0,0,0,0,0,0,0,0,0,0] # num of correct
    #
    start_time = time.time()
    r.propagate(-1, -1, -1, -1, 0)
    infs = r.get_inference()
    elapsed_time = time.time() - start_time
    t = format(elapsed_time, "0")
    print "time = %s" % (t)
    #
    ca = 0
    for i in range(batch_size):
        dist[r._batch_class[i]] = dist[r._batch_class[i]] + 1
        inf = infs[i]
        index = -1
        mx = max(inf)
        if mx>0.0:
            for k in range(10):
                if inf[k] == mx:
                    index = k
        #
        rets[index] = rets[index] + 1
        if index==r._batch_class[i]:
            oks[index] = oks[index] +1
            ca = ca + 1
    #
    print "---------------------------------"
    print "result : %d / %d" % (ca, batch_size)
    accuracy = float(ca) / float(batch_size)
    print "accuracy : %f" % (accuracy)
    print "---------------------------------"
    print "class\t|dist\t|infs\t|ok"
    print "---------------------------------"
    for i in range(10):
        print "%d\t| %d\t| %d\t| %d"  % (i, dist[i], rets[i], oks[i])
    #
    print "---------------------------------"
#
#
#
def main():
    argvs = sys.argv
    argc = len(argvs)
    #
    debug = 1
    it = 20*20
    batch_size = 2000
    #
    # GPU
    #
    platform_id = 0
    device_id = 1
    print "- Select a GPU -"
    print "0 : AMD Server"
    print "1 : Intel on MBP"
    print "2 : eGPU (AMD Radeon Pro 580)"
    menu = get_key_input("input command >")
    if menu==0:
        device_id = 0
    elif menu==1:
        device_id = 1
    elif menu==2:
        device_id = 2
    else:
        device_id = 1
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
    menu = get_key_input("input command >")
    if menu==0:
        mode = 0
    elif menu==1:
        mode = 1
    elif menu==2:
        mode = 2
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
        package.load_batch()
        batch_size = package._test_batch_size
        r.set_batch(package._test_image_batch, package._test_label_batch, 0, batch_size, package._image_size, package._num_class, 0)
        test(r)
    elif mode==2: # test (single)
        pass
 
# self-test
#        package.load_batch()
#        r.set_batch(package._train_image_batch, package._train_label_batch, 0, batch_size, package._image_size, package._num_class, 0)
#        test(r)
#    elif mode==3 # init_WI
#        package.load_batch()
#        init_WI(r, batch, batch_size, data_size):
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
