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
def get_key_input(prompt):
    try:
        c = input(prompt)
    except:
        c = -1
    return c
#
#
#
def test(r, package):
    package.load_batch()
    eval_size = package._test_batch_size
    print ">>batch test mode (%d)" % (eval_size)
    
    #
    data_size = package._image_size
    num_class = package._num_class
    data_array = np.zeros((1, data_size), dtype=np.float32)
    class_array = np.zeros(1, dtype=np.int32)
    #
    r.set_batch(data_array, class_array, 0, 1, data_size, num_class, 0)
    #
    dist = [0,0,0,0,0,0,0,0,0,0] # data_class
    rets = [0,0,0,0,0,0,0,0,0,0] # result of infs
    oks  = [0,0,0,0,0,0,0,0,0,0] # num of correct
    #
#    data_array[0] = package._train_image_batch[0]
#    class_array[0] = package._train_label_batch[0]
#    r.set_data(data_array, data_size, class_array)
    #
#    r.propagate(-1, -1, -1, -1, 0)
#    infs = r.get_inference()
#    print infs[0]
    #
    start_time = time.time()
    ca = 0
    for i in range(eval_size):
        data_array[0] = package._train_image_batch[i]
        class_array[0] = package._train_label_batch[i]
        r.set_data(data_array, data_size, class_array)
        r.propagate(-1, -1, -1, -1, 0)
        infs = r.get_inference()
        #
        dist[class_array[0]] = dist[class_array[0]] + 1
        inf = infs[0]
        #print inf
        index = -1
        mx = max(inf)
        if mx>0.0:
            for k in range(10):
                if inf[k] == mx:
                    index = k
        #
        rets[index] = rets[index] + 1
        #
        if index==class_array[0]:
            oks[index] = oks[index] +1
            ca = ca + 1
        #
    #
    print "---------------------------------"
    print "result : %d / %d" % (ca, eval_size)
    accuracy = float(ca) / float(eval_size)
    print "accuracy : %f" % (accuracy)
    print "---------------------------------"
    print "class\t|dist\t|infs\t|ok"
    print "---------------------------------"
    for i in range(10):
        print "%d\t| %d\t| %d\t| %d"  % (i, dist[i], rets[i], oks[i])
    #
    print "---------------------------------"
    #
    elapsed_time = time.time() - start_time
    t = format(elapsed_time, "0")
    print "time = %s" % (t)
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
        print "abort"
        return 0
    #
    my_gpu = gpu.Gpu(platform_id, device_id)
    my_gpu.set_kernel_code()
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
        print "abort"
        return 0
    #
    print "0 : test"
    print "1 : self-test"
    menu = get_key_input("input command >")
    if menu==0:
        mode = 0
    elif menu==1:
        mode = 1
    else:
        print "abort"
        return 0
    #
    package = util.Package(package_id)
    r = package.setup_dnn(my_gpu)
    if r is None:
        print "fatal DNN error"
        return 0
    #
    if mode==0: # test
        test(r, package)
    #
#    elif mode==1: # self-test
#        package.load_batch()
#        r.set_batch(package._train_image_batch, package._train_label_batch, 0,
# batch_size, package._image_size, package._num_class, 0)
#        test(r)
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
