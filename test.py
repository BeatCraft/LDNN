#! /usr/bin/python
# -*- coding: utf-8 -*-
#

#
# LDNN : lesser's Deep Neural Network
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

def print_result(ca, eval_size, class_num, dist, rets, oks):
    print("---------------------------------")
    print("result : %d / %d" % (ca, eval_size))
    accuracy = float(ca) / float(eval_size)
    print("accuracy : %f" % (accuracy))
    print("---------------------------------")
    print("class\t|dist\t|infs\t|ok")
    print("---------------------------------")
    for i in range(10):
        print("%d\t| %d\t| %d\t| %d"  % (i, dist[i], rets[i], oks[i]))
    #
    print("---------------------------------")
    
def test_n(r, pack, n):
    pack.load_batch()
    data_size = pack._image_size
    num_class = pack._num_class
    eval_size = pack._test_batch_size
    dist = np.zeros(num_class, dtype=np.int32)
    rets = np.zeros(num_class, dtype=np.int32)
    oks = np.zeros(num_class, dtype=np.int32)
    print(">>test(%d) = %d" % (n, eval_size))
    #
    it, left = divmod(eval_size, n)
    if left>0:
        print("error : n(=%d) is not appropriate" % (n))
    #
    start_time = time.time()
    #
    r.prepare(n, data_size, num_class)
    data_array = np.zeros((n, data_size), dtype=np.float32)
    class_array = np.zeros(n, dtype=np.int32)
    class_array_2 = np.zeros((n, num_class), dtype=np.float32)
    #
    for i in range(it):
        class_array_2 = class_array_2*0.0
        for j in range(n):
            data_array[j] = pack._train_image_batch[i*n+j]
            class_array[j] = pack._train_label_batch[i*n+j]
            k = pack._train_label_batch[i*n+j]
            class_array_2[j][k] = 1.0
            #print k
            #print class_array_2[j]
        #
        scale = 1
        r.set_data(data_array, data_size, class_array_2, n, scale)
        r.propagate()
        #
        answes = r.get_answer()
        for j in range(n):
            ans = answes[j]
            label = class_array[j]
            rets[ans] = rets[ans] + 1
            dist[label] = dist[label] + 1
            if ans == label:
                oks[ans] = oks[ans] + 1
            #
        #
    #
    ca = sum(oks)
    print("---------------------------------")
    print("result : %d / %d" % (ca, eval_size))
    accuracy = float(ca) / float(eval_size)
    print("accuracy : %f" % (accuracy))
    print("---------------------------------")
    print("class\t|dist\t|infs\t|ok")
    print("---------------------------------")
    for i in range(num_class):
        print("%d\t| %d\t| %d\t| %d"  % (i, dist[i], rets[i], oks[i]))
    #
    print("---------------------------------")
    #
    elapsed_time = time.time() - start_time
    t = format(elapsed_time, "0")
    print("time = %s" % (t))
    #
    #r.propagate()
    ce = r.get_cross_entropy()
    print("CE = %f" % (ce))
#
#
#
def test_single(r, pack, bi):
    pack.load_batch()
    data_size = pack._image_size
    num_class = pack._num_class
    eval_size = pack._test_batch_size #
    dist = np.zeros(num_class, dtype=np.int32)
    rets = np.zeros(num_class, dtype=np.int32)
    oks = np.zeros(num_class, dtype=np.int32)
    print(">>batch test mode (%d)" % (eval_size))
    #
    data_array = np.zeros((1, data_size), dtype=np.float32)
    class_array = np.zeros(1, dtype=np.int32)
    batch_size = 1
    r.prepare(batch_size, data_size, num_class)
    #
    start_time = time.time()
    ca = 0
    for i in range(eval_size):
        if i%100==0:
            print(i)
        #i = 60
        data_array[0] = pack._test_image_batch[i]
        answer = pack._test_label_batch[i]
        class_array[0] = answer
        r.set_data(data_array, data_size, class_array, 1)
        r.propagate(-1, -1, -1, -1, 0)
        infs = r.get_inference()
        dist[answer] = dist[answer] + 1
        inf = infs[0]
        #print inf
        #
        index = -1
        mx = max(inf)
        if mx>0.0:
            for k in range(10):
                if inf[k] == mx:
                    index = k
                #
            #
        #
        #if index>0:
        #    print i
        #
        rets[index] = rets[index] + 1
        if index==answer:
            oks[index] = oks[index] +1
            ca = ca + 1
        #
    #
    print("---------------------------------")
    print("result : %d / %d" % (ca, eval_size))
    accuracy = float(ca) / float(eval_size)
    print("accuracy : %f" % (accuracy))
    print("---------------------------------")
    print("class\t|dist\t|infs\t|ok")
    print("---------------------------------")
    for i in range(10):
        print("%d\t| %d\t| %d\t| %d"  % (i, dist[i], rets[i], oks[i]))
    #
    print("---------------------------------")
    #
    elapsed_time = time.time() - start_time
    t = format(elapsed_time, "0")
    print("time = %s" % (t))
    


    
