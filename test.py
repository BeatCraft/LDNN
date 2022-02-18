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
def print_result(ca, eval_size, num_class, dist, rets, oks):
    print("---------------------------------")
    print(("result : %d / %d" % (ca, eval_size)))
    accuracy = float(ca) / float(eval_size)
    print(("accuracy : %f" % (accuracy)))
    print("---------------------------------")
    print("class\t|dist\t|infs\t|ok")
    print("---------------------------------")
    for i in range(num_class):
        print(("%d\t| %d\t| %d\t| %d"  % (i, dist[i], rets[i], oks[i])))
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
    print((">>test(%d) = %d" % (n, eval_size)))
    print(num_class)
    #
    it, left = divmod(eval_size, n)
    if left>0:
        print(("error : n(=%d) is not appropriate" % (n)))
    #
    start_time = time.time()
    #
    n = 1
    it = 1
    r.prepare(n, data_size, num_class)
    mode = r.mode()
    if mode==0:
        data_array = np.zeros((n, data_size), dtype=np.float32)
    elif mode==1:
        data_array = np.zeros((n, data_size), dtype=np.int32)
    #
    class_array = np.zeros(n, dtype=np.int32)
    #class_array_2 = np.zeros((n, num_class), dtype=np.float32)
    #
    for i in range(it):
        #class_array_2 = class_array_2*0.0
        for j in range(n):
            data_array[j] = pack._test_image_batch[i*n+j]
            class_array[j] = pack._test_label_batch[i*n+j]
            #k = pack._test_label_batch[i*n+j]
            #class_array_2[j][k] = 1.0
        #
        scale = 1
        #print data_array[0]
        r.set_batch(data_size, num_class, data_array, class_array, n, 0)
        #r.set_data(data_array, data_size, class_array_2, n, scale)
        r.propagate(0)
        #
        infs = r.get_inference()
        print(infs)
        answes = r.get_answer()
        print(answes)
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
    print_result(ca, eval_size, num_class, dist, rets, oks)
    #
    elapsed_time = time.time() - start_time
    t = format(elapsed_time, "0")
    print(("time = %s" % (t)))
    #
    #r.propagate()
    ce = r.get_cross_entropy(0)
    print(("CE = %f" % (ce)))
    #
  
    #self.mpi_save(pack.save_path(), mpi, com, rank, size)

    


    
