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
def test(r, package):
    package.load_batch()
    batch_size = 10#package._test_batch_size
    data_size = package._image_size
    num_class = package._num_class
    #
    data_array = np.zeros((batch_size, data_size), dtype=np.float32)
    class_array = np.zeros(batch_size, dtype=np.int32)
    #
    for i in range(batch_size):
        data_array[i] = package._test_image_batch[i]
        class_array[i] = package._test_label_batch[i]
    #
    r.prepare(batch_size, data_size, num_class)
    r.set_data(data_array, data_size, class_array, batch_size)
    #
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
        ans = r._batch_class[i]
        dist[ans] = dist[ans] + 1
        #
        inf = infs[i]
        index = -1
        mx = max(inf)
        if mx>0.0:
            for k in range(10):
                if inf[k] == mx:
                    index = k
                #
            #
        #
        rets[index] = rets[index] + 1
        if index==ans:
            oks[index] = oks[index] +1
            ca = ca + 1
        #
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
def test_single(r, package):
    package.load_batch()
    data_size = package._image_size
    num_class = package._num_class
    eval_size = package._test_batch_size #
    dist = np.zeros(num_class, dtype=np.int32)
    rets = np.zeros(num_class, dtype=np.int32)
    oks = np.zeros(num_class, dtype=np.int32)
    print ">>batch test mode (%d)" % (eval_size)
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
            print i
        #i = 60
        data_array[0] = package._test_image_batch[i]
        answer = package._test_label_batch[i]
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
def stat(r, package, path, debug):
    package.load_batch()
    data_size = package._image_size
    num_class = package._num_class
    eval_size = package._test_batch_size
    dist = np.zeros(num_class, dtype=np.int32)
    rets = np.zeros(num_class, dtype=np.int32)
    oks = np.zeros(num_class, dtype=np.int32)
    if debug:
        print ">>batch test mode (%d)" % (eval_size)
    #
    data_array = np.zeros((1, data_size), dtype=np.float32)
    class_array = np.zeros(1, dtype=np.int32)
    #r.set_batch(data_array, class_array, 0, 1, data_size, num_class, 0)
    #r.init_mem(1, data_size, num_class)
    batch_size = 1
    r.prepare(batch_size, data_size, num_class)
    #
    r.import_weight_index(path)
    r.update_weight()
    #
    start_time = time.time()
    ca = 0
    for i in range(eval_size):
        data_array[0] = package._test_image_batch[i]
        answer = package._test_label_batch[i]
        class_array[0] = answer
        r.set_data(data_array, data_size, class_array)
        r.propagate(-1, -1, -1, -1, 0)
        infs = r.get_inference()
        dist[answer] = dist[answer] + 1
        inf = infs[0]
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
        rets[index] = rets[index] + 1
        if index==answer:
            oks[index] = oks[index] +1
            ca = ca + 1
        #
    #
    accuracy = float(ca) / float(eval_size)
    #
    if debug:
        print "---------------------------------"
        print "result : %d / %d" % (ca, eval_size)
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
    return accuracy
#
#
#
