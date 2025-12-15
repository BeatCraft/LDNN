#! /usr/bin/python
# -*- coding: utf-8 -*-
#

import os
import sys
import time
import math
import numpy as np

#
# LDNN : lesser's Deep Neural Network
#
import core
import util

sys.setrecursionlimit(10000)

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

def regression(r, debug=0):
    pass
                
def classification(r, b, n, debug=0, single=0):
    data_size = b.data_size
    num_class = b.class_num
    batch_size = b.batch_size
    
    print("== classification test ==")
    print("total size", batch_size)
    print("mini batch size", n)
    if single==1:
        print("single test")
    #
    
    mini_batch_num = int(batch_size / n) #mini_batch_size)
    print("mini_batch_num", mini_batch_num)
    
    dist = np.zeros(num_class, dtype=np.int32)
    rets = np.zeros(num_class, dtype=np.int32)
    oks = np.zeros(num_class, dtype=np.int32)
    
    # for single test
    if single==1:
        it = 1
        n = 1
        left = 0
        #b.batch_size = 1
    else:
        it, left = divmod(batch_size, n)
    #
    if left>0:
        print(("error : n(=%d) is not appropriate" % (n)))
    #
    
    elapsed_time = 0.0
    #r.prepare(n, data_size, num_class)
    for i in range(it):
        (data_array, label_list, label_array) = b.get_batch(n, i*n)
        r.reset()
        r.direct_set_data(data_array)
        r.direct_set_label(label_array)
        
        start_time = time.time()
        r.propagate(debug)
        elapsed_time += (time.time() - start_time)
        #infs = r.get_inference()
        answers = r.get_answer()
        #print(answers)
        
        for j in range(n):
            ans = answers[j]
            #print(label_array)
            label = np.argmax(label_array[j])
            rets[ans] = rets[ans] + 1
            dist[label] = dist[label] + 1
            if ans == label:
                oks[ans] = oks[ans] + 1
            #
        #
    #
    ca = sum(oks)
    print_result(ca, batch_size, num_class, dist, rets, oks)
    #
    #elapsed_time = time.time() - start_time
    t = format(elapsed_time, "0")
    print(("time = %s" % (t)))
    print(r.get_cross_entropy())

    print("done")
    return float(ca) / float(batch_size)
    
    
def inference(r, num_class, data, data_size, debug=0):
    #data_array = np.zeros((1, data_size), dtype=np.float32)
    #data_array[0] = data[0]
    data_array = np.array([data,])
    print(data_array.shape)
    class_array = np.zeros(1, dtype=np.int32)
    class_array[0] = 0 # dummy
    #
    r.set_batch(data_size, num_class, data_array, class_array, 1, 0)
    r.propagate(debug)
    answers = r.get_answer()
    return answers[0]
    
def inference2(r, num_class, data, data_size, debug=0):
    data_array = np.array([data,])
    print(data_array.shape)
    class_array = np.zeros(1, dtype=np.int32)
    class_array[0] = 0 # dummy
    #
    r.set_batch(data_size, num_class, data_array, class_array, 1, 0)
    r.propagate(debug)
    inf = r.get_inference()
    #print(inf)
    #r.get_answer()
    
    max_index = -1
    max = -1.0
    for j in range(num_class):
        if inf[0][j]>max:
            max = inf[0][j]
            max_index = j
        #
    #
    return max_index, max
    
def inference3(r, num_class, data, data_size, debug=0):
    data_array = np.array([data,])
    print(data_array.shape)
    class_array = np.zeros(1, dtype=np.int32)
    class_array[0] = 0 # dummy
    #
    r.set_batch(data_size, num_class, data_array, class_array, 1, 0)
    r.propagate(debug)
    inf = r.get_inference()
    #print(inf)
    return inf
    
    
