#! /usr/bin/python
# -*- coding: utf-8 -*-
#

#
# LDNN : lesser Deep Neural Network
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

from multiprocessing import Process
from multiprocessing import Queue
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
# constant values
#
TRAIN_BASE_PATH  = "./data/train/"
TRAIN_BATCH_PATH = "./train_batch.pickle"
TEST_BASE_PATH   = "./data/test/"
TEST_BATCH_PATH = "./test_batch.pickle"
NETWORK_PATH     = "./network.pickle"
PROCEEDED_PATH   = "./proceeded.pickle"

TRAIN_IMAGE_PATH  = "./MNIST/train-images-idx3-ubyte"
TRAIN_LABEL_PATH = "./MNIST/train-labels-idx1-ubyte"
TEST_IMAGE_PATH   = "./MNIST/t10k-images-idx3-ubyte"
TEST_LABEL_PATH   = "./MNIST/t10k-labels-idx1-ubyte"

MNIST_IMAGE_WIDTH  = 28
MNIST_IMAGE_HEIGHT = 28
MNIST_IMAGE_SIZE   = MNIST_IMAGE_WIDTH*MNIST_IMAGE_HEIGHT

TRAIN_BATCH_SIZE  = 60000
TEST_BATCH_SIZE   = 10000
IMAGE_HEADER_SIZE = 16
LABEL_HEADER_SIZE  = 8


WEIGHT_INDEX_CSV_PATH   = "./wi.csv"

NUM_OF_CLASS     = 10    # 0,1,2,3,4,5,6,7,8,9
NUM_OF_SAMPLES   = 5000  # must be 5,000 per a class
NUM_OF_TEST      = 500
#
#
#
def setup_dnn(path, my_gpu):
    r = core.Roster()
    r.set_gpu(my_gpu)

    input_layer = r.add_layer(0, 784, 784)
    hidden_layer_1 = r.add_layer(1, 784, 32)
    hidden_layer_2 = r.add_layer(1, 32, 32)
    output_layer = r.add_layer(2, 32, 10)
    
    if os.path.isfile(WEIGHT_INDEX_CSV_PATH):
        r.import_weight_index(WEIGHT_INDEX_CSV_PATH)
    else:
        print "init weights"
        r.init_weight()

    if my_gpu:
        r.update_weight()
    
    #r.export_weight_index(WEIGHT_INDEX_CSV_PATH)
    return r
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
def evaluate(r, batch_size, labels):
    infs = r.get_inference()
    sum = 0.0
    for i in range(batch_size):
        data_class = r._batch_class[i]
        labels[data_class] = 1.0
        inf = infs[i]
        #print inf
        #
        mse =  util.cross_emtropy_error_2(inf, len(inf), labels, len(labels))
        #if mse==np.nan:
        #    #print mse
        #    #mse = 100.0
        #    print mse
        #    sum = sum + 100.0
        #else:
        #    sum = sum + mse
    
        #mse =  util.cross_emtropy_error_fast(inf, labels, data_class)
        #mse =  util.cross_emtropy_error_fast(inf, data_class)
        #print mse
        #
        #mse = util.mean_squared_error(inf, len(inf), labels, len(labels))
        #
        sum = sum + mse
        labels[data_class] = 0.0

    #print infs[0]
    return sum/float(batch_size)
#
#
#
def weight_shift(r, batch, batch_size, li, ni, ii, mse_base, labels):
    layer = r.getLayerAt(li)
    lock = layer.get_weight_lock(ni, ii)
    if lock>1:
        print "  locked(%d, %d)" % (ni, li)
        return mse_base, 0
    
    wp = layer.get_weight_property(ni, ii)
    wi = layer.get_weight_index(ni, ii)
    wi_alt = wi
    
    if wp==0:
        if wi==core.WEIGHT_INDEX_MAX:
            layer.set_weight_property(ni, ii, -1)
            wp = - 1
        else:
            layer.set_weight_property(ni, ii,  1)
            wp = 1
    
        #layer.set_weight_lock(ni, ii, 1)
    
        wi_alt = wi + wp
        r.propagate(li, ni, ii, wi+wp, 0)
        mse_alt = evaluate(r, batch_size, labels)
        print "  - %d > %d : %f > %f" % (wi, wi + wp , mse_base, mse_alt)
        if mse_alt<mse_base:
            layer.set_weight_index(ni, ii, wi + wp)
            layer.update_weight_gpu()
            return mse_alt, 1
        else:
            print "  reversed A %d" % (wi)
            layer.set_weight_property(ni, ii, wp*-1)
            layer.set_weight_lock(ni, ii, 1)
            return mse_base, 0

    if wi==core.WEIGHT_INDEX_MAX:
        layer.set_weight_property(ni, ii, -1)
        wp = -1
        print "  reversed B %d" % (wi)
        layer.set_weight_lock(ni, ii, lock+1)
        return mse_base, 0
    elif wi==core.WEIGHT_INDEX_MIN:
        layer.set_weight_property(ni, ii, 1)
        wp = 1
        print "  reversed C %d" % (wi)
        layer.set_weight_lock(ni, ii, lock+1)
        if lock>0:
            print "  lock"
        return mse_base, 0

    wi_alt = wi + wp
    r.propagate(li, ni, ii, wi_alt, 0)
    mse_alt = evaluate(r, batch_size, labels)
    print "  = %d > %d : %f > %f" % (wi, wi_alt , mse_base, mse_alt)
    if  mse_alt<mse_base:
        layer.set_weight_index(ni, ii, wi_alt)
        layer.update_weight_gpu()
        return mse_alt, 1
    #
    layer.set_weight_lock(ni, ii, lock+1)
    return mse_base, 0
#
#
#
def train_mode(it, r, batch, batch_size, data_size):
    labels = np.zeros(NUM_OF_CLASS, dtype=np.float32)
    cnt_update = 0
    r.set_batch(batch, batch_size, data_size)
    #
    start_time = time.time()
    r.propagate()
    mse_base = evaluate(r, batch_size, labels)
    print mse_base
    
    #
    #weight_list = r.get_weight_list()
    cnt0 = 0
    c = r.countLayers()
    #for li in range(1, c):
    for li in range(c-1, 0, -1):
        layer = r.getLayerAt(li)
        num_node = layer._num_node
        num_w = layer._num_input
        #
        node_index_list = list(range(num_node))
        random.shuffle(node_index_list)
        for ni in node_index_list:
            patial = num_w/5/li
            for p in range(patial):
                ii = random.randrange(num_w)
                print "%d : %d [%d] (%d, %d, %d) %f" % (it, cnt0, cnt_update, li, ni, ii, mse_base)
                mse_base, c = weight_shift(r, batch, batch_size, li, ni, ii, mse_base, labels)
                #
                cnt_update = cnt_update + c
                cnt0 = cnt0 + 1
            #
    #
    elasped_time = time.time() - start_time
    t = format(elasped_time, "0")
    print "[elasped time] %s" % (t)
    return cnt_update
#
#
#
def test_mode(platform_id, device_id, batch, batch_size):
    print ">>batch test mode"
    my_gpu = gpu.Gpu(platform_id, device_id)
    my_gpu.set_kernel_code()
    r = setup_dnn(NETWORK_PATH, my_gpu)
    if r is None:
        print "fatal DNN error"
        return 0

    print "batch_size=%d" % batch_size
    dist = [0,0,0,0,0,0,0,0,0,0] # data_class
    rets = [0,0,0,0,0,0,0,0,0,0] # result of infs
    oks  = [0,0,0,0,0,0,0,0,0,0] # num of correct
    #
    print "batch_size=%d" % batch_size
    start_time = time.time()
    r.set_batch(batch, batch_size, 28*28, 0)
    r.propagate(-1, -1, -1, -1, 0)
    infs = r.get_inference()
    #
    ca = 0
    for i in range(batch_size):
        dist[r._batch_class[i]] = dist[r._batch_class[i]] + 1
        inf = infs[i]
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
    
    print "---------------------------------"
    #
    elapsed_time = time.time() - start_time
    t = format(elapsed_time, "0")
    print "test time = %s" % (t)
#
#
#
def main():
    argvs = sys.argv
    argc = len(argvs)
    #
    platform_id = 0
    device_id = 1 # 0 : AMD Server, 1 : Intel on MBP, 2 : eGPU (AMD Radeon Pro 580)
    batch_size = 1500
    data_size = 28*28
    #
    # GPU
    #
#    platform_id = 0
#    device_id = 1 # 0 : AMD Server, 1 : Intel on MBP, 2 : eGPU (AMD Radeon Pro 580)
#    my_gpu = gpu.Gpu(platform_id, device_id)
#    my_gpu.set_kernel_code()
#    r_0 = setup_dnn(NETWORK_PATH, my_gpu)
#    if r_0 is None:
#        print "fatal DNN error"
#        return 0
#
#    #
#    my_gpu = gpu.Gpu(platform_id, 2)
#    my_gpu.set_kernel_code()
#    r_1 = setup_dnn(NETWORK_PATH, my_gpu)
#    if r_1 is None:
#        print "fatal DNN error"
#        return 0
#
#    print "a"
    batch = util.pickle_load(TEST_BATCH_PATH)
    batch_size = 100

    p_0 = Process(target=test_mode, args=(platform_id, device_id, batch, batch_size,))
    p_0.start()

    device_id = 2
    b2 = batch[100:200]
    p_1 = Process(target=test_mode, args=(platform_id, device_id, b2, batch_size,))
    p_1.start()
    print("Process started.")
    
    p_0.join()
    p_1.join()
    print("Process joined.")

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
