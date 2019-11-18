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
# constant values
#
#TRAIN_BASE_PATH  = "./data/train/"
#TRAIN_BATCH_PATH = "./train_batch.pickle"
#TEST_BASE_PATH   = "./data/test/"
#TEST_BATCH_PATH = "./test_batch.pickle"
#TRAIN_IMAGE_PATH  = "./MNIST/train-images-idx3-ubyte"
#TRAIN_LABEL_PATH = "./MNIST/train-labels-idx1-ubyte"
#TEST_IMAGE_PATH   = "./MNIST/t10k-images-idx3-ubyte"
#TEST_LABEL_PATH   = "./MNIST/t10k-labels-idx1-ubyte"

#MNIST_IMAGE_WIDTH  = 28
#MNIST_IMAGE_HEIGHT = 28
#MNIST_IMAGE_SIZE   = MNIST_IMAGE_WIDTH*MNIST_IMAGE_HEIGHT
#NUM_OF_CLASS     = 10    # 0,1,2,3,4,5,6,7,8,9
#NUM_OF_SAMPLES   = 5000  # must be 5,000 per a class
#NUM_OF_TEST      = 500
#TRAIN_BATCH_SIZE  = 60000
#TEST_BATCH_SIZE   = 10000
#IMAGE_HEADER_SIZE = 16
#LABEL_HEADER_SIZE  = 8

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
def evaluate(r, batch_size):
    #return r.get_cross_entropy()
    infs = r.get_inference()
    
    sum = 0.0
    for i in range(batch_size):
        data_class = r._batch_class[i]
        labels[data_class] = 1.0
        #print labels
        inf = infs[i]
        #print inf
        mse = util.cross_emtropy_error_2(inf, len(inf), labels, len(labels))
        sum = sum + mse
        labels[data_class] = 0.0

    return sum/float(batch_size)
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
        #print inf
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
def weight_shift_mode(r, li, ni, ii, mse_base, mode):
    layer = r.getLayerAt(li)
    wp = layer.get_weight_property(ni, ii) # default : 0
    lock = layer.get_weight_lock(ni, ii)   # default : 0
    wi = layer.get_weight_index(ni, ii)
    wi_alt = wi
    #
    if lock>0:
        return mse_base, 0
    #
    if mode>0: # heat
        if wp<0:
            return mse_base, 0
        #
        if wi==core.WEIGHT_INDEX_MAX:
            layer.set_weight_property(ni, ii, 0)
            #
            layer.set_weight_index(ni, ii, wi-1)
            layer.update_weight_gpu()
            r.propagate()
            mse_base = r.get_cross_entropy()
            return mse_base, 1
            #
#            wi_alt = wi -1
#            r.propagate(li, ni, ii, wi_alt, 0)
#            mse_alt = r.get_cross_entropy()
#            if  mse_alt<mse_base:
#                layer.set_weight_property(ni, ii, -1)
#                layer.set_weight_index(ni, ii, wi_alt)
#                layer.update_weight_gpu()
#                return mse_base, 1
            #
#            layer.set_weight_lock(ni, ii, 1)
#            return mse_base, 0
        #
    else:
        if wp>0:
            return mse_base, 0
        #
        if wi==core.WEIGHT_INDEX_MIN:
            layer.set_weight_property(ni, ii, 0)
            #
            layer.set_weight_index(ni, ii, wi+1)
            layer.update_weight_gpu()
            r.propagate()
            mse_base = r.get_cross_entropy()
            return mse_base, 1
        #
    #
    wi_alt = wi + mode
    r.propagate(li, ni, ii, wi_alt, 0)
    mse_alt = r.get_cross_entropy()
    #
    if  mse_alt<mse_base:
        layer.set_weight_property(ni, ii, mode)
        layer.set_weight_index(ni, ii, wi_alt)
        layer.update_weight_gpu()
        return mse_alt, 1
    #
#    if wp!=mode:
#        layer.set_weight_lock(ni, ii, 1)
#    else:
    layer.set_weight_property(ni, ii, 0)
    #
    return mse_base, 0
#
#
#
def train(it, r, limit):
    divider = 4
    t_cnt = 0
    h_cnt = 0
    c_cnt = 0
    #labels = np.zeros(NUM_OF_CLASS, dtype=np.float32)
    w_list = []
    #
    r.propagate()
    mse_base = r.get_cross_entropy()
    #
    c = r.countLayers()
    for li in range(1, c):
    #for li in range(c-1, 0, -1):
        layer = r.getLayerAt(li)
        num_node = layer._num_node
        num_w = layer._num_input
        #
        node_index_list = list(range(num_node))
        random.shuffle(node_index_list)
        nc = 0
        for ni in node_index_list:
            nc = nc + 1
            w_p = num_w/divider
            for p in range(w_p):
                ii = random.randrange(num_w)
                mse_base, ret = weight_shift_mode(r, li, ni, ii, mse_base, 1)
                if ret>0:
                    print "[%d] H=%d/%d, N(%d/%d), W(%d/%d) : W(%d,%d,%d), CE:%f" % (it, h_cnt, t_cnt, nc, num_node, p, w_p, li, ni, ii, mse_base)
                #print "    %f" % mse_base
                #
                h_cnt = h_cnt + ret
                t_cnt = t_cnt +1
                if mse_base<limit:
                    print "exit iterations"
                    return h_cnt, c_cnt, mse_base
                #
            #
        #
    #
    t_cnt = 0
    c = r.countLayers()
    for li in range(1, c):
        layer = r.getLayerAt(li)
        num_node = layer._num_node
        num_w = layer._num_input
        #
        node_index_list = list(range(num_node))
        random.shuffle(node_index_list)
        nc = 0
        for ni in node_index_list:
            nc = nc + 1
            w_p = num_w/divider
            for p in range(w_p):
                ii = random.randrange(num_w)
                mse_base, ret = weight_shift_mode(r, li, ni, ii, mse_base, -1)
                if ret>0:
                    print "[%d] C=%d/%d, N(%d/%d), W(%d/%d) : W(%d,%d,%d), CE:%f" % (it, c_cnt, t_cnt, nc, num_node, p, w_p, li, ni, ii, mse_base)
                #print "    %f" % mse_base
                #
                c_cnt = c_cnt + ret
                t_cnt = t_cnt + 1
                if mse_base<limit:
                    print "exit iterations"
                    return h_cnt, c_cnt, mse_base
                #
            #
        #
    #
    return h_cnt, c_cnt, mse_base
#
#
#
def loop(it, r, package, debug=0):
    h_cnt_list = []
    c_cnt_list = []
    ce_list = []
    #
    limit = 0.000001
    pre_ce = 0.0
    lim_cnt = 0
    #
    start_time = time.time()
    #
    for i in range(it):
        h_cnt, c_cnt, ce = train(i, r, limit)
        #
        h_cnt_list.append(h_cnt)
        c_cnt_list.append(c_cnt)
        ce_list.append(ce)
        # debug
        if debug==1:
            save_path = "./debug/wi.csv.%f" % ce
            r.export_weight_index(save_path)
        #
        r.export_weight_index(package._wi_csv_path)
        if pre_ce == ce:
            lim_cnt = lim_cnt + 1
            if lim_cnt>5:
                print "locked with local optimum"
                print "exit iterations"
                break
            #
        #
        if ce<limit:
            print "exit iterations"
            break
        #
        #r.reset_weight_property()
        #r.unlock_weight_all()
        pre_ce = ce
    #
    elapsed_time = time.time() - start_time
    t = format(elapsed_time, "0")
    print "time = %s" % (t)
    #
    k = len(h_cnt_list)
    for j in range(k):
        print "%d, %d, %d, %f," % (j, h_cnt_list[j], c_cnt_list[j], ce_list[j])
    
    #
#
#
#
def init_WI(r, batch, batch_size, data_size):
    r.set_batch(batch, batch_size, data_size, 10)
    #labels = np.zeros(NUM_OF_CLASS, dtype=np.float32)
    #
    for i in range(100):
        r.init_weight()
        r.update_weight()
        r.propagate()
        cr =  evaluate(r, batch_size)
        print cr
        #
        save_path = "./wi/wi.csv.%f" % cr
        r.export_weight_index(save_path)
    #
#
#
#
def main():
    argvs = sys.argv
    argc = len(argvs)
    #
    debug = 1
    it = 20*20
    package = -1
    mode = -1
    #
    batch_size = 5000
    #num_class = 10
    #data_size = 32*32*3

    print "0 : MNIST : train"
    print "1 : MNIST : test"
    print "2 : MNIST : self-test"
    print "3 : CIFAR-10 : train"
    print "4 : CIFAR-10 : test"
    print "5 : CIFAR-10 : self-test"
    print "6 : "
    print "7 : "
    print "8 : "
    print "9 : "
    menu = get_key_input("input command >")
    if menu==0:
        # MNIST : train
        package = 0
        mode = 0
    elif menu==1:
        # MNIST : test
        package = 0
        mode = 1
    elif menu==2:
        # MNIST : self-test
        package = 0
        mode = 2
    elif menu==3:
        # CIFAR-10 : train
        package = 1
        mode = 0
    elif menu==4:
        # CIFAR-10 : test
        package = 1
        mode = 1
    elif menu==5:
        # CIFAR-10 : self-test
        package = 1
        mode = 2
    #
    if package<0 or mode<0:
        print "bad menu select"
        return 0
    #
    # GPU
    #
    platform_id = 0
    device_id = 1 # 0 : AMD Server, 1 : Intel on MBP 2 : eGPU (AMD Radeon Pro 580)
    #
    my_gpu = gpu.Gpu(platform_id, device_id)
    my_gpu.set_kernel_code()
    #
    path = WEIGHT_INDEX_CSV_PATH
    if argc==2:
        path = argvs[1]
    #
    if package==0: # MNIST
        mnist = util.Mnist(mode)
        r = mnist.setup_dnn(my_gpu)
        if r is None:
            print "fatal DNN error"
            return 0
        #
        if mode==0: # train
            r.set_batch(mnist._data, mnist._labels, batch_size, util.MNIST_IMAGE_SIZE, util.MNIST_NUM_CLASS)
            loop(it, r, mnist, debug)
        elif mode==1: # test
            batch_size = util.MNIST_TEST_BATCH_SIZE
            r.set_batch(mnist._data, mnist._labels, batch_size, util.MNIST_IMAGE_SIZE, util.MNIST_NUM_CLASS)
            test(r)
        elif mode==2: # self-test
            r.set_batch(mnist._data, mnist._labels, batch_size, util.MNIST_IMAGE_SIZE, util.MNIST_NUM_CLASS)
            test(r)
        #
    elif package==1: # CIFAR-10
        cifar10 = util.Cifar10(mode)
        r = cifar10.setup_dnn(my_gpu)
        if r is None:
            print "fatal DNN error"
            return 0
        #
        if mode==0: # train
            r.set_batch(cifar10._data, cifar10._labels, batch_size, util.CIFAR10_IMAGE_Y_SIZE, util.CIFAR10_NUM_CLASS)
            loop(it, r, cifar10, debug)
        elif mode==1: # test
            batch_size = util.MNIST_TEST_BATCH_SIZE
            r.set_batch(cifar10._data, cifar10._labels, batch_size, util.CIFAR10_IMAGE_Y_SIZE, util.CIFAR10_NUM_CLASS)
            test(r)
        elif mode==2: # self-test
            r.set_batch(cifar10._data, cifar10._labels, batch_size, util.CIFAR10_IMAGE_Y_SIZE, util.CIFAR10_NUM_CLASS)
            test(r)
        #
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
