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
    if wp!=mode and wp!=0:
        return mse_base, 0
    #
    if mode>0: # heat
        if wi==core.WEIGHT_INDEX_MAX:
            layer.set_weight_property(ni, ii, 0)
            #
            layer.set_weight_index(ni, ii, wi-1)
            layer.update_weight_gpu()
            r.propagate()
            mse_base = r.get_cross_entropy()
            return mse_base, 1
        #
    else: # cool
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
    #
    #
    wi_alt = wi + mode
    r.propagate(li, ni, ii, wi_alt, 0)
    mse_alt = r.get_cross_entropy()
    if  mse_alt<mse_base:
        layer.set_weight_property(ni, ii, mode)
        layer.set_weight_index(ni, ii, wi_alt)
        layer.update_weight_gpu()
        return mse_alt, 1
    #
    layer.set_weight_property(ni, ii, 0)
    #
    return mse_base, 0
#
#
#
def weight_loop(it, r, limit, divider, entropy, layer, li, ni, direction):
    cnt = 0
    num_w = layer._num_input
    w_p = num_w/divider
    #
    for p in range(w_p):
        ii = random.randrange(num_w)
        entropy, ret = weight_shift_mode(r, li, ni, ii, entropy, direction)
        if ret>0:
            cnt = cnt + ret
            print "[%d] L=%d, N=%d, W=%d, %d/%d, %d: CE:%f" % (it, li, ni, ii, p, w_p, cnt, entropy)
        #
        if entropy<limit:
            print "reach to the limit(%f), exit iterations" %(limit)
            break
        #
    # for p
    return entropy, cnt
#
#
#
def node_loop(it, r, limit, divider, entropy, layer, li, direction):
    cnt = 0
    num_node = layer._num_node
    num_w = layer._num_input
    #
    node_index_list = list(range(num_node))
    random.shuffle(node_index_list)
    nc = 0
    for ni in node_index_list:
        entropy, ret = weight_loop(it, r, limit, divider, entropy, layer, li, ni, direction)
        cnt = cnt + ret
        if entropy<limit:
            print "reach to the limit(%f), exit iterations" %(limit)
            break
        #
    # for ni
    return entropy, cnt
#
#
#
def layer_loop(it, r, limit, reverse, divider, direction):
    cnt = 0
    r.propagate()
    entropy = r.get_cross_entropy()
    c = r.countLayers()
    list_of_layer_index = []
    #
    if reverse==0: # input to output
        for i in range(1, c):
            list_of_layer_index.append(i)
    else: # output to intput
        for i in range(c-1, 0, -1):
            list_of_layer_index.append(i)
    #
    for li in list_of_layer_index:
        layer = r.getLayerAt(li)
        entropy, ret = node_loop(it, r, limit, divider, entropy, layer, li, direction)
        cnt = cnt + ret
        if entropy<limit:
            print "reach to the limit(%f), exit iterations" %(limit)
            break
        #
    # for li
    return entropy, cnt
#
#
#
def train(it, r, limit):
    divider = 4
    entropy = 0.0
    reverse = 0
    direction = 1
    w_list = []
    t_cnt = 0
    h_cnt = 0
    c_cnt = 0
    #
    entropy, h_cnt = layer_loop(it, r, limit, reverse, divider, direction)
    #
    direction = -1
    entropy, c_cnt = layer_loop(it, r, limit, reverse, divider, direction)
    #
    return entropy, h_cnt, c_cnt
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
        #
        ce, h_cnt, c_cnt = train(i, r, limit)
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
#    path = WEIGHT_INDEX_CSV_PATH
#    if argc==2:
#        path = argvs[1]
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
    print "1 : MNIST (2)"
    print "2 : CIFAR-10"
    menu = get_key_input("input command >")
    if menu==0:
        package_id = 0
    elif menu==1:
        package_id = 1
    elif menu==2:
        package_id = 2
    else:
        package_id = 0
    #
    #
    #
    print "0 : train"
    print "1 : test"
    print "2 : self-test"
    print "3 : "
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
    #
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
        loop(it, r, package, debug)
    elif mode==1: # test
        package.load_batch()
        batch_size = package._test_batch_size
        r.set_batch(package._test_image_batch, package._test_label_batch, 0, batch_size, package._image_size, package._num_class, 0)
        test(r)
    elif mode==2: # self-test
        package.load_batch()
        r.set_batch(package._train_image_batch, package._train_label_batch, 0, batch_size, package._image_size, package._num_class, 0)
        test(r)
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
