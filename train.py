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
def weight_shift_mode(r, li, ni, ii, entropy, mode):
    #print "li=%d, ni=%d, ii=%d" % (li, ni, ii)
    layer = r.getLayerAt(li)
    wp = layer.get_weight_property(ni, ii) # default : 0
    lock = layer.get_weight_lock(ni, ii)   # default : 0
    wi = layer.get_weight_index(ni, ii)
    wi_alt = wi
    #
    if lock>0:
        return entropy, 0
    #
    if wp!=mode and wp!=0:
        return entropy, 0
    #
    if mode>0: # heat
        if wi==core.WEIGHT_INDEX_MAX:
            layer.set_weight_property(ni, ii, 0)
            layer.set_weight_index(ni, ii, wi-1)
            if r._gpu:
                layer.update_weight()
                r.propagate()
                entropy = r.get_cross_entropy()
            else:
                entropy = r._remote.update(li, ni, ii, wi-1)
            #
            return entropy, 1
        #
    else: # cool
        if wi==core.WEIGHT_INDEX_MIN:
            layer.set_weight_property(ni, ii, 0)
            layer.set_weight_index(ni, ii, wi+1)
            if r._gpu:
                layer.update_weight()
                r.propagate()
                entropy = r.get_cross_entropy()
            else:
                entropy = r._remote.update(li, ni, ii, wi+1)
            #
            return entropy, 1
        #
    #
    #
    #
    wi_alt = wi + mode
    entropy_alt = entropy
    if r._gpu:
        r.propagate(li, ni, ii, wi_alt, 0)
        entropy_alt = r.get_cross_entropy()
    else:
        entropy_alt = r._remote.set_alt(li, ni, ii, wi_alt)
    #
    if  entropy_alt<entropy:
        layer.set_weight_property(ni, ii, mode)
        layer.set_weight_index(ni, ii, wi_alt)
        if r._gpu:
            layer.update_weight()
        else:
            entropy_alt = r._remote.update(li, ni, ii, wi_alt)
        #
        return entropy_alt, 1
    #
    layer.set_weight_property(ni, ii, 0)
    #
    return entropy, 0
#
#
#
def weight_loop(it, r, limit, divider, entropy, layer, li, ni, direction, epoc=0):
    cnt = 0
    num_w = layer.get_num_input()#_num_input
    w_p = num_w
    if num_w>divider:
        w_p = num_w/divider
    #
    for p in range(w_p):
        ii = random.randrange(num_w)
        entropy, ret = weight_shift_mode(r, li, ni, ii, entropy, direction)
        if ret>0:
            cnt = cnt + ret
            if direction>0:
                print "+[%d|%d] L=%d, N=%d, W=%d, %d/%d, %d: CE:%f" % (it, epoc, li, ni, ii, p, w_p, cnt, entropy)
            else:
                print "-[%d|%d] L=%d, N=%d, W=%d, %d/%d, %d: CE:%f" % (it, epoc, li, ni, ii, p, w_p, cnt, entropy)
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
def node_loop(it, r, limit, divider, entropy, layer, li, direction, epoc=0):
    cnt = 0
    num_node = layer.get_num_node()#_num_node
    num_w = layer._num_input
    #
    node_index_list = list(range(num_node))
    random.shuffle(node_index_list)
    nc = 0
    for ni in node_index_list:
        entropy, ret = weight_loop(it, r, limit, divider, entropy, layer, li, ni, direction, epoc)
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
def layer_loop(it, r, limit, reverse, divider, direction, epoc=0):
    cnt = 0
    #
    if r._gpu:
        r.propagate()
        entropy = r.get_cross_entropy()
        print entropy
    else:
        entropy = r._remote.evaluate()
    #
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
        entropy, ret = node_loop(it, r, limit, divider, entropy, layer, li, direction, epoc)
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
def train(it, r, limit, epoc=0):
    divider = 64#4
    entropy = 0.0
    reverse = 0
    w_list = []
    t_cnt = 0
    h_cnt = 0
    c_cnt = 0
    #
    direction = 1
    entropy, h_cnt = layer_loop(it, r, limit, reverse, divider, direction, epoc)
    direction = -1
    entropy, c_cnt = layer_loop(it, r, limit, reverse, divider, direction, epoc)
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
            save_path = "./debug/%04d-%f.csv" % (i, ce)
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
def train_minibatch(r, package, mini_batch_size, num, epoc):
    limit = 0.000001
    package.load_batch()
    batch_size = package._train_batch_size
    data_size = package._image_size
    num_class = package._num_class
    #
    data_array = np.zeros((mini_batch_size, data_size), dtype=np.float32)
    class_array = np.zeros(mini_batch_size, dtype=np.int32)
    #
    r.prepare(mini_batch_size, data_size, num_class)
    #
    print ">>mini_batch_size(%d)" % (mini_batch_size)
    #
    start_time = time.time()
    for j in range(num):
        for i in range(mini_batch_size):
            bi = random.randrange(batch_size)
            data_array[i] = package._train_image_batch[bi]
            class_array[i] = package._train_label_batch[bi]
        #
        #r.set_batch(data_array, class_array, 0, mini_batch_size, data_size, num_class)
        r.set_data(data_array, data_size, class_array, mini_batch_size)
        for k in range(epoc):
            entropy, h_cnt, c_cnt = train(j, r, limit, k)
            r.export_weight_index(package._wi_csv_path)
    #
    elapsed_time = time.time() - start_time
    t = format(elapsed_time, "0")
    print "time = %s" % (t)
    #
#
#
#
def train_minibatch_preset(r, package, mini_batch_size, num, epoc):
    limit = 0.000001
    package.load_batch()
    batch_size = package._train_batch_size
    data_size = package._image_size
    num_class = package._num_class
    #
    data_array = np.zeros((mini_batch_size, data_size), dtype=np.float32)
    class_array = np.zeros(mini_batch_size, dtype=np.int32)
    #
    r.prepare(mini_batch_size, data_size, num_class)
    #
#    if r._gpu:
#        r.init_mem(mini_batch_size, data_size, num_class)
#    else:
#        pass
    #
    print ">> mini_batch_size(%d)" % (mini_batch_size)
    #
    start_time = time.time()
    for j in range(num):
        if r._gpu:
            load_path = "../ldnn_config/%s/mini/%d/%03d.pickle" % (package._name, mini_batch_size, j)
            print load_path
            random_index = util.pickle_load(load_path)
            print random_index
            for i in range(mini_batch_size):
                bi = random_index[i]
                print bi
                data_array[i] = package._train_image_batch[bi]
                class_array[i] = package._train_label_batch[bi]
            #
            r.set_data(data_array, data_size, class_array, mini_batch_size)
        else:
            r._remote.set_batch(j)
        #
        entropy = 0.0
        for k in range(epoc):
            entropy, h_cnt, c_cnt = train(j, r, limit)
            r.export_weight_index(package._wi_csv_path)
            #
            if entropy<limit:
                print "exit iterations"
                break
            #
        #
        if entropy<limit:
            print "exit iterations"
            break
        #
    #
    elapsed_time = time.time() - start_time
    t = format(elapsed_time, "0")
    print "time = %s" % (t)
#
