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
                layer.update_weight_gpu()
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
                layer.update_weight_gpu()
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
            layer.update_weight_gpu()
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
    #
    if r._gpu:
        r.propagate()
        entropy = r.get_cross_entropy()
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
    w_list = []
    t_cnt = 0
    h_cnt = 0
    c_cnt = 0
    #
    direction = 1
    entropy, h_cnt = layer_loop(it, r, limit, reverse, divider, direction)
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
