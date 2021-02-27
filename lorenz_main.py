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
import pyopencl as cl
#
# LDNN Modules
#
import util
import package
import core
import gpu
import train
import test
#
sys.setrecursionlimit(10000)
#
#
#
def test(r):
    data_size = 3
    num_class = 3
    bsize = 10
    data_path = "./lorenz/test_in.pickle"
    label_path = "./lorenz/test_out.pickle"
    data = util.pickle_load(data_path)
    label = util.pickle_load(label_path)
    prepare(r, bsize, data, label)
    #
    r.propagate()
    ce = r.get_cross_entropy()
    print("CE = %f" % (ce))
    #
    infs = r.get_inference()
    print infs
    #
    ce = r.get_cross_entropy()
    print("CE = %f" % (ce))
    #
    layer = r.getLayerAt(1)
    r._gpu.copy(layer._output_array, layer._gpu_output)
    print(layer._output_array[0])
    print(layer._weight_matrix[0])

def prepare(r, bsize, data, label):
    data_size = 3
    num_class = 3
    #
    dist = np.zeros(num_class, dtype=np.int32)
    rets = np.zeros(num_class, dtype=np.int32)
    oks = np.zeros(num_class, dtype=np.int32)
    #
    r.prepare(bsize, data_size, num_class)
    #
    data_array = np.zeros((bsize, data_size), dtype=np.float32)
    label_array = np.zeros((bsize, num_class), dtype=np.float32)
    for j in range(bsize):
        for i in range(data_size):
            data_array[j][i] = data[j][i]
            label_array[j][i] = label[j][i]
        #
    #
    r.set_data(data_array, data_size, label_array, bsize)
#    r.propagate(-1, -1, -1, -1, 0)
#    ce = r.get_cross_entropy()
#    print("CE = %f" % (ce))

def weight_ops(r, w_list, attack_i, mode):
    w = w_list[attack_i]
    li = w[0]
    ni = w[1]
    ii = w[2]
    #
    layer = r.getLayerAt(li)
    wi = layer.get_weight_index(ni, ii)
    wi_alt = wi
    maximum = core.WEIGHT_INDEX_MAX
    minimum = core.WEIGHT_INDEX_MIN
    #
    if mode>0: # heat
        if wi<maximum:
            wi_alt = wi + 1
        #
    else:
        if wi>minimum:
            wi_alt = wi - 1
        #
    #
    return wi, wi_alt

def make_w_list(r):
    w_list  = []
    c = r.countLayers()
    for li in range(1, c):
        layer = r.getLayerAt(li)
        type = layer.get_type()
        if type==core.LAYER_TYPE_HIDDEN or type==core.LAYER_TYPE_OUTPUT or type==core.LAYER_TYPE_CONV_4:
            for ni in range(layer._num_node):
                for ii in range(layer._num_input):
                    w_list.append((li, ni, ii))
                #
            #
        #
    #
    return w_list

def multi_attack(r, ce, mode=1, kt=0):
    w_list = make_w_list(r)
    w_num = len(w_list)
    attack_num = int(w_num/100*kt) # 1%
    if attack_num<1:
        attack_num = 1
    #
    attack_num = 1
    #
    attack_list = []
    for i in range(attack_num*10):
        if i>=attack_num:
            break
        #
        attack_i = random.randrange(w_num)
        #
        w = w_list[attack_i]
        #
        li = w[0]
        ni = w[1]
        ii = w[2]
        wi, wi_alt = weight_ops(r, w_list, attack_i, mode)
        if wi!=wi_alt:
            attack_list.append((attack_i, wi, wi_alt))
        #
    #
    for wt in attack_list:
        attack_i = wt[0]
        wi = wt[1]
        wi_alt = wt[2]
        w = w_list[attack_i]
        li = w[0]
        ni = w[1]
        ii = w[2]
        #
        layer = r.get_layer_at(li)
        layer.set_weight_index(ni, ii, wi_alt)
    #
    c = r.countLayers()
    for li in range(c):
        layer = r.get_layer_at(li)
        layer.update_weight()
    #
    r.propagate()
    ce_alt = r.get_cross_entropy()
#    print("    ce_alt=%f" % (ce_alt))
    if ce_alt<ce:
        ce = ce_alt
    else:
        for wt in attack_list:
            attack_i = wt[0]
            wi = wt[1]
            wi_alt = wt[2]
            #
            w = w_list[attack_i]
            #
            li = w[0]
            ni = w[1]
            ii = w[2]
            #
            layer = r.get_layer_at(li)
            layer.set_weight_index(ni, ii, wi)
        #
        for li in range(c):
            layer = r.get_layer_at(li)
            layer.update_weight()
        #
    #
    return ce

def loop(r):
    #
    r.propagate()
    ce = r.get_cross_entropy()
    print("CE = %f" % (ce))
    #
    it = 50
    kt = [1, 0.1, 0.01, 0.001, 0.01, 0.1]
        #kt = [1, 0.5, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 0.0009765625,]
        # 1*2**(-k)
    k = 0
    for j in range(it):
        for i in range(100):
            ce = multi_attack(r, ce, 1, kt[k])
            print("%d : H : %d : %f, %f" % (j, i, ce, kt[k]))
        #
        for i in range(100):
            ce = multi_attack(r, ce, 0, kt[k])
            print("%d : C : %d : %f, %f" % (j, i, ce, kt[k]))
        #
        r.export_weight("./lorenz/wi.csv")
        if k==len(kt)-1:
            k = 0
        else:
            k = k+1
        #
    #
    return 0

def train(r):
    data_size = 3
    num_class = 3
    bsize = 100
    data_path = "./lorenz/train_in.pickle"
    label_path = "./lorenz/train_out.pickle"
    data = util.pickle_load(data_path)
    label = util.pickle_load(label_path)
    prepare(r, bsize, data, label)
    #
    loop(r)

def main():
    argvs = sys.argv
    argc = len(argvs)
    print(argc)
    #
    platform_id = 0
    device_id = 1
    train_size = 2001
    test_size = 500
    mode = 0
    #
    my_gpu = gpu.Gpu(platform_id, device_id)
    my_gpu.set_kernel_code()
    r = core.Roster()
    r.set_gpu(my_gpu)
    
    c = r.countLayers()
    input = core.InputLayer(c, 3, 3, None, my_gpu)
    r.layers.append(input)
    # 1 : hidden
    c = r.countLayers()
    hidden_1 = core.HiddenLayer(c, 3, 8, input, my_gpu)
    r.layers.append(hidden_1)
    # 2 : hidden
    c = r.countLayers()
    hidden_2 = core.HiddenLayer(c, 8, 8, hidden_1, my_gpu)
    r.layers.append(hidden_2)
    # 3 : hidden
    c = r.countLayers()
    hidden_3 = core.HiddenLayer(c, 8, 8, hidden_2, my_gpu)
    r.layers.append(hidden_3)
    # 4 : hidden
    c = r.countLayers()
    hidden_4 = core.HiddenLayer(c, 8, 8, hidden_3, my_gpu)
    r.layers.append(hidden_4)
    # 3 : output
    c = r.countLayers()
    output = core.OutputLayer(c, 8, 3, hidden_4, my_gpu)
    r.layers.append(output)
    #
    if os.path.isfile("./lorenz/wi.csv"):
        r.import_weight("./lorenz/wi.csv")
    else:
        r.init_weight()
        r.export_weight("./lorenz/wi.csv")
    #
    r.update_weight()
    #
    if mode==0:
        train(r)
    elif mode==1:
        test(r)
    else:
        print("mode error : %d" % (mode))
        return 0
    #
    return 0
#
#
#
if __name__=='__main__':
    print(">> start")
    sts = main()
    print(">> end")
    print("\007")
    sys.exit(sts)
#
#
#
