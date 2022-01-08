#! /usr/bin/python
# -*- coding: utf-8 -*-
#

#
# LDNN : lesser's Deep Neural Network
#

import os, sys, time, math
#from stat import *
import random
#import copy
#import math
#import multiprocessing as mp
import numpy as np
from PIL import Image
#import struct
#import pickle
#import pyopencl as cl
#
# LDNN Modules
#
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import util
#import package
import core
import gpu
#import train
#import test
#
sys.setrecursionlimit(10000)
#
#
#
DATA_SIZE = 32#128
BATCH_SIZE = 2400*4

def make_w_list(r, type_list=None):
    if type_list is None:
        type_list = [core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT, core.LAYER_TYPE_CONV_4]
    #
    w_list  = []
    #r = self._r
    c = r.count_layers()
    for li in range(1, c):
    #for li in range(1, 4): # enc
    #for li in range(4, c): # dec
        layer = r.get_layer_at(li)
        type = layer.get_type()
        
        for t in type_list:
            if type!=t:
                continue
            #
            for ni in range(layer._num_node):
                for ii in range(layer._num_input):
                    w_list.append((li, ni, ii))
                #
            #
        #
    #
    return w_list

def evaluate(r, data_size, batch_size):
    r.propagate()
    r._gpu.mse(r.output._gpu_output, r.input._gpu_output, r._gpu_entropy, data_size, batch_size)
    r._gpu.copy(r._batch_cross_entropy, r._gpu_entropy)
    ce = np.sum(r._batch_cross_entropy)/np.float32(batch_size)
    return ce

def weight_ops(r, attack_i, mode, w_list):
    w = w_list[attack_i]
    li = w[0]
    ni = w[1]
    ii = w[2]
    #
    #r = self._r
    layer = r.get_layer_at(li)
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

def make_attack_list(r, div, mode, w_list):
    w_num = len(w_list)
    attack_num = int(w_num/div)
    if attack_num<1:
        attack_num = 1
    #
    #print(attack_num)
    attack_list = []
    for i in range(attack_num*10):
        if i>=attack_num:
            break
        #
        attack_i = random.randrange(w_num)
        w = w_list[attack_i]
        li = w[0]
        ni = w[1]
        ii = w[2]
        wi, wi_alt = weight_ops(r, attack_i, mode, w_list)
        if wi!=wi_alt:
            attack_list.append((li, ni, ii, wi, wi_alt))
        #
    #
    return attack_list

def undo_attack(r, attack_list):
    for wt in attack_list:
        li = wt[0]
        ni = wt[1]
        ii = wt[2]
        wi = wt[3]
        wi_alt = wt[4]
        #
        layer = r.get_layer_at(li)
        layer.set_weight_index(ni, ii, wi)
    #
    c = r.count_layers()
    for li in range(c):
        layer = r.get_layer_at(li)
        layer.update_weight()
    #
                #(r, ce, w_list, 1, div)
def multi_attack(r, ce, w_list, mode=1, div=0):#, mpi=0, com=None, rank=0, size=0):
    attack_list = make_attack_list(r, div, mode, w_list)
    w_num = len(attack_list)
    #print(len(attack_list))
    for wt in attack_list:
        li = wt[0]
        ni = wt[1]
        ii = wt[2]
        wi = wt[3]
        wi_alt = wt[4]
        layer = r.get_layer_at(li)
        layer.set_weight_index(ni, ii, wi_alt)
    #
    c = r.count_layers()
    for li in range(c):
        layer = r.get_layer_at(li)
        layer.update_weight()
    #
    ce_alt = evaluate(r, DATA_SIZE, BATCH_SIZE)
    #print(ce_alt)
    ret = 0
    if ce_alt<ce:
        ce = ce_alt
        ret = 1
    else:
        undo_attack(r, attack_list)
    #
    return ce, ret

    #w_loop(r, i, 500, d, ce, w_list, lv_min, lv_max, "all")
def w_loop(r, c, n, d, ce, w_list, lv_min, lv_max, label):#, label, mpi=0, com=None, rank=0, size=0):
    level = lv_min
    mode = 1
    #r = self._r
    #pack = self._package
    #
    for j in range(n):
        #
        div = float(d)*float(2**(level))
        #
        cnt = 0
        for i in range(100):
            ce, ret = multi_attack(r, ce, w_list, 1, div)#, mpi, com, rank, size)
            cnt = cnt + ret
            print("[%d|%s] %d : H : %d : %f, %d (%d, %d) %d (%d, %d)" % (c, label, j, i, ce, level, lv_min, lv_max, cnt, 2**(level), int(len(w_list)/div)))
        #
        for i in range(100):
            ce, ret = multi_attack(r, ce, w_list, 0, div)#, mpi, com, rank, size)
            cnt = cnt + ret
            print("[%d|%s] %d : C : %d : %f, %d (%d, %d) %d (%d, %d)" % (c, label, j, i, ce, level, lv_min, lv_max, cnt, 2**(level), int(len(w_list)/div)))
        #
        if mode==1:
            if level==lv_max:
                if level==lv_min:
                    mode = 0
                else:
                    mode = -1
            elif level<lv_max-1:
                if cnt==0:
                    if level==lv_min:
                        lv_min = lv_min + 1
                    #
                #
            #
            level = level + mode
        elif mode==-1:
            if level==lv_min:
                if level==lv_max:
                    mode = 0
                else:
                    mode = 1
                    if cnt==0:
                        lv_min = lv_min + 1
                    #
                #
            #
            level = level + mode
        #
        r.export_weight("./wi.csv")
        path = "./test-%04d.png" % (j)
        save_img(r, path)
        #self.mpi_save(pack.save_path(), mpi, com, rank, size)
    #
    return ce, lv_min, lv_max

def train_loop(r, data_size, batch_size, n=1):
    w_list = None
    ce = 0.0
    ret = 0
    #r = self._r
    #pack = self._package
    ce = evaluate(r, data_size, batch_size)
    print("CE starts with %f" % ce)
    #
    w_list = make_w_list(r, [core.LAYER_TYPE_CONV_4, core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT])
    w_num = len(w_list)
    print(len(w_list))
    #
    d = 100
    lv_min = 0
    lv_max = int(math.log(w_num/d, 2)) + 1
    #
    #return 0
        
    for i in range(n):
        ce, lv_min, lv_min = w_loop(r, i, 500, d, ce, w_list, lv_min, lv_max, "all")
        r.export_weight("./wi.csv")
       # (r, c, n, d, ce, w_list, lv_min, lv_max):
#        #self.mpi_save(pack.save_path(), mpi, com, rank, size)

    #
    return 0

def save_img(r, path):
    r._gpu.copy(r.output._output_array, r.output._gpu_output)
    print(r.output._output_array)
    imgArray = np.reshape(r.output._output_array, (480, 640))
    imgArray = imgArray*255
    imgArray = np.array(imgArray, dtype=np.uint8)
    print(imgArray)
    pilImg = Image.fromarray(np.uint8(imgArray))
    pilImg.save(path)

def main():
    argvs = sys.argv
    argc = len(argvs)
    print(argc)
    #
    platform_id = 0 # Apple MBP
    device_id = 1 # Intel(R) Iris(TM) Plus Graphics 640
    data_size = DATA_SIZE
    batch_size = BATCH_SIZE
    batch_offset = 0
    #
    print("platform_id=%d" % (platform_id))
    print("device_id=%d" % (device_id))
    print("batch_size=%d" % (batch_size))
    #
    my_gpu = gpu.Gpu(platform_id, device_id)
    my_gpu.set_kernel_code()
    
    mode = 0
    r = core.Roster(mode)
    r.set_gpu(my_gpu)
    wi_path = "./wi.csv"
    
    # 0 : input
    c = r.count_layers()
    input = core.InputLayer(c, data_size, data_size, None, my_gpu, mode)
    r.layers.append(input)
    r.input = input
    # 1 : enc
    c = r.count_layers()
    enc_1 = core.HiddenLayer(c, data_size, 32, input, my_gpu, mode)
    r.layers.append(enc_1)
    # 2 : enc
    c = r.count_layers()
    enc_2 = core.HiddenLayer(c, 32, 16, enc_1, my_gpu, mode)
    r.layers.append(enc_2)
    # 3 : intermediate
    c = r.count_layers()
    inter = core.HiddenLayer(c, 16, 8, enc_2, my_gpu, mode)
    r.layers.append(inter)
    # 4 : dec
    c = r.count_layers()
    dec_1 = core.HiddenLayer(c, 8, 16, inter, my_gpu, mode)
    r.layers.append(dec_1)
    # 5 : dec
    c = r.count_layers()
    dec_2 = core.HiddenLayer(c, 16, 32, dec_1, my_gpu, mode)
    r.layers.append(dec_2)
    # 3 : output
    c = r.count_layers()
    output = core.RegressionOutputLayer(c, 32, data_size, dec_2, my_gpu, mode)
    r.layers.append(output)
    r.output = output
    
    if os.path.isfile(wi_path):
        r.import_weight(wi_path)
    else:
        r.init_weight()
        r.export_weight(wi_path)
    #
    
    if my_gpu:
        r.update_weight()
    #
    batch = util.pickle_load("./data.pickle")
    # prepare memory in roster
    r._batch_size = batch_size
    if r._gpu:
        r._batch_data = np.zeros((batch_size, data_size), dtype=np.float32)
        #r._gpu_input = r._gpu.dev_malloc(r._batch_data)
        r._batch_cross_entropy = np.zeros(batch_size, dtype=np.float32)
        r._gpu_entropy = r._gpu.dev_malloc(r._batch_cross_entropy)
    #
    for layer in r.layers:
        layer.prepare(batch_size)
    #
    
    # set batch data to roster
    data_array = np.zeros((batch_size, data_size), dtype=np.float32)
    for j in range(batch_size):
        #data_array[j] = batch[batch_offset+j]
        data_array[j] = batch[j]
    #
    print(data_array)

    # copy data to GPU
    r.reset()
    layer = r.get_layer_at(0) # input layer
    r._gpu.copy(layer._gpu_output, data_array)
    
    debug = 0
    #r.propagate(debug)
    ce = evaluate(r, data_size, batch_size)
    print(ce)
    #save_img(r, "./test.png")
    #return 0
    
    #c = r.count_layers()
    #output = r.get_layer_at(c-1)
    #r._gpu.copy(output._output_array, output._gpu_output)
    
    # cross entorpy
#    r._gpu.cross_entropy_rg(output._gpu_output, layer._gpu_output, r._gpu_entropy, data_size, batch_size)
#    r._gpu.copy(r._batch_cross_entropy, r._gpu_entropy)
#    ce = np.sum(r._batch_cross_entropy)/np.float32(batch_size)
#    print(ce)
    
    ce = evaluate(r, data_size, batch_size)
    print(ce)
    #return 0

    train_loop(r, data_size, batch_size)
    
    return 0
    # save
#    imgArray = np.reshape(output._output_array, (480, 640))
#    imgArray = imgArray*255
#    imgArray = np.array(imgArray, dtype=np.uint8)
#    print(imgArray)
#    pilImg = Image.fromarray(np.uint8(imgArray))
#    pilImg.save("./test.png")
#
    #layer = r.get_layer_at(0) # input layer
    #            r._gpu.copy(r._output_array, self._gpu_output)
#
    #r.set_batch(pack, batch_size, batch_offset)
    #t = train.Train(pack, r)
    #t.mpi_loop(1)
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
