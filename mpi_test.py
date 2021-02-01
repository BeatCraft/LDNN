#! /usr/bin/python
# -*- coding: utf-8 -*-
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
import glob
import re
#
from mpi4py import MPI
#
# LDNN Modules
#
import core
import util
import gpu
import train
import test
#
#
#
sys.setrecursionlimit(10000)
#
#
#
HOST_NAMES = ["threadripper", "amd-1", "amd-2", "amd-3"]
PACKAGE_IDS = [1, 0, 0, 0]
DEVICE_IDS =  [0, 0, 0, 0]
MINI_BATCH_SIZE = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
MINI_BATCH_START = [0, 1000, 2000, 3000, 4000, 50000, 6000, 7000, 8000, 9000, 10000]
#
def get_host_id_by_name(name):
    return HOST_NAMES.index(name)

def get_package_by_host_id(h):
    return PACKAGE_IDS[h]
    
def get_device_by_host_id(h):
    return DEVICE_IDS[h]

class worker(object):
    def __init__(self, com, package_id, config_id):
        self._com = com
        self._gpu = gpu
        #
        self._processor_name = MPI.Get_processor_name()
        self._rank = self._com.Get_rank()
        self._size = self._com.Get_size()
        self._host_id = get_host_id_by_name(self._processor_name)
        self._platform_id = get_package_by_host_id(self._host_id)
        self._device_id = get_device_by_host_id(self._host_id)
        #
        self._gpu = gpu.Gpu(self._platform_id, self._device_id)
        self._gpu.set_kernel_code()
        #
        self._package = util.Package(package_id)
        self._roster = self._package.setup_dnn(self._gpu, config_id)
        self._package.load_batch()
        self._data_size = self._package._image_size
        self._num_class = self._package._num_class
        #
        self._batch_size = MINI_BATCH_SIZE[self._rank]
        self._batch_start = MINI_BATCH_START[self._rank]
        #
        self._data_array = np.zeros((self._batch_size , self._data_size), dtype=np.float32)
        self._class_array = np.zeros(self._batch_size , dtype=np.int32)
        self._roster.prepare(self._batch_size, self._data_size, self._num_class)
        #
        if self._rank==0:
            self._w_list = []
            self._attack_num = 0
            self._attack_cnt = 0
        #

    def debug(self):
        print("processor_name=%s" %(self._processor_name))
        print("host_id=%d" % (self._host_id))
        print("rank=%d" % (self._rank ))
        print("size=%d" % (self._size ))

    def set_batch(self):
        print("[%d] bsize = %d, start = %d" % (self._rank, self._batch_size, self._batch_start))
        for i in range(self._batch_size):
            self._data_array[i] = self._package._train_image_batch[self._batch_start + i]
            self._class_array[i] = self._package._train_label_batch[self._batch_start + i]
        #
        self._roster.set_data(self._data_array, self._data_size, self._class_array, self._batch_size)

    def evaluate(self):
        self._roster.propagate()
        self._ce = self._roster.get_cross_entropy()
        ce_list = self._com.gather(self._ce)#root=0
        ce_list = self.com.scatter(ce_list, root=0)
        #
        print(ce_list)
        sum = 0.0
        for i in ce_list:
            sum = sum + i
        #
        avg = sum/float(self._size)
        self._ce_avg = avg
        #
        #self._ce_avg = self._com.bcast(self._ce_avg, root=0)
        return self._ce_avg
    
    def evaluate_alt(self, li, ni, ii, wi_alt):
        self._roster.propagate(li, ni, ii, wi_alt, 0)
        print("    %d : propagate" % (self._rank))
        ce = self._roster.get_cross_entropy()
        print("    %d : ce : %f" % (self._rank, ce))
        ce_list = self._com.gather(ce, root=0)
        print("    %d : gather" % (self._rank))
        #
        sum = 0.0
        for i in ce_list:
            if self._rank==0:
                print("        %f" % (i))
            #
            sum = sum + i
        #
        avg = sum/float(self._size)
        self._ce_avg = avg
        #
        return self._ce_avg
        
    def update_weight(self, li, ni, ii, wi):
        layer = self._roster.getLayerAt(li)
        layer.set_weight_index(ni, ii, wi)
        layer.update_weight()
        
    # this is probably used only rank_0
    def get_weight_index(self, li, ni, ii):
        layer = self._roster.getLayerAt(li)
        wi = layer.get_weight_index(ni, ii)
        return wi
        
    def make_w_list(self):
        self._w_list  = []
        r = self._roster
        c = r.countLayers()
        for li in range(1, c):
            layer = r.getLayerAt(li)
            for ni in range(layer._num_node):
                for ii in range(layer._num_input):
                    self._w_list.append((li, ni, ii))
                #
            #
        
        #
        #self._attack_num = int(len(self._w_list)/1000)
        return len(self._w_list)
        #random.shuffle(self._w_list)
        #random.shuffle(self._w_list)
        #else:
        #    self._attack_num = 0
        #
#        attack_num = self._com.bcast(self._attack_num, root=0)
#        return attack_num
    
    def get_weight_pack(self, i):
        tp = self._w_list[i]
        li = tp[0]
        ni = tp[1]
        ii = tp[2]
        layer = self._roster.getLayerAt(li)
        wi = layer.get_weight_index(ni, ii)
        return (li, ni, ii, wi)
        #tp = self._com.bcast((li, ni, ii, wi), root=0)
        #return tp
    
#    def weight_shift(self, li, ni, ii, wi):
#        layer = self._roster.getLayerAt(li)
#        wi_alt = wi
#        entropy = self._ce
#        entropy_alt = entropy
#        maximum = core.WEIGHT_INDEX_MAX
#        minimum = core.WEIGHT_INDEX_MIN
#
#
#
def bcast_random_int(i, com, rank, size, max):
    print("[%d | %d] bcast_random_int()" % (i, rank))
    if rank==0:
        k = random.randrange(max)
        ri = [k] * size
    else:
        ri = 0
    #
    #ri = com.bcast(ri, root=0)
    ri = com.scatter(ri, root=0)
    print("[%d | %d]    =%d" % (i, rank, ri))
    return ri

def average_float(com, rank, v):
    v_list = com.gather(v, root=0)
    if rank==0:
        v_sum = 0
        for n in v_list:
            v_sum = v_sum + n
        #
        avg = v_sum/float(len(v_list))
    else:
        avg = 0
    #
    return avg
    
def weight_shift(i, com, rank, wk, entropy, attack_i):
    print("[%d | %d] weight_shift : %d" % (i, rank, attack_i))
#    if rank==0:
#        print("weight_shift : %d" % attack_i)
#    #
    w = wk._w_list[attack_i]
    li = w[0]
    ni = w[1]
    ii = w[2]
    r = wk._roster
    layer = r.getLayerAt(li)
    lock = layer.get_weight_lock(ni, ii)   # default : 0
    if lock>0:
#        if rank==0:
#            print("locked")
#        #
        print("[%d][%d] locked(%d)" % (i, rank, wi))
        return entropy, 0
    #
    wp = layer.get_weight_property(ni, ii) # default : 0
    wi = layer.get_weight_index(ni, ii)
    wi_alt = wi
    entropy_alt = entropy
    maximum = core.WEIGHT_INDEX_MAX
    minimum = core.WEIGHT_INDEX_MIN
    #
    wp_alt = wp
    if wp_alt==0:
        if wi==maximum:
            wp_alt = -1
        else:
            wp_alt = 1
        #
    else:
        if wi==maximum or wi==minimum:
            layer.set_weight_property(ni, ii, 0)
            layer.set_weight_lock(ni, ii, 1)
#            if rank==0:
#                print("lock (%d)" % wi)
#            #
            print("[%d][%d] lock_1(%d)" % (i, rank, wi))
            return entropy, 0
        #
    #
    wi_alt = wi + wp_alt
    entropy_alt = wk.evaluate_alt(li, ni, ii, wi_alt)
    print("[%d][%d] ce=%f, alt=%f (%f)" % (i, rank, entropy, entropy_alt, entropy-entropy_alt))
#    if rank==0:
#        print("ce_alt = %f | %f" %(entropy_alt, entropy))
#    #
    if entropy_alt<entropy:
        layer.set_weight_property(ni, ii, wp_alt)
        layer.set_weight_index(ni, ii, wi_alt)
        layer.update_weight()
        return entropy_alt, 1
    else:
        if wp==0:
            # reverse
            wp_alt = wp_alt*(-1)
            layer.set_weight_property(ni, ii, wp_alt)
            print("[%d][%d] reverse(%d)" % (i, rank, wp_alt))
        else:
            layer.set_weight_property(ni, ii, 0)
            layer.set_weight_lock(ni, ii, 1)
            print("[%d][%d] lock_2(%d)" % (i, rank, wi))
        #
    #
    return entropy, 0
#
#
#
def main():
    argvs = sys.argv
    argc = len(argvs)
    #
    com = MPI.COMM_WORLD
    rank = com.Get_rank()
    size = com.Get_size()
    my_self = MPI.COMM_SELF
    my_rank = my_self.group.Get_rank()
    my_size = my_self.group.Get_size()
    print("%d, %d, %d, %d" % (rank, size, my_rank, my_size))
    #
    package_id = 0  # MNIST
    config_id = 0   # FC
    #
#    if rank == 0:
#        data = [2, 2]
#        print(data)
#    else:
#        data = None
#    #
#    data = com.scatter(data, root=0)
#    print(data)
#    #
#    return 0
    #
    #
    #
    wk = worker(com, package_id, config_id)
    wk.set_batch()
    ce = wk.evaluate()
    w_num = wk.make_w_list()
    attack_num = int(w_num / 10)
    if rank==0:
        print("CE : %d : %f" % (rank, ce))
        print("%d : num=%d" % (rank, w_num))
    #
    #
    cnt = 0
    for i in range(attack_num):
        if rank==0:
            print("%d" % i)
        #
        #attack_i = bcast_random_int(i, com, rank, size, attack_num)
        if rank==0:
            k = random.randrange(attack_num)
            ri = [k] * size
        else:
            ri = None
        #
        attack_i = com.scatter(ri, root=0)
        print("[%d | %d] scatter(%d)" % (i, rank, attack_i))
    
        #
        ce, k = weight_shift(i, com, rank, wk, ce, attack_i)
        cnt = cnt + k
        if rank==0:
            print("[%d][%d] %f (%d) %d" %(i, attack_i, ce, k, cnt))
        #
    #
    if rank==0:
        package = wk._package
        wk._roster.export_weight_index(package._wi_csv_path)
    else:
        pass
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
