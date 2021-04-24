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
import package
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
MINI_BATCH_START = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
#
def get_host_id_by_name(name):
    return HOST_NAMES.index(name)

def get_platform_by_host_id(h):
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
        self._platform_id = get_platform_by_host_id(self._host_id)
        self._device_id = get_device_by_host_id(self._host_id)
        #
        self._gpu = gpu.Gpu(self._platform_id, self._device_id)
        self._gpu.set_kernel_code()
        self._package = package.Package(package_id)
        self._roster = self._package.setup_dnn(self._gpu, config_id)
        #
        self._train = train.Train(self._package, self._roster)
        #
        self._batch_size = MINI_BATCH_SIZE[self._rank]
        self._batch_start = MINI_BATCH_START[self._rank]
        self._roster.set_batch(self._package , self._batch_size, self._batch_start)
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

    def evaluate(self):
        self._roster.propagate()
        self._ce = self._roster.get_cross_entropy()
        ce_list = self._com.gather(self._ce, root=0)
        #
        #print(ce_list)
        sum = 0.0
        if self._rank==0:
            for i in ce_list:
                sum = sum + i
            #
            avg = sum/float(self._size)
            self._ce_avg = avg
        #
        if self._rank==0:
            ce_avg_list = [self._ce_avg]*self._size
        else:
            ce_avg_list = None
        #
        ce_avg_list = self._com.scatter(ce_avg_list, root=0)
        self._ce_avg = ce_avg_list
        #print("[%d] ce_avg = %f" % (self._rank, self._ce_avg))
        return self._ce_avg
    
    def evaluate_alt(self, li, ni, ii, wi_alt):
        self._roster.propagate(li, ni, ii, wi_alt, 0)
        ce = self._roster.get_cross_entropy()
        ce_list = self._com.gather(ce, root=0)
        #
        sum = 0.0
        if self._rank==0:
            for i in ce_list:
                sum = sum + i
            #
            avg = sum/float(self._size)
            self._ce_avg = avg
        #
        if self._rank==0:
            ce_avg_list = [self._ce_avg]*self._size
        else:
            ce_avg_list = None
        #
        ce_avg_list = self._com.scatter(ce_avg_list, root=0)
        self._ce_avg = ce_avg_list
        return self._ce_avg
        
    def update_weight(self, li, ni, ii, wi):
        layer = self._roster.get_layer_at(li)
        layer.set_weight_index(ni, ii, wi)
        layer.update_weight()
        
    # this is probably used only rank_0
    def get_weight_index(self, li, ni, ii):
        layer = self._roster.get_layer_at(li)
        wi = layer.get_weight_index(ni, ii)
        return wi
        
    def make_w_list(self):
        self._w_list  = []
        r = self._roster
        c = r.count_layers()
        for li in range(1, c):
            layer = r.get_layer_at(li)
            type = layer.get_type()
            if type==core.LAYER_TYPE_HIDDEN or type==core.LAYER_TYPE_OUTPUT or type==core.LAYER_TYPE_CONV_4:
                for ni in range(layer._num_node):
                    for ii in range(layer._num_input):
                        self._w_list.append((li, ni, ii))
                    #
                #
            #
        #
        return len(self._w_list)
        
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
    w = wk._w_list[attack_i]
    li = w[0]
    ni = w[1]
    ii = w[2]
    r = wk._roster
    layer = r.get_layer_at(li)
    lock = layer.get_weight_lock(ni, ii)   # default : 0
    if lock>0:
        if rank==0:
            print("[%d][%d] locked" % (i, rank))
        #
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
            if rank==0:
                print("[%d][%d] lock_1(%d)" % (i, rank, wi))
            #
            return entropy, 0
        #
    #
    wi_alt = wi + wp_alt
    entropy_alt = wk.evaluate_alt(li, ni, ii, wi_alt)
    #print("[%d][%d] ce=%f, alt=%f (%f)" % (i, rank, entropy, entropy_alt, entropy-entropy_alt))
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
            if rank==0:
                print("[%d][%d] reverse(%d)" % (i, rank, wp_alt))
            #
        else:
            layer.set_weight_property(ni, ii, 0)
            layer.set_weight_lock(ni, ii, 1)
            if rank==0:
                print("[%d][%d] lock_2(%d)" % (i, rank, wi))
            #
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
    package_id = 0  # 0 : MNIST, 1 : Cifar-10
    config_id = 0   # 0 : FC, 1 : CNN
    loop_n = 3*10
    #
    mpi_loop(self, cpm, rank, size)
    return 0
    #
    wk = worker(com, package_id, config_id)
    ce = wk._train.mpi_evaluate(com, rank, size)
    print("rank=%d, ce=%f" % (rank, ce))
#
#
#
    w_num = wk.make_w_list()
    attack_num = int(w_num/10*3)
    if rank==0:
        print("CE : %d : %f" % (rank, ce))
        print("%d : num=%d" % (rank, w_num))
    #
    level = 0
    l_min = 0
    l_max = int(math.log(w_num/100, 2)) + 1
    l_cnts = [1] * l_max
    mode = 1
    #
    cnt = 0
    for j in range(loop_n):
        div = 1.0/float(2**(level))
        cnt = 0
        ret = 0
        for i in range(100):
            #ce, ret = wk._train.multi_attack(ce, 1, div)
            cnt = cnt + ret
            print("[%d] %d : H : %d : %f, %d (%d, %d) %d" % (rank, j, i, ce, level, l_min, l_max, cnt))
        #
        for i in range(100):
            #ce, ret = wk._train.multi_attack(ce, 0, div)
            cnt = cnt + ret
            print("[%d] %d : C : %d : %f, %d (%d, %d) %d" % (rank, j, i, ce, level, l_min, l_max, cnt))
        #
        l_cnts[level] = cnt
        if level == l_max-1:
            mode = -1
        elif level == l_min:
            if cnt==0:
                if l_min==l_max-2:
                    pass
                else:
                    l_min = l_min + 1
                #
            #
            mode = 1
        #
        level = level + mode
        if rank==0:
            package = wk._package
            wk._roster.export_weight(package.save_path())
        else:
            pass
        #
    #
    return 0
#
#
#
    cnt = 0
    for n in range(loop_n):
        # reset
        if n>0 and n%3==0:
            wk._roster.reset_weight_property()
            wk._roster.unlock_weight_all()
            wk._roster.reset_weight_mbid()
        #
        for i in range(attack_num):
            if rank==0:
                k = random.randrange(attack_num)
                ri = [k] * size
            else:
                ri = None
            #
            attack_i = com.scatter(ri, root=0)
            ce, k = weight_shift(i, com, rank, wk, ce, attack_i)
            cnt = cnt + k
            if rank==0:
                print("(%d) [%d][%d] %f (%d) %d" %(n, i, attack_i, ce, k, cnt))
            #
        #
        if rank==0:
            package = wk._package
            wk._roster.export_weight_index(package._wi_csv_path)
        else:
            pass
        #
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
