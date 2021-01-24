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
MINI_BATCH_SIZE = [0, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000]
MINI_BATCH_START = [0, 0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000]
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
        #
        self._host_id = get_host_id_by_name(self._processor_name)
        if self._rank==0: # server
            self._platform_id = -1
            self._device_id = -1
            self._gpu = None
        else:
            self._platform_id = get_package_by_host_id(self._host_id)
            self._device_id = get_device_by_host_id(self._host_id)
            self._gpu = gpu.Gpu(self._platform_id, self._device_id)
            #
            self._gpu.set_kernel_code()
        #
        self._package = util.Package(package_id)
        self._roster = self._package.setup_dnn(self._gpu, config_id)
        self._package.load_batch()
        self._data_size = self._package._image_size
        self._num_class = self._package._num_class
        #
        if self._rank==0: # server
            self._w_list = []
            #self._attack_num = 0
        else:
            self._batch_size = MINI_BATCH_SIZE[self._rank]
            self._batch_start = MINI_BATCH_START[self._rank]
            #
            self._data_array = np.zeros((self._batch_size , self._data_size), dtype=np.float32)
            self._class_array = np.zeros(self._batch_size , dtype=np.int32)
            self._roster.prepare(self._batch_size, self._data_size, self._num_class)
        #
        self._attack_num = 0

    def debug(self):
        print("processor_name=%s" %(self._processor_name))
        print("host_id=%d" % (self._host_id))
        print("rank=%d" % (self._rank ))
        print("size=%d" % (self._size ))

    def set_batch(self):
        if self._rank==0:
            pass
        else:
            for i in range(self._batch_size):
                self._data_array[i] = self._package._train_image_batch[self._batch_start + i]
                self._class_array[i] = self._package._train_label_batch[self._batch_start + i]
            #
            self._roster.set_data(self._data_array, self._data_size, self._class_array, self._batch_size)
        #

    def evaluate(self):
        if self._rank==0:
            ce = 0.0
        else:
            self._roster.propagate()
            ce = self._roster.get_cross_entropy()
        #
        ce_list = self._com.gather(ce, root=0)
        #
        if self._rank==0:
            sum = 0.0
            for i in ce_list:
                sum = sum + i
            #
            entropy = sum/float(self._size-1)
            print("entropy=%f" % (entropy))
            self._ce = entropy
        #
    
    def evaluate_alt(self, li, ni, ii, wi_alt):
        if self._rank==0:
            ce = 0
        else:
            self._roster.propagate(li, ni, ii, wi_alt, 0)
            ce = self._roster.get_cross_entropy()
        #
        ce_list = self._com.gather(ce, root=0)
        if self._rank==0:
            sum = 0.0
            for i in ce_list:
                sum = sum + i
            #
            entropy = sum/float(self._size-1)
            print("entropy=%f" % (entropy))
            self._ce_alt = entropy
        #
        
    def update_weight(self, li, ni, ii, wi):
        if self._rank==0:
            pass
        else:
            layer = self._roster.getLayerAt(li)
            layer.set_weight_index(ni, ii, wi)
            layer.update_weight()
        #
        
    # this is probably used only rank_0
    def get_weight_index(self, li, ni, ii):
        layer = self._roster.getLayerAt(li)
        wi = layer.get_weight_index(ni, ii)
        return wi
        
    def init_weight_list(self):
        if self._rank==0:
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
            self._attack_num = int(len(self._w_list)/1000)
            random.shuffle(self._w_list)
            random.shuffle(self._w_list)
        else:
            self._attack_num = 0
        #
        attack_num = self._com.bcast(self._attack_num, root=0)
        return attack_num

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
    print("s : %d, %d, %d, %d" % (rank, size, my_rank, my_size))
    #
    package_id = 0
    config_id = 0
    #
    #
    #
    wk = worker(com, package_id, config_id)
    wk.set_batch()
    num = wk.init_weight_list()
    print("%d : num=%d" % (rank, num))
    
    
#    wk.evaluate()
#    wi  = wk.get_weight_index(1, 2, 3)
#    print("%d : %d" % (rank, wi))
    #
#    wk.evaluate_alt(1, 2, 3, 5)
#    wk.update_weight(1, 2, 3, 5)
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
