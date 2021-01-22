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
MINI_BATCH_SIZE = [1000, 250, 250, 250]
MINI_BATCH_START = [0, 1000, 1250, 1500, 1750]
#
def get_host_id_by_name(name):
    return HOST_NAMES.index(name)

def get_package_by_host_id(host_id):
    return PACKAGE_IDS[host_id]
    
def get_device_by_host_id(host_id):
    return DEVICE_IDS[host_id]

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
            self._gpu.set_kernel_code()
        #
        self._package = util.Package(package_id)
        self._roster = self._package.setup_dnn(self._gpu, config_id)
        self._package.load_batch()
        self._data_size = self._package._image_size
        self._num_class = self._package._num_class

    def debug(self):
        print("processor_name=%s" %(self._processor_name))
        print("host_id=%d" % (self._host_id))
        print("rank=%d" % (self._rank ))
        print("size=%d" % (self._size ))
        
    def set_batch(self, start):
        pass
        
    def evaluate(self):
        pass

    def alt(self):
        pass

    def update(self):
        pass

class client(worker):
    def __init__(self, com, package_id, config_id):
        super(client, self).__init__(com, package_id, config_id)
        #
        self._batch_size = MINI_BATCH_SIZE[self._host_id]
        self._batch_start = MINI_BATCH_START[self._host_id]
        #
        self._data_array = np.zeros((self._batch_size , self._data_size), dtype=np.float32)
        self._class_array = np.zeros(self._batch_size , dtype=np.int32)
        self._roster.prepare(self._batch_size, self._data_size, self._num_class)


    def set_batch(self):
        for i in range(self._batch_size):
            self._data_array[i] = self._package._train_image_batch[self._batch_start + i]
            self._class_array[i] = self._package._train_label_batch[self._batch_start + i]
        #
        self._roster.set_data(self._data_array, self._data_size, self._class_array, self._batch_size)

    def evaluate(self):
        self._roster.propagate()
        ce = self._roster.get_cross_entropy()
        return ce
        
    def alt(self):
        pass

    def update(self):
        pass

class server(worker):
    def __init__(self, com, package_id, config_id):
        super(server, self).__init__(com, package_id, config_id)

#   def init(self, size):
#        pass
#        comm.send((10, "test"), dest=1, tag=1)
        
    def set_batch(self, start):
        pass
        
    def evaluate(self):
        pass

    def alt(self):
        pass

    def update(self):
        pass

    def train(self):
        pass

def main():
    argvs = sys.argv
    argc = len(argvs)
    #
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    my_self = MPI.COMM_SELF
    my_rank = my_self.group.Get_rank()
    my_size = my_self.group.Get_size()
    print(f"info : %d, %d, %d, %d" % (rank, size, my_rank, my_size))
    #
    package_id = 0
    config_id = 0
    #
    cmd = 0
    ent = np.arange(10, dtype=np.float32)
    #
    if rank == 0: # server
        platform_id = 0
        device_id = 0
        #
        #gpu = None
        s = server(comm, package_id, config_id)
        s.debug()
        cmd = comm.bcast(cmd, root=0)
    else:
        c = client(comm, package_id, config_id)
        c.debug()
        cmd = cmd +1
    #
    print(cmd)
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
