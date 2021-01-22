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
MINI_BATCH_SIZE = [1000, 250, 250, 250]

def get_host_index_by_name(name):
    return HOST_NAMES.index(name)

class worker(object):
    def __init__(self, com, platform_id, device_id, package_id, config_id):
        self._com = com
        self._gpu = gpu.Gpu(platform_id, device_id)
        self._gpu.set_kernel_code()
        #
        self._package = util.Package(package_id)
        self._roster = self._package.setup_dnn(self._gpu, config_id)
        self._package.load_batch()
        self._data_size = self._package._image_size
        #
        self._processor_name = MPI.Get_processor_name()
        self._host_index = get_host_index_by_name(self._processor_name)
        self._rank = self._com.Get_rank()
        self._size = self._com.Get_size()
        
    def debug(self):
        print("processor_name=%s" %(self._processor_name))
        print("host_id=%d" % (self._host_index ))
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
    def __init__(self, com, platform_id, device_id, package_id, config_id, size, start):
        super(client, self).__init__(com, platform_id, device_id, package_id, config_id)
        #
        self._batch_size = size
        self._batch_start = start
        #
        self._data_array = np.zeros((self._batch_size , self._data_size), dtype=np.float32)
        self._class_array = np.zeros(self._batch_size , dtype=np.int32)
        self._roster.prepare(self._batch_size, self._data_size)


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
    def __init__(self, com, platform_id, device_id, package_id, config_id):
        super(server, self).__init__(com, platform_id, device_id, package_id, config_id)

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
    print(f"comm : rank={rank}, size={size}")
    my_self = MPI.COMM_SELF
    my_size = my_self.group.Get_size()
    my_rank = my_self.group.Get_rank()
    print(f"self : rank={my_rank}, size={my_size}")

    #pname = MPI.Get_processor_name()
    #print(f"Hello, world! from {pname}")
    #h_id = get_host_index_by_name(pname)
    #print(f"index={i}")
    #
    #
    #
    package_id = 0
    config_id = 0
    platform_id = 0
    device_id = 0
    #
    cmd = 0
    ent = np.arange(10, dtype=np.float32)
    #
    if rank == 0: # server
        platform_id = 0
        device_id = 0
        #
        s = server(comm, platform_id, device_id, package_id, config_id)
        s.debug()
        #cmd = comm.bcast(cmd, root=0)
    else:
        platform_id = 1
        device_id = 1
        #
        size = 100
        start = 100
        c = client(comm, platform_id, device_id, package_id, config_id, size, start)
        c.debug()
    #
    return 0
#
#
#
    if rank == 0: # server
        #print(f"server({rank})={pname}")
        print("server : %s, %d, %d" % (pname, rank, i))
        comm.send((10, "test"), dest=1, tag=1)
        if i==0:
            pass
            #comm.send((rank, size, pname, my_rank, my_size), dest=0, tag=1)
        else:
            pass
        #
    else: # client
        print("client : %s, %d, %d" % (pname, rank, i))
        k, t = comm.recv(source=0, tag=1)
        print("%d : %s" % (k, t))
        if i==0:
            pass
        else:
            pass
            #world_rank, world_size, your_name, your_rank, your_size = comm.recv(source=i, tag=1)
            #print("%d, %d, %d, %d,%d" % (world_rank, world_size, your_name, your_rank, your_size))
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
