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
MINI_BATCH_SIZE = [ 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000,
                    2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000 ]
MINI_BATCH_START = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000,
                    20000, 22000, 24000, 26000, 28000, 30000, 32000, 34000, 36000, 38000]
#
def get_host_id_by_name(name):
    return HOST_NAMES.index(name)

def get_platform_by_host_id(h):
    return PACKAGE_IDS[h]
    
def get_device_by_host_id(h):
    return DEVICE_IDS[h]

class worker(object):
    def __init__(self, com, rank, size, package_id, config_id):
        self._com = com
        self._gpu = gpu
        #
        #self._rank = rank
        #self._size = size
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
        #if self._rank==0:
        #    self._w_list = []
        #    self._attack_num = 0
        #    self._attack_cnt = 0
        #
        ce = self._train.mpi_evaluate(1, com, rank, size)
        if rank==0:
            print("[%d] CE starts with %f" % (rank, ce))
        #
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
    config_id = 1   # 0 : FC, 1 : CNN
    #
    wk = worker(com, rank, size, package_id, config_id)
    #print("exit of rank=%d" % (rank))
    #return 0
    #
    wk._train.mpi_loop(1, 1, com, rank, size)
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
