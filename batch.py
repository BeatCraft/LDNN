#! /usr/bin/python
# -*- coding: utf-8 -*-
#

#
# LDNN : lesser Deep Neural Network
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
#
#
# LDNN Modules
import core
import util
import gpu
import main
#
#
#
sys.setrecursionlimit(10000)
#
#
#
if __name__=='__main__':
    print ">> start"
    #
    debug = 1
    it = 20*20
    package = -1
    mode = -1
    package = 0
    batch_size = 10000
    #
    # GPU
    #
    platform_id = 0
    # 0 : AMD Server
    # 1 : Intel on MBP
    # 2 : eGPU (AMD Radeon Pro 580)
    device_id = 1
    #
    my_gpu = gpu.Gpu(platform_id, device_id)
    my_gpu.set_kernel_code()
    #
    if package==0: # MNIST
        mnist = util.Mnist(mode)
        r = mnist.setup_dnn(my_gpu)
        r.set_batch(mnist._data, mnist._labels, batch_size, util.MNIST_IMAGE_SIZE, util.MNIST_NUM_CLASS)
        loop(it, r, mnist, debug)
    elif package==1: # CIFAR-10
        cifar10 = util.Cifar10(mode)
        r = cifar10.setup_dnn(my_gpu)
        r.set_batch(cifar10._data, cifar10._labels, batch_size, util.CIFAR10_IMAGE_Y_SIZE, util.CIFAR10_NUM_CLASS)
        loop(it, r, cifar10, debug)
    #
    print ">> end"
    print("\007")
    #
    sts = 0
    sys.exit(sts)
#
#
#
