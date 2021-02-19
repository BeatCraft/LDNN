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
def main():
    argvs = sys.argv
    argc = len(argvs)
    print(argc)
    #
    if argc==7:
        pass
    else:
        print("error in sh")
        return 0
    #
    platform_id = int(argvs[1])
    device_id = int(argvs[2])
    package_id = int(argvs[3])
    config = int(argvs[4])
    mode = int(argvs[5])
    size = int(argvs[6])
    print("platform_id=%d" % (platform_id))
    print("device_id=%d" % (device_id))
    print("package_id=%d" % (package_id))
    print("config=%d" % (config))
    print("mode=%d" % (mode))
    print("mini_batch_size=%d" % (size))
    #
    my_gpu = gpu.Gpu(platform_id, device_id)
    my_gpu.set_kernel_code()
    pack = package.Package(package_id)
    r = pack.setup_dnn(my_gpu, config)
    #
    if mode==0: # train
        mini_batch_size = size
        print("package._train_batch_size=%d" % (pack._train_batch_size))
        t = train.Train(pack, r)
        t.set_mini_batch_size(mini_batch_size)
        t.loop()
    elif mode==1: # test
        test.test_n(r, pack, 500)
    elif mode==2: #
        test.unit_test(r, pack)
    elif mode==3: #
        test.cnn_test(r, pack)
    elif mode==4: #
        mini_batch_size = size
        print("package._train_batch_size=%d" % (pack._train_batch_size))
        t = train.Train(pack, r)
        t.set_mini_batch_size(mini_batch_size)
        t.loop_hb()
    elif mode==5: #
        mini_batch_size = size
        print("package._train_batch_size=%d" % (pack._train_batch_size))
        t = train.Train(pack, r)
        t.set_mini_batch_size(mini_batch_size)
        t.loop_hb2()
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
#    logger.debug('hello')
    sts = main()
    print(">> end")
    print("\007")
    sys.exit(sts)
#
#
#
