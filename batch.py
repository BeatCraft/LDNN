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
    sts = 0
    batch_size = 100
    data_size = 28*28
    it = 20*10
    # GPU
    # 0 : AMD Server
    # 1 : Intel on MBP
    # 2 : eGPU (AMD Radeon Pro 580)
    platform_id = 0
    device_id = 2
    my_gpu = gpu.Gpu(platform_id, device_id)
    my_gpu.set_kernel_code()
    #
    r = main.setup_dnn(main.NETWORK_PATH, my_gpu)
    if r is None:
        print "fatal DNN error"
        sys.exit(sts)

    start_time = time.time()
    #
    batch = util.pickle_load(main.TRAIN_BATCH_PATH)
    if batch is None:
        print "error : no train batch"
        sys.exit(sts)
    #
    main.loop(it, r, batch, batch_size, data_size)
    #
    elasped_time = time.time() - start_time
    t = format(elasped_time, "0")
    print "[total elasped time] %s" % (t)
    #
    print ">> end"
    print("\007")
    sys.exit(sts)
#
#
#
