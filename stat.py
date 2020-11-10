#! /usr/bin/python
# -*- coding: utf-8 -*-
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
import pickle
import pyopencl as cl
import glob
import re

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
def main():
    argvs = sys.argv
    argc = len(argvs)
    #
    platform_id = 0
    device_id = 1
    package_id = 0
    #
    my_gpu = gpu.Gpu(platform_id, device_id)
    my_gpu.set_kernel_code()
    #
    package = util.Package(package_id)
    r = package.setup_dnn(my_gpu)
    #
    
    n = 18
    for i in range(n):
        search_path = "../test-data/64/7-2000-64x4/%04d-*.csv" % (i)
        l = glob.glob(search_path)
        if len(l)==1:
            filepath = l[0]
            filename = os.path.basename(filepath)
            root, ext = os.path.splitext(filename)
            tl = root.split("-")
            #if my_gpu:
            #    r.update_weight()
            #
            
            acc = 0.0
            #print filepath
            #r.import_weight_index(filepath)
            acc = test.stat(r, package, filepath, 0)
            print("%d, %s, %f" % (i, tl[1], acc))
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
