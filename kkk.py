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
import cPickle
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
    width = 32
    height = 32
    num_input = width * height # 1024
    stride = 3
    kx = width - (stride-1)
    ky = height - (stride-1)
    
    for y in range(ky):
        for x in range(kx):
            #
            c0 = x + 0 + y*width
            c1 = x + 1 + y*width
            c2 = x + 2 + y*width
            #
            c3 = x + 0 + (y+1)*width
            c4 = x + 1 + (y+1)*width
            c5 = x + 2 + (y+1)*width
            #
            c6 = x + 0 + (y+2)*width
            c7 = x + 1 + (y+2)*width
            c8 = x + 2 + (y+2)*width
            #
            print "(%d, %d, %d), (%d, %d, %d), (%d, %d, %d)" % (c0, c1, c2, c3, c4, c5, c6, c7, c8)
        #
    #


    #
    #
    return 0
#
#
#
if __name__=='__main__':
    print ">> start"
    sts = main()
    print ">> end"
    print("\007")
    sys.exit(sts)
#
#
#
