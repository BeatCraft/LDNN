#! /usr/bin/python
# -*- coding: utf-8 -*-
#

#
# LDNN : lesser's Deep Neural Network
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
#
#
# LDNN Modules
import util
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
    #
    #
    wl = util.pickle_load("../ldnn_config/MNIST/wf.pickle")
    print len(wl)
    print wl[0].shape
    print wl[1].shape
    print wl[2].shape
    print wl[0][0]
    #
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
