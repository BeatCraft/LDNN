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



def random_hit(delta, temperature):
    A = np.exp(-1.0*delta/(temperature+0.000001))
    if A<1.0:
        hit = random.choices([0, 1], k=1, weights=[A, 1-A])
        if hit[0]==1:
            return 1
        #
    else:
        return 1
    #
    return 0

def main():
    argvs = sys.argv
    argc = len(argvs)
    print(argc)
    #

    delta = 0.1
    for i in range(10):
        h = random_hit(delta, float(i))
        if h>0:
            print(h)
        #
    #
    return 0

    
    for i in range(10):
        A = np.exp(-1*delta/(float(i)+0.000001))
        print(i, A, 1-A)
        #if A<1.0:
        #    hit = random.choices([0, 1], k=1, weights=[A, 1-A])
        #    if hit[0]==1:
        #        print(hit)
        #    #
        #
        #r = random.random()
        
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
