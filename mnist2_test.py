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
import cPickle
import csv
#
#
# LDNN Modules
import core
import util
import gpu
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
    cat = 0
    clut = 0
    data_size = 28*28
    batch_size = 60000
    batch_data = np.zeros((batch_size, data_size), dtype=np.float32)
    batch_label = np.zeros(batch_size, dtype=np.int32)
    
    pic_index = 0
    for cat in range(10):
        for clut in range(10):
            path = "./package/MNIST2/clustered_data/class_%d/cluster_%d/image.csv" % (cat, clut)
            print path
            #
            with open(path, "r") as f:
                reader = csv.reader(f)
                n = 0
                for row in reader:
                    #print n
                    #print len(row)
                    i = 0
                    #for cell in row:
                    for i in range(data_size):
                        cell = row[i]
                        tmp = int(cell)
                        batch_data[pic_index][i] = float(tmp)
                        i = i + 1
                    #
                    #print batch_data[pic_index]
                    batch_label[pic_index] = cat
                    n = n + 1
                    pic_index = pic_index +1
                # for
                print n
            # with
        #
    #
    print pic_index
    print batch_label
    

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
