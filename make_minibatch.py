#! /usr/bin/python
# -*- coding: utf-8 -*-
#

#
# LDNN : lesser Deep Neural Network
#
PACKAGE_BASE_PATH = "../ldnn_package/"
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
import csv
from PIL import Image
from PIL import ImageFile
from PIL import JpegImagePlugin
from PIL import ImageFile
from PIL import PngImagePlugin
import zlib
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
def make_minibatch(package, mini_batch_size, num):
    batch_size = package._train_batch_size
    data_size = package._image_size
    #
    data_array = np.zeros((mini_batch_size, data_size), dtype=np.float32)
    class_array = np.zeros(mini_batch_size, dtype=np.int32)
    #
    
    for j in range(num):
        random_index = []
        for i in range(mini_batch_size):
            bi = random.randrange(batch_size)
            random_index.append(bi)
            #data_array[i] = package._train_image_batch[bi]
            #class_array[i] = package._train_label_batch[bi]
        #
        #data = (data_array, class_array)
        save_path = "../ldnn_config/%s/mini/%d/%03d.pickle" % (package._name, mini_batch_size, j)
        util.pickle_save(save_path, random_index)
        print(save_path)
    #
#
#
#
def main():
    argvs = sys.argv
    argc = len(argvs)
    #
    # 0 : MNIST
    # 1 : MNIST2 (smaller network)
    # 2 : CIFAR-10
    package_id = 0
    package = util.Package(package_id)
    package.load_batch()
    mini_batch_size = 20000
    num = 1000
    #
    if package_id==0:
        make_minibatch(package, mini_batch_size, num)
    elif package_id==1:
        pass
    elif package_id==2:
        pass
    else:
        pass
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
