#! /usr/bin/python
# -*- coding: utf-8 -*-
#

#
# LDNN : lesser Deep Neural Network
#

import os, sys, time, math
import numpy as np
from PIL import Image


# LDNN Modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import util


#
sys.setrecursionlimit(10000)
#

DATA_SIZE = 32#128
#BATCH_SIZE = 2400
#
#
#
def main():
    argvs = sys.argv
    argc = len(argvs)
    #
    path = "./img/jakar_lp-2.png"
    img = Image.open(path)
    img = img.convert("L")
    
    imgArray = np.asarray(img)
    print(imgArray.shape)
    
    n = imgArray.shape[0] * imgArray.shape[1] / DATA_SIZE
    imgArray = np.reshape(imgArray, (n, DATA_SIZE))
    print(imgArray.shape)
    
    imgArray = np.array(imgArray, dtype=np.float32)
    print(imgArray)
    
    imgArray = imgArray/255
    print(imgArray)
    
    util.pickle_save("./data.pickle", imgArray)
    return 0

    
    #pix = img.load()
    #print(pix)
    #print(imgArray[0])
    #pilImg = Image.fromarray(numpy.uint8(imgArray))
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
