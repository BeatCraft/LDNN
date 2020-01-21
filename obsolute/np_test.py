import os, sys, time, math
from stat import *
import random
import copy
import multiprocessing as mp
import pickle
import numpy as np
#
#
#
def main():

    a =[1,2]
    b = [[3,4],[5,6]]
    aa = np.array(a)
    ba = np.array(b)
    print np.dot(aa, ba)
    print np.dot(ba, aa)
    
    
    print aa * ba[0]
    print aa * ba[1]
    
    print aa * 2
    
    print ba[1,1]
    print ba[1][1]
    
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
