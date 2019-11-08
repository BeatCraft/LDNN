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

    n = 10
    c = 20
    r = []
    for i in range(c):
        r.append(0)
    
    rn_list = []
    for i in range(n):
        rn_list.append( random.randrange(c) )
    
    for w in rn_list:
        r[w] = r[w] + 1


    print r

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
