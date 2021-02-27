import os, sys, time, math
from stat import *
import random
import copy
import math
import numpy as np
import struct
import pickle
#
import numpy as np
#
import util
#
#
#
p = 10
r = 28
b = 8/3
dt = 0.01
t_0 = 0
t_1 = 50
X_0 = np.array([1, 1, 1])

def RungeKutta(t, X):
  k_1 = Lorenz(t, X)
  k_2 = Lorenz(t + dt/2, X + k_1*dt/2)
  k_3 = Lorenz(t + dt/2, X + k_2*dt/2)
  k_4 = Lorenz(t + dt, X + k_3*dt)
  X_next = X + dt/6*(k_1 + 2*k_2 + 2*k_3 + k_4)
  return X_next

def Lorenz(t, X):
  x = X[0]
  y = X[1]
  z = X[2]
  return np.array([-p*x + p*y, -x*z + r*x - y, x*y - b*z])

def main():
    argvs = sys.argv
    argc = len(argvs)
    print(argc)
    #
    t = t_0
    X = X_0
    data = np.r_[X]
    ims = []
    #
    while t < t_1:
        X = RungeKutta(t, X)
        data = np.c_[data, X]
        t += dt
    #
    print data.shape
    data= data.T
    print data.shape
    size = data.shape[0]
    #
    max = np.max(abs(data))
    print("max=%f" % (max))
    data = data/max + 1.0
    #
    data_in = []
    data_out = []
    k = 0
    for i in range(size):
        if i%2==0:
            data_in.append([data[i][0], data[i][1], data[i][2]])
        else:
            data_out.append([data[i][0], data[i][1], data[i][2]])
            k = k + 1
        #
    #
    print data[0]
    print data_in[0]
    print size
    print len(data_in)
    
    i_list = [i for i in range(size/2)]
    random.shuffle(i_list)
    
    train_i_list = i_list[:2001]
    test_i_list = i_list[2001:]
    
    train_in = []
    train_out = []
    for i in train_i_list:
        train_in.append(data_in[i])
        train_out.append(data_out[i])
    #
    test_in = []
    test_out = []
    for i in test_i_list:
        test_in.append(data_in[i])
        test_out.append(data_out[i])
    #
    util.pickle_save("./lorenz/train_in.pickle", np.array(train_in))
    util.pickle_save("./lorenz/train_out.pickle", np.array(train_out))
    util.pickle_save("./lorenz/test_in.pickle", np.array(test_in))
    util.pickle_save("./lorenz/test_out.pickle", np.array(test_out))
    #
    return 0
    
if __name__=='__main__':
    print(">> start")
    sts = main()
    print(">> end")
    print("\007")
    sys.exit(sts)
#
#
#
