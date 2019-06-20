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
# constant values
#
TRAIN_BASE_PATH  = "./data/train/"
TRAIN_BATCH_PATH = "./train_batch.pickle"
TEST_BASE_PATH   = "./data/test/"
TEST_BATCH_PATH = "./test_batch.pickle"
NETWORK_PATH     = "./network.pickle"
PROCEEDED_PATH   = "./proceeded.pickle"

TRAIN_IMAGE_PATH  = "./MNIST/train-images-idx3-ubyte"
TRAIN_LABEL_PATH = "./MNIST/train-labels-idx1-ubyte"
TEST_IMAGE_PATH   = "./MNIST/t10k-images-idx3-ubyte"
TEST_LABEL_PATH   = "./MNIST/t10k-labels-idx1-ubyte"

MNIST_IMAGE_WIDTH  = 28
MNIST_IMAGE_HEIGHT = 28
MNIST_IMAGE_SIZE   = MNIST_IMAGE_WIDTH*MNIST_IMAGE_HEIGHT

TRAIN_BATCH_SIZE  = 60000
TEST_BATCH_SIZE   = 10000
IMAGE_HEADER_SIZE = 16
LABEL_HEADER_SIZE  = 8


WEIGHT_INDEX_CSV_PATH   = "./wi.csv"

NUM_OF_CLASS     = 10    # 0,1,2,3,4,5,6,7,8,9
NUM_OF_SAMPLES   = 5000  # must be 5,000 per a class
NUM_OF_TEST      = 500
#
#
#
def check_weight_distribution():
    w_list = util.csv_to_list(WEIGHT_INDEX_CSV_PATH)
    w_total = len(w_list)
    print w_total
    if w_total<=0:
        print "error"
        return 0

    v_list = []
    total = 0.0
    for i in range(core.WEIGHT_INDEX_SIZE):
        key = str(i)
        num =  w_list.count(key)
        v = float(num)/w_total*100.0
        print "[%02d] %d : %f" % (i, num, v)
        v_list.append(v)
        total = total + v

    ave = total / float(len(v_list))
    print "average : %f" % (ave)

    for i in range(core.WEIGHT_INDEX_SIZE):
        dif = v_list[i] - ave
        print "[%02d] %f" % (i, dif)

    return 0

def setup_dnn(path, my_gpu):
    r = core.Roster()
    r.set_gpu(my_gpu)

    input_layer = r.add_layer(0, 784, 784)
    hidden_layer_1 = r.add_layer(1, 784, 32)
    hidden_layer_2 = r.add_layer(1, 32, 32)
    output_layer = r.add_layer(2, 32, 10)
    
    wi = util.csv_to_list(WEIGHT_INDEX_CSV_PATH)
    if len(wi)>0:
        print "restore weights"
        r.restore_weighgt(wi)
    else:
        print "init weights"
        r.init_weight()

    if my_gpu:
        r.update_weight()
    
    return r
#
#
#
def get_key_input(prompt):
    try:
        c = input(prompt)
    except:
        c = -1
    return c
#
#
#
def make_batch(size, image_path, label_path):
    batch = []
    
    file_in = open(label_path)
    header = file_in.read(LABEL_HEADER_SIZE)
    data = file_in.read()
    label_list = [0 for i in range(size)]
    for i in range(TEST_BATCH_SIZE):
        label = struct.unpack('>B', data[i])
        label_list[i] = label[0]
        
    file_in = open(image_path)
    header = file_in.read(IMAGE_HEADER_SIZE)
    for i in range(size):
        data = file_in.read(MNIST_IMAGE_SIZE)
        da = np.frombuffer(data, dtype=np.uint8)
        a_float = da.astype(np.float32) # convert from uint8 to float32
        batch.append((a_float, label_list[i]))

    return batch
#
#
#
def test(r, minibatch, num_of_class, debug=0):
    dist = [0,0,0,0,0,0,0,0,0,0]
    stat = [0,0,0,0,0,0,0,0,0,0]

    for label in range(len(minibatch)):
        data = minibatch[label]
        r.propagate(data)
        inf = r.get_inference()
        if inf is None:
            print "ERROR"
            continue
            
        index = -1
        mx = max(inf)
        if mx>0.0:
            for k in range(num_of_class):
                if inf[k] == mx:
                    index = k
            
            dist[index] = dist[index] + 1
        else:
            print "ASS HOLE : %d, %f" % (label, mx)
        
        if label==index:
            stat[index] = stat[index] + 1
         
    return [dist, stat]
#
#
#
def test_mode(r, batch, num_of_class, iteration, debug=0):
    print ">>test mode(%d)" % iteration
    start_time = time.time()
    
    it = 0
    dist = [0,0,0,0,0,0,0,0,0,0]
    stat = [0,0,0,0,0,0,0,0,0,0]
    labels = [0,0,0,0,0,0,0,0,0,0]
    
    for entry in batch:
        if it >= iteration:
            break
    
        data = entry[0]
        label = entry[1]
        labels[label] = labels[label] + 1
        r.propagate(data)
        inf = r.get_inference()
        if inf is None:
            print "ERROR"
            it = it + 1
            continue
    
        index = -1
        mx = max(inf)
        if mx>0.0:
            for k in range(num_of_class):
                if inf[k] == mx:
                    index = k
                                    
            dist[index] = dist[index] + 1
        else:
            print "[%d] ASS HOLE : %d, %f" % (it, label, mx)
            r.propagate(data, 1)
            it = it + 1
            continue
    
        if label==index:
            stat[index] = stat[index] + 1

        it = it + 1

    print "---------------------------------"
    print "class\t|label\t|dist\t|stat"
    print "---------------------------------"
    for i in range(num_of_class):
        print "%d\t| %d\t| %d\t| %d"  % (i, labels[i], dist[i], stat[i])
    
    print "---------------------------------"

    sum = 0.0
    for s in stat:
        sum = sum + s

    print "ok : %d" % (sum)
    print "it : %d" %(iteration)
    print "accuracy = %f" % (sum/float(iteration))

    elapsed_time = time.time() - start_time
    t = format(elapsed_time, "0")
    print "test time = %s" % (t)
#
#
#
def make_train_label(label, num):
    labelList = []
    
    for i in range(num):
        if label is i:
            labelList.append(1)
        else:
            labelList.append(0)

    return labelList
#
#
#
def evaluate_alt(r, data, data_class, num_of_class, w, wi_alt):
    sum_of_mse = 0.0
    labels = np.zeros(num_of_class, dtype=np.float32)
    labels[data_class] = 1.0
    
    r.propagate_alt(data, w, wi_alt)
    inf = r.get_inference()
    if inf is None:
        print "ERROR"
        sys.exit(0)

    ret = inf[data_class]
    mse =  util.mean_squared_error(inf, len(inf), labels, len(labels))
    return mse, ret
#
#
#
def evaluate(r, data, data_class, num_of_class):
    num = float(num_of_class)
    sum_of_mse = 0.0
    labels = np.zeros(num_of_class, dtype=np.float32)
    labels[data_class] = 1.0
        
    r.propagate(data)
    inf = r.get_inference()
    if inf is None:
        print "ERROR"
        sys.exit(0)

    ret = inf[data_class]
    mse = util.mean_squared_error(inf, len(inf), labels, len(labels))
    return mse, ret
#
#
#
def evaluate_minibatch(r, minibatch, num_of_class):
    num = float(num_of_class)
    sum_of_mse = 0.0
    #ret = 0.0
    labels = np.zeros(num_of_class, dtype=np.float32)
    
    for j in range(num_of_class):
        labels[j] = 1.0
        data = minibatch[j]
        r.propagate(data)
        inf = r.get_inference()
        if inf is None:
            print "ERROR"
            print r
            sys.exit(0)
        
        #print inf
        #print len(labels)
        mse =  util.mean_squared_error(inf, len(inf), labels, len(labels))
        #print mse
        sum_of_mse = sum_of_mse + mse
        labels[j] = 0.0
    
    #ret = ret / num
    mse = sum_of_mse / num
    return mse#, ret
#
#
#
def evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt):
    num = float(num_of_class)
    sum_of_mse = 0.0
    labels = np.zeros(num_of_class, dtype=np.float32)
    
    for j in range(num_of_class):
        labels[j] = 1.0
        data = minibatch[j]
        r.propagate_alt(data, w, wi_alt)
        inf = r.get_inference()
        if inf is None:
            print "ERROR"
            print r
            sys.exit(0)
        #print j
        #print inf
        
        mse =  util.mean_squared_error(inf, len(inf), labels, len(labels))
        #print mse
        if mse==0.1:
            print "FUCK!! %f" % mse
            mse = 10.0
                
        sum_of_mse = sum_of_mse + mse
        labels[j] = 0.0
    
    #ret = ret / num
    #print sum_of_mse
    mse = sum_of_mse / num
    return mse
#
#
#
def weight_shift_signed(r, minibatch, num_of_class, w, base_mse):
    inc = 0
    dec = 0
    wi = w.get_index()
    id = w.get_id()
    if wi>=core.WEIGHT_INDEX_ZERO:
        if wi==core.WEIGHT_INDEX_MAX:
            wi_alt = wi - 1
            mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
            if mse_alt<base_mse:
                w.set_index(wi_alt)
                dec = dec + 1
                print " - : %d > %d   | %f > %f" % (wi, wi_alt, base_mse, mse_alt)
        else:
            wi_alt = wi + 1
            mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
            if mse_alt<base_mse:
                w.set_index(wi_alt)
                inc = inc + 1
                print " + : %d > %d   | %f > %f" % (wi, wi_alt, base_mse, mse_alt)
            elif mse_alt==base_mse:
                pass
            else:
                wi_alt = wi - 1
                w.set_index(wi_alt)
                dec = dec + 1
                print " - : %d > %d   | %f > %f" % (wi, wi_alt, base_mse, mse_alt)
    else:
        wi_alt = wi + 1
        mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
        if mse_alt<=base_mse:
            w.set_index(wi_alt)
            inc = inc + 1
            print " + : %d > %d   | %f > %f" % (wi, wi_alt, base_mse, mse_alt)
        else:
            if wi==core.WEIGHT_INDEX_MIN:
                pass
            else:  # WEIGHT_INDEX_MIN < wi < WEIGHT_INDEX_ZERO
                wi_alt = wi - 1
                w.set_index(wi_alt)
                dec = dec + 1
                print " - : %d > %d   | %f > %f" % (wi, wi_alt, base_mse, mse_alt)

    return inc, dec
#
#
#
def weight_shift_blaze_4(r, minibatch, num_of_class, w, base_mse):
    wi = w.get_index()
    
    if w._lock==1:
        return 0, 0
    
    if w._step==0:
        if wi==core.WEIGHT_INDEX_MAX:
            #w.set_index(core.WEIGHT_INDEX_ZERO)
            w._step = -1
        #return 0, 0
        elif wi==core.WEIGHT_INDEX_MIN:
            #w.set_index(core.WEIGHT_INDEX_ZERO)
            w._step = 1
        #   return 0, 0
        else:
            wi_alt = wi+1
            mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
            if mse_alt<base_mse:
                w._step = 1
                w.set_index(wi_alt)
                return 1, 0
            
            wi_alt = wi-1
            mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
            if mse_alt<base_mse:
                w._step = -1
                w.set_index(wi_alt)
                return 0, 1
            else:
                #w._lock = 1
                w._step = 0
                return 0, 0

    if w._step==1 and wi==core.WEIGHT_INDEX_MAX:
        #w.set_index(core.WEIGHT_INDEX_ZERO)
        w._step = -1
        #return 0, 0
    elif w._step==-1 and wi==core.WEIGHT_INDEX_MIN:
        #w.set_index(core.WEIGHT_INDEX_ZERO)
        w._step = 1
        #return 0, 0

    wi_alt = wi + w._step
    mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
    if mse_alt<base_mse:
        w.set_index(wi_alt)
        if w._step==1:
            return 1, 0
        
        return 0, 1

    #w._lock = 1
    w._step = 0
    return 0, 0
#
#
#
def weight_shift_blaze_3(r, minibatch, num_of_class, w, base_mse):
    wi = w.get_index()
    
    if w._lock==1:
        return 0, 0
    
    if w._step==0:
        if wi==core.WEIGHT_INDEX_MAX:
            w.set_index(core.WEIGHT_INDEX_ZERO)
            w._step = 0
            return 0, 0
        elif wi==core.WEIGHT_INDEX_MIN:
            w.set_index(core.WEIGHT_INDEX_ZERO)
            w._step = 0
            return 0, 0

#        if wi==core.WEIGHT_INDEX_MAX or wi==core.WEIGHT_INDEX_MIN:
#            w._step = w._step * -1
#            wi_alt = wi + w._step
#            mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
#            if mse_alt<base_mse:
#                w.set_index(wi_alt)
#                if w._step==1:
#                    return 1, 0
#
#                return 0, 1
#
#            w._lock = 1
#            w._step = 0
#            return 0, 0
            
        else:
            wi_alt = wi+1
            mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
            if mse_alt<base_mse:
                w._step = 1
                w.set_index(wi_alt)
                return 1, 0
        
            wi_alt = wi-1
            mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
            if mse_alt<base_mse:
                w._step = -1
                w.set_index(wi_alt)
                return 0, 1
            else:
                w._lock = 1
                w._step = 0
                return 0, 0
    
    if w._step==1 and wi==core.WEIGHT_INDEX_MAX:
        w.set_index(core.WEIGHT_INDEX_ZERO)
        w._step = 0
        return 0, 0
    elif w._step==-1 and wi==core.WEIGHT_INDEX_MIN:
        w.set_index(core.WEIGHT_INDEX_ZERO)
        w._step = 0
        return 0, 0

    wi_alt = wi + w._step
    mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
    if mse_alt<base_mse:
        w.set_index(wi_alt)
        if w._step==1:
            return 1, 0
        
        return 0, 1

    w._lock = 1
    w._step = 0
    return 0, 0
#
#
#
def weight_shift_blaze_zero(r, minibatch, num_of_class, w, base_mse):
    wi = w.get_index()
    
    if w._lock==1:
        return 0, 0
    
    if w._step==0:
        if wi==core.WEIGHT_INDEX_MAX or wi==core.WEIGHT_INDEX_MIN:
            w._lock = 1
            w._step = 0
            return 0, 0
        
        wi_alt = wi+1
        mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
        if mse_alt<=base_mse:
            w._step = 1
            w.set_index(wi_alt)
            return 1, 0
        else:
            wi_alt = wi-1
            mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
            if mse_alt<=base_mse:
                w._step = -1
                w.set_index(wi_alt)
                return 0, 1
            else:
                w._lock = 1
                w._step = 0
                return 0, 0
    
    return 0, 0
#
#
#
def weight_shift_blaze_2(r, minibatch, num_of_class, w, base_mse):
    wi = w.get_index()
    
    if w._lock==1:
        return 0, 0
    
    if w._step==0:
        if wi==core.WEIGHT_INDEX_MAX:
            w._step = -1
            wi_alt = wi + w._step
            mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
            if mse_alt<base_mse:
                w.set_index(wi_alt)
                return 0, 1
            else:
                #w._lock = 1
                w._step = 1
                return 0, 0
        elif wi==core.WEIGHT_INDEX_MIN:
            w._step = 1
            wi_alt = wi + w._step
            mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
            if mse_alt<base_mse:
                w.set_index(wi_alt)
                return 1, 0
            else:
                #w._lock = 1
                w._step = -1
                return 0, 0
        else:
            wi_alt = wi + 1
            mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
            if mse_alt<base_mse:
                w._step = 1
                w.set_index(wi_alt)
                return 1, 0
            else:
                wi_alt = wi - 1
                mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
                if mse_alt<base_mse:
                    w._step = -1
                    w.set_index(wi_alt)
                    return 0, 1
                else:
                    w._lock = 1
                    return 0, 0
    elif w._step>0:
        if wi==core.WEIGHT_INDEX_MAX:
            w._step = -1
            return 0, 0
        else:
            wi_alt = wi + w._step
            mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
            if mse_alt<base_mse:
                w.set_index(wi_alt)
                return 1, 0
            else:
                w._lock = 1
                return 0, 0
    else: # w._step<0
        if wi==core.WEIGHT_INDEX_MIN:
            w._step = 1
            return 0, 0
        else:
            wi_alt = wi + w._step
            mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
            if mse_alt<base_mse:
                w.set_index(wi_alt)
                return 0, 1
            else:
                w._lock = 1
                return 0, 0


    return 0, 0
#
#
#
def weight_shift_blaze(r, minibatch, num_of_class, w, base_mse):
    wi = w.get_index()
    
    if w._lock==1:
        return 0, 0
    
    if wi==core.WEIGHT_INDEX_MAX:
        wi_alt = wi-1
        mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
        if mse_alt<base_mse:
            w._lock = 0
            w._step = -1
            w.set_index(wi_alt)
            return 0, 1
        else:
            #print "    lock 1"
            w._lock = 1
            w._step = 0
            return 0, 0

    elif wi==core.WEIGHT_INDEX_MIN:
        wi_alt = wi+1
        mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
        if mse_alt<base_mse:
            w._lock = 0
            w._step = 1
            w.set_index(wi_alt)
            return 1, 0
        else:
            #print "    lock 2"
            w._lock = 1
            w._step = 0
            return 0, 0
    else:
        if w._step==0:
            wi_alt = wi+1
            mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
            if mse_alt<base_mse:
                w._step = 1
                w.set_index(wi_alt)
                return 1, 0
            else:
                wi_alt = wi-1
                mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
                if mse_alt<base_mse:
                    w._step = -1
                    w.set_index(wi_alt)
                    return 0, 1
                else:
                    #print "    lock 3"
                    w._lock = 1
                    w._step = 0
                    return 0, 0

        elif w._step==1:
            wi_alt = wi+1
            mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
            if mse_alt<base_mse:
                w.set_index(wi_alt)
                return 1, 0
            else:
                #print "    lock 4"
                w._lock = 1
                w._step = 0
                return 0, 0

        elif w._step==-1:
            wi_alt = wi-1
            mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
            if mse_alt<base_mse:
                w.set_index(wi_alt)
                return 0, 1
            else:
                #print "    lock 5"
                w._lock = 1
                w._step = 0
                return 0, 0

    
    return 0, 0
#
#
#
def weight_shift_rigid(r, minibatch, num_of_class, w, base_mse):
    inc = 0
    dec = 0
    wi = w.get_index()
    id = w.get_id()
    
    if wi==core.WEIGHT_INDEX_MAX:
        wi_alt = wi - 1
        mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
        if mse_alt<=base_mse:
            w.set_index(wi_alt)
            dec = dec + 1
            #print " - : %d > %d   | %f > %f" % (wi, wi_alt, base_mse, mse_alt)

    elif wi==core.WEIGHT_INDEX_MIN:
        wi_alt = wi + 1
        mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
        if mse_alt<=base_mse:
            w.set_index(wi_alt)
            inc = inc + 1
            #print " + : %d > %d   | %f > %f" % (wi, wi_alt, base_mse, mse_alt)

    else:
        wi_alt = wi + 1
        mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
        if mse_alt<base_mse:
            w.set_index(wi_alt)
            inc = inc + 1
            #print " + : %d > %d   | %f > %f" % (wi, wi_alt, base_mse, mse_alt)

        else:
            wi_alt = wi - 1
            mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
            if mse_alt<base_mse:
                w.set_index(wi_alt)
                dec = dec + 1
                #print " - : %d > %d   | %f > %f" % (wi, wi_alt, base_mse, mse_alt)
                    
    return inc, dec
#
#
#
def weight_shift_rigid_single(r, data, data_class, num_of_class, w, base_mse):
    inc = 0
    dec = 0
    wi = w.get_index()
    id = w.get_id()
    
    if wi==core.WEIGHT_INDEX_MAX:
        wi_alt = wi - 1
        mse_alt, ret_alt = evaluate_alt(r, data, data_class, num_of_class, w, wi_alt)
        if mse_alt<=base_mse:
            w.set_index(wi_alt)
            dec = dec + 1
            print " - : %d > %d   | %f > %f" % (wi, wi_alt, base_mse, mse_alt)

    elif wi==core.WEIGHT_INDEX_MIN:
        wi_alt = wi + 1
        mse_alt, ret_alt = evaluate_alt(r, data, data_class, num_of_class, w, wi_alt)
        if mse_alt<=base_mse:
            w.set_index(wi_alt)
            inc = inc + 1
            print " + : %d > %d   | %f > %f" % (wi, wi_alt, base_mse, mse_alt)

    else:
        wi_alt = wi + 1
        mse_alt, ret_alt = evaluate_alt(r, data, data_class, num_of_class, w, wi_alt)
        if mse_alt<base_mse:
            w.set_index(wi_alt)
            inc = inc + 1
            print " + : %d > %d   | %f > %f" % (wi, wi_alt, base_mse, mse_alt)

        else:
            wi_alt = wi - 1
            mse_alt, ret_alt = evaluate_alt(r, data, data_class, num_of_class, w, wi_alt)
            if mse_alt<base_mse:
                w.set_index(wi_alt)
                dec = dec + 1
                print " - : %d > %d   | %f > %f" % (wi, wi_alt, base_mse, mse_alt)

    return inc, dec
#
#
#
def weight_shift_positive(r, minibatch, num_of_class, w, base_mse):
#    if w._lock==1:
#        return 0, 0
    inc = 0
    dec = 0
    wi = w.get_index()
    id = w.get_id()
    
    if wi==core.WEIGHT_INDEX_MAX:
        wi_alt = wi - 1
        mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
        if mse_alt<base_mse:
            w.set_index(wi_alt)
            dec = 1
#        else:
#            w._lock = 1

    elif wi==core.WEIGHT_INDEX_MIN:
        wi_alt = wi + 1
        mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
        if mse_alt<=base_mse:
            w.set_index(wi_alt)
            inc = 1
#        else:
#            w._lock = 1

    else:
        wi_alt = wi + 1
        mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
        if mse_alt<base_mse:
            w.set_index(wi_alt)
            inc = 1
        elif mse_alt>base_mse:
            wi_alt = wi - 1
            w.set_index(wi_alt)
            dec = 1
        else:
            w._lock = 1

    return inc, dec
#
#
#
def weight_shift_mse(r, minibatch, num_of_class, w, base_mse):
    wi = w.get_index()
    id = w.get_id()
    mse_alt = base_mse
    wi_alt = -1 # no change

    if wi==core.WEIGHT_INDEX_MAX:
        wi_alt = wi - 1
        mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
        if mse_alt<base_mse:
            #w.set_index(wi_alt)
            pass
        else:
            wi_alt = -1
    elif wi==core.WEIGHT_INDEX_MIN:
        wi_alt = wi + 1
        mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
        if mse_alt<=base_mse:
            #w.set_index(wi_alt)
            pass
        else:
            wi_alt = -1
    else:
        wi_alt = wi + 1
        mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
        if mse_alt<base_mse:
            #w.set_index(wi_alt)
            pass
        elif mse_alt>base_mse:
            wi_alt = wi - 1
            #w.set_index(wi_alt)
        else:
            wi_alt = -1

    return wi_alt, mse_alt
#
#
#
def weight_shift(r, minibatch, num_of_class, w, base_mse):
    inc = 0
    dec = 0
    wi = w.get_index()
    id = w.get_id()
    if wi>=core.WEIGHT_INDEX_MAX:
        #print "%d : MAX" % (id)
        #continue
        wi_alt = wi - 1
        mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
        if mse_alt<base_mse:
            print "     MAX(%d) : %d > %d | %f > %f" % (id, wi, wi_alt, base_mse, mse_alt)
            w.set_index(wi_alt)
            dec = dec + 1
        else:
            print "     %d : %d NOC @MAX" % (id, wi)
    elif wi==core.WEIGHT_INDEX_MIN:
        #print "%d : MIN" % (id)
        #continue
        wi_alt = wi + 1
        mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
        if mse_alt<base_mse:
            print "     MIN(%d) : %d > %d | %f > %f" % (id, wi, wi_alt, base_mse, mse_alt)
            w.set_index(wi_alt)
            inc = inc + 1
        else:
            pass
            print "     %d : %d NOC @MIN" % (id, wi)
    else:
        wi_alt = wi + 1
        mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
        if mse_alt<base_mse:
            print "     %d : %d > %d | %f > %f" % (id, wi, wi_alt, base_mse, mse_alt)
            w.set_index(wi_alt)
            inc = inc + 1
        else:
            wi_alt = wi - 1
            mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, wi_alt)
            if mse_alt<base_mse:
                print "     %d : %d > %d | %f > %f" % (id, wi, wi_alt, base_mse, mse_alt)
                w.set_index(wi_alt)
                dec = dec + 1
            else:
                pass
                print "     %d : %d NOC" % (id, wi)

    return inc, dec
#
#
#
def weight_scan(r, minibatch, num_of_class, w, base_mse):
    wi = w.get_index()
    id = w.get_id()
    min = base_mse
    min_index = -1
    #min_index = core.WEIGHT_INDEX_ZERO
    
    #mse_alt, ret_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, 0)
    min = base_mse
    for i in range(core.WEIGHT_INDEX_SIZE):
        mse_alt = evaluate_minibatch_alt(r, minibatch, num_of_class, w, i)
        #print " %d : %f" % (i, mse_alt)
        if mse_alt<min:
            min = mse_alt
            min_index = i

    if min_index>=0:
        print "     %d -> %d" % (wi, min_index)
        w.set_index(min_index)
#
#
#
def process_minibatch_layer_by_layer(r, minibatch, num_of_class):
    c = r.countLayers()
    wcnt = 0
    inc_total = 0
    dec_total = 0
    for index in range(1, c):
        layer = r.getLayerAt(index)
        w_num = layer._num_node * layer._num_input
        num = w_num / 10 / index
        #
        # mse
        base_mse = evaluate_minibatch(r, minibatch, num_of_class)
        #
        for k in range(num):
            n = random.randint(0, layer._num_node-1)
            i = random.randint(0, layer._num_input-1)
            rwi = n*layer._num_input + i
            w = r._weight_list[wcnt + rwi]
            wi = w.get_index()
            #print "%d : %d/%d (%d, %d) %d" % (index, k, num, n, i, rwi)
            #print "     %f : %f" %(core.WEIGHT_SET[wi], layer.get_weight(n, i))
            inc, dec = weight_shift_signed(r, minibatch, num_of_class, w, base_mse)
            #inc, dec = weight_shift_positive(r, minibatch, num_of_class, w, base_mse)
            #weight_scan(r, minibatch, num_of_class, w, base_mse)
            inc_total = inc_total + inc
            dec_total = dec_total + dec
        # weight update
        wcnt = wcnt + layer._num_node * layer._num_input
        r.update_weight()
        # renew mse
        #base_mse = evaluate_minibatch(r, minibatch, num_of_class)

    #print "inc=%d, dec=%d" % (inc_total, dec_total)
    return inc_total, dec_total
#
#
#
def process_minibatch_layer_by_layer_reversed(r, minibatch, num_of_class):
    c = r.countLayers()
    wcnt = 0
    inc_total = 0
    dec_total = 0
    for index in range(c-1, 0, -1):
        layer = r.getLayerAt(index)
        w_num = layer._num_node * layer._num_input
        num = w_num / 10 / index
        # mse
        base_mse = evaluate_minibatch(r, minibatch, num_of_class)
        for k in range(num):
            n = random.randint(0, layer._num_node-1)
            i = random.randint(0, layer._num_input-1)
            rwi = n*layer._num_input + i
            w = r._weight_list[wcnt + rwi]
            wi = w.get_index()
            #inc, dec = weight_shift_signed(r, minibatch, num_of_class, w, base_mse)
            inc, dec = weight_shift_positive(r, minibatch, num_of_class, w, base_mse)
            inc_total = inc_total + inc
            dec_total = dec_total + dec
        
        # weight update
        wcnt = wcnt + layer._num_node * layer._num_input
        r.update_weight()

    return inc_total, dec_total
#
#
#
def process_minibatch_layer_by_layer_reversed_even(r, minibatch, num_of_class):
    c = r.countLayers()
    wcnt = 0
    inc_total = 0
    dec_total = 0
    for index in range(c-1, 0, -1):
        layer = r.getLayerAt(index)
        num = layer._num_input / 10 * 2
        print "%d/%d" % (num, layer._num_input)
        # mse
        base_mse = evaluate_minibatch(r, minibatch, num_of_class)
        for node_index in range(layer._num_node):
            #print node_index
            for k in range(num):
                i = random.randint(0, layer._num_input-1)
                rwi = node_index*layer._num_input + i
                w = r._weight_list[wcnt + rwi]
                wi = w.get_index()
                #
                #inc, dec = weight_shift_signed(r, minibatch, num_of_class, w, base_mse)
                #inc, dec = weight_shift_positive(r, minibatch, num_of_class, w, base_mse)
                inc, dec = weight_shift_rigid(r, minibatch, num_of_class, w, base_mse)
                #print "    inc=%d, dec=%d " % (inc, dec)
                #
                inc_total = inc_total + inc
                dec_total = dec_total + dec
    
        print "    inc=%d, dec=%d " % (inc_total, dec_total)
        # weight update
        wcnt = wcnt + layer._num_node * layer._num_input
        r.update_weight()
    
    return inc_total, dec_total
#
#
#
def process_minibatch_1st_hidden_layer(r, minibatch, num_of_class):
    inc_total = 0
    dec_total = 0
    c = r.countLayers()
    wcnt = 0
    layer = r.getLayerAt(1)
    w_num = layer._num_node * layer._num_input
    num = w_num/10

    base_mse = evaluate_minibatch(r, minibatch, num_of_class)
        #
    for k in range(num):
        n = random.randint(0, layer._num_node-1)
        i = random.randint(0, layer._num_input-1)
        rwi = n*layer._num_input + i
        w = r._weight_list[rwi]
        wi = w.get_index()
        #inc, dec = weight_shift_signed(r, minibatch, num_of_class, w, base_mse)
        inc, dec = weight_shift_positive(r, minibatch, num_of_class, w, base_mse)
        inc_total = inc_total + inc
        dec_total = dec_total + dec

    r.update_weight()
    
    return inc_total, dec_totalweight_shift_randomweight_shift_random
#
#
#
def process_minibatch_parallel(r, minibatch, num_of_class):
    weight_list = r.get_weight_list()
    w_num = len(weight_list)
    inc_total = 0
    dec_total = 0
    num = w_num / 10
    
    base_mse = evaluate_minibatch(r, minibatch, num_of_class)
    for i in range(num):
        k = random.randrange(w_num)
        w = weight_lis[k]
        inc, dec = weight_shift_signed(r, minibatch, num_of_class, w, base_mse)
        inc_total = inc_total + inc
        dec_total = dec_total + dec
        
    r.update_weight()
    return inc_total, dec_total
#
#
#
def process_single(r, data, data_class, num_of_class):
    print "class=%d" %(data_class)
    c = r.countLayers()
    wcnt = 0
    inc_total = 0
    dec_total = 0
    for index in range(c-1, 0, -1):
        layer = r.getLayerAt(index)
        num = layer._num_input / 10
        print num
        # mse
        base_mse = evaluate(r, data, data_class, num_of_class)
        for node_index in range(layer._num_node):
            #if index==c-1:
            #    if node_index==data_class:
            #        num = num*2
            #    else:
            #        continue
            #
            print "class %d, layer %d, node %d" % (data_class, index, node_index)
            #
            for k in range(num):
                i = random.randint(0, layer._num_input-1)
                rwi = node_index*layer._num_input + i
                w = r._weight_list[wcnt + rwi]
                wi = w.get_index()
                #
                inc, dec = weight_shift_rigid_single(r, data, data_class, num_of_class, w, base_mse)
                #
                inc_total = inc_total + inc
                dec_total = dec_total + dec
        
        # weight update
        wcnt = wcnt + layer._num_node * layer._num_input
        r.update_weight()
    
    return inc_total, dec_total
#
#
#
def train_mode_one_by_one(r, train_batch, it_train, num_of_class, num_of_processed):
    print ">> train mode"
    print len(train_batch)
    print "train (%d, %d)" % (num_of_processed, it_train)
    inc = 0
    dec = 0
    start = num_of_processed
    epoc = 8# currently, epoch is fixed to 1
    k = 0 # iteration
    total_start_time = time.time()
    #
    for i in range(start, start+it_train, 1):
        minibatch = train_batch[i]
        start_time = time.time()
        #
        for data_class in range(num_of_class):
            for j in range(epoc):
                data = minibatch[data_class]
                inc, dec  = process_single(r, data, data_class, num_of_class)
                print "[%d/%d] inc=%d, dec=%d" % (i, start+it_train, inc, dec)
        #
        elapsed_time = time.time() - start_time
        t = format(elapsed_time, "0")
        print "[%03d|%03d] %s" % (k, it_train, t)
        k = k + 1
    
    total_time = time.time() - total_start_time
    t = format(total_time, "0")
    print "[total time] %s" % (t)
    return k
#
#
#
def evaluate_minibatch_alt_2(r, minibatch, size, w, wi_alt):
    num = float(size)
    sum_of_mse = 0.0
    labels = np.zeros(NUM_OF_CLASS, dtype=np.float32)
    
    for entry in minibatch:
        data = entry[0]
        label = entry[1]
        labels[label] = 1.0
        r.propagate_alt(data, w, wi_alt)
        inf = r.get_inference()
        if inf is None:
            print "ERROR"
            print r
            sys.exit(0)
        
        mse =  util.mean_squared_error(inf, len(inf), labels, len(labels))
        #print mse
        #if mse==0.1:
        #    print "ASS"
        #    mse = 10.0
        
        sum_of_mse = sum_of_mse + mse
        labels[label] = 0.0
    
    mse = sum_of_mse / num
    #print mse
    return mse
#
#
#
def weight_shift_positive_2(r, minibatch, size, w, base_mse):
    inc = 0
    dec = 0
    wi = w.get_index()
    id = w.get_id()
    
    if wi==core.WEIGHT_INDEX_MAX:
        wi_alt = wi - 1
        mse_alt = evaluate_minibatch_alt_2(r, minibatch, size, w, wi_alt)
        if mse_alt<base_mse:
            w.set_index(wi_alt)
            dec = 1

    elif wi==core.WEIGHT_INDEX_MIN:
        wi_alt = wi + 1
        mse_alt = evaluate_minibatch_alt_2(r, minibatch, size, w, wi_alt)
        if mse_alt<=base_mse:
            w.set_index(wi_alt)
            inc = 1

    else:
        wi_alt = wi + 1
        mse_alt = evaluate_minibatch_alt_2(r, minibatch, size, w, wi_alt)
        if mse_alt<base_mse:
            w.set_index(wi_alt)
            inc = 1
        elif mse_alt>base_mse:
            wi_alt = wi - 1
            w.set_index(wi_alt)
            dec = 1
        else:
            w._lock = 1
            print "    << locked >>"

    return inc, dec
#
#
#
def weight_shift_random(r, minibatch, size, w, base_mse):
    inc = 0
    dec = 0
    wi = w.get_index()
    id = w.get_id()
    w._lock = 1
    
    wi_alt = random.randrange(core.WEIGHT_INDEX_SIZE)
    while wi_alt==wi:
        wi_alt = random.randrange(core.WEIGHT_INDEX_SIZE)
    
    mse_alt = evaluate_minibatch_alt_2(r, minibatch, size, w, wi_alt)
    if mse_alt<base_mse:
        w.set_index(wi_alt)
        print "  %d > %d (%f, %f)" % (wi, wi_alt, base_mse, mse_alt)
        return 1
    else:
        print "  no change (%d, %d) (%f, %f)" % (wi, wi_alt, base_mse, mse_alt)

    return 0
#
#
#
def evaluate_minibatch_2(r, minibatch, size):
    num = float(size)
    sum_of_mse = 0.0
    labels = np.zeros(NUM_OF_CLASS, dtype=np.float32)
    
    for entry in minibatch:
        data = entry[0]
        label = entry[1]
        labels[label] = 1.0
        r.propagate(data)
        inf = r.get_inference()
        if inf is None:
            print "ERROR"
            print r
            sys.exit(0)
    
        mse =  util.mean_squared_error(inf, len(inf), labels, len(labels))
        sum_of_mse = sum_of_mse + mse
        labels[label] = 0.0
    
    mse = sum_of_mse / num
    return mse
#
#
#
def process_minibatch(r, minibatch, size):
    total = 0
    weight_list = r.get_weight_list()
    
    base_mse = evaluate_minibatch_2(r, minibatch, size)
    c = r.countLayers()
    for i in range(1, c):
        layer = r.getLayerAt(i)
        num_node = layer._num_node
        #
        node_index_list = list(range(num_node))
        random.shuffle(node_index_list)
        for k in node_index_list:
            node = layer.get_node(k)
            num_w = len(node._w_id_list)# / 10 / i
            w_id_list = list(range(num_w))
            random.shuffle(w_id_list)
            #
            patial = num_w/5/i
            w_id_list = w_id_list[0:patial]
            #
            for p in w_id_list:
                print "layer: %d, node: %d, weight: %d" % (i, k, p)
                w_index = node.get_weight(p)
                w = weight_list[w_index]
                #
                ret = weight_shift_random(r, minibatch, size, w, base_mse)
                #
                if ret>0:
                    r.update_weight()
                    base_mse = evaluate_minibatch_2(r, minibatch, size)
                
                total = total + ret

#            r.update_weight()
#            base_mse = evaluate_minibatch_2(r, minibatch, size)

    r.update_weight()
    print "tatal = %d" % (total)
    return total
#
#
#
def train_mode(r, batch, size):
    print ">> train mode"
    inc = 0
    dec = 0
    total = 0
    minibatch = batch[0:size]
    print len(minibatch)
    total_start_time = time.time()
    #
    for j in range(1):
        total = process_minibatch(r, minibatch, size)
    #
    total_time = time.time() - total_start_time
    t = format(total_time, "0")
    print "[total time] %s" % (t)
#
#
#
def batch_evaluate(r, batch, batch_size, labels):
    num = float(NUM_OF_CLASS)
    sum = 0.0
    for i in range(batch_size):
        entry = batch[i]
        data_class = entry[1]
        labels[data_class] = 1.0
        inf = r._batch_out[i]
        #mse =  util.mean_squared_error(inf, len(inf), labels, len(labels))
        mse =  util.cross_emtropy_error(inf, len(inf), labels, len(labels))
        sum = sum + mse
        labels[data_class] = 0.0

    return sum/float(batch_size)
#
#
#
def batch_weight_shift(r, batch, batch_size, labels, w, mse_base):
    wi = w.get_index()
    wi_alt = wi
    if w._lock==1:
        print "  locked"
        return mse_base, 0

    if w._step==0:
        if wi==core.WEIGHT_INDEX_MAX:
            w._step  = -1
        else:
            w._step  = 1

        r.batch_propagate_all(w, wi+w._step, 0)
        mse_alt = batch_evaluate(r, batch, batch_size, labels)
        if mse_alt<mse_base:
            w.set_index(wi+w._step)
            r.update_weight()
            print "  %d > %d : %f > %f" % (wi, wi+w._step , mse_base, mse_alt)
            return mse_alt, 1
        else:
            w._lock = 1
            print "  lock at initial eval. (%d)" % (wi)
            return mse_base, 0

    if wi==core.WEIGHT_INDEX_MAX:
        w._step = -1
        print "  reversed at %d" % (wi)
    elif wi==core.WEIGHT_INDEX_MIN:
        w._step = 1
        print "  reversed at %d" % (wi)

    r.batch_propagate_all(w, wi+w._step, 0)
    mse_alt = batch_evaluate(r, batch, batch_size, labels)
    if  mse_alt<mse_base:
        w.set_index(wi+w._step)
        r.update_weight()
        print "  %d > %d : %f > %f" % (wi, wi+w._step, mse_base, mse_alt)
        return mse_alt, 1
    else:
        w._lock = 1
        print "  lock at %d" % (wi)

    return mse_base, 0
#
#
#
def batch_weight_random(r, batch, batch_size, labels, w, mse_base):
    inc = 0
    dec = 0
    wi = w.get_index()
    id = w.get_id()
    if w._lock==1:
        print "  locked"
        return mse_base, 0
    
    wi_alt = random.randrange(core.WEIGHT_INDEX_SIZE)
    while wi_alt==wi:
        wi_alt = random.randrange(core.WEIGHT_INDEX_SIZE)
    
    r.batch_propagate_all(w, wi_alt, 0)
    mse_alt = batch_evaluate(r, batch, batch_size, labels)
    if mse_alt<mse_base:
        w.set_index(wi_alt)
        r.update_weight()
        w._lock=1
        print "  %d > %d (%f, %f)" % (wi, wi_alt, mse_base, mse_alt)
        return mse_alt, 1
        
    print "  no change (%d, %d) (%f, %f)" % (wi, wi_alt, mse_base, mse_alt)
    return mse_base, 0
#
#
#
def batch_train_mode(it, r, batch, batch_size):
#    r.makeBatchBufferIn(28*28, batch_size)
#    r.makeBatchBufferOut(NUM_OF_CLASS, batch_size)
    labels = np.zeros(NUM_OF_CLASS, dtype=np.float32)
    cnt_update = 0
#
#    for i in range(batch_size):
#        entry = batch[i]
#        r._batch_data[i] = entry[0].copy()
#        r._batch_class[i] = entry[1].copy()
#
    #r._gpu.copy(r._gpu_batch_in, r._batch_in)
    #
    #
#    layer = r.getLayerAt(0)
#    r._gpu.copy(layer._gpu_input, r._batch_data)
#
#    layer.makeBatchBuffer(28*28, batch_size)
#    for i in range(batch_size):
#        layer._gpu.batch_scale(r._gpu_batch_in, layer._gpu_batch_mem, i, 28*28, float(255.0), layer._num_node, 0)
#
    #
    r.set_batch(batch, batch_size, 28*28):
    #
    #
    start_time = time.time()
    r.batch_propagate_all(None, None, 0)
    mse_base = batch_evaluate(r, batch, batch_size, labels)
    print mse_base
    #
    weight_list = r.get_weight_list()
    c = r.countLayers()
    #for i in range(1, c):
    for i in range(c-1, 0, -1):
        layer = r.getLayerAt(i)
        num_node = layer._num_node
        #
        node_index_list = list(range(num_node))
        random.shuffle(node_index_list)
        cnt0 = 0
        cnt1 = 1
        for k in node_index_list:
            node = layer.get_node(k)
            num_w = len(node._w_id_list)# / 10 / i
            w_id_list = list(range(num_w))
            random.shuffle(w_id_list)
            #
            patial = num_w/5/i
            w_id_list = w_id_list[0:patial]
            #
            for p in w_id_list:
                print "%d, %d [%d, %d] layer: %d, node: %d, weight: %d" % (it, cnt_update, cnt0, cnt1, i, k, p)
                w_index = node.get_weight(p)
                w = weight_list[w_index]
                #
                mse_base, c = batch_weight_shift(r, batch, batch_size, labels, w, mse_base)
                #mse_base, c = batch_weight_random(r, batch, batch_size, labels, w, mse_base)
                cnt_update = cnt_update + c
                #
                cnt1 = cnt1 + 1
            #
            cnt0 = cnt0 + 1
    
    #
    elasped_time = time.time() - start_time
    t = format(elasped_time, "0")
    print "[elasped time] %s" % (t)
    return cnt_update

#
#
#
def save_weight(r):
    wl = r.get_weight_list()
    wi_list = []
    for w in wl:
        wi_list.append( w.get_index() )

    util.list_to_csv(WEIGHT_INDEX_CSV_PATH, wi_list)
#
#
#
def load_weight(r):
    wi_list = util.csv_to_list(WEIGHT_INDEX_CSV_PATH)
    wl = r.get_weight_list()
    return
#
#
#
def main():
    argvs = sys.argv
    argc = len(argvs)
    #
    minibatch_size = 100
    #
    # GPU
    #
    my_gpu = gpu.Gpu()
    my_gpu.set_kernel_code()
    #
    num_of_processed = util.pickle_load(PROCEEDED_PATH)
    if num_of_processed is None:
        num_of_processed = 0
 
    r = setup_dnn(NETWORK_PATH, my_gpu)
    if r is None:
        print "fatal DNN error"
        return 0

    print "0 : make train batch"
    print "1 : make test batch"
    print "2 : train"
    print "3 : test"
    print "4 : self-test"
    print "5 : ..."
    print "6 : export weight to CSV"
    print "7 : new minibatch"
    print "8 : save and quit"
    print "9 : weight distribution"
    mode = get_key_input("input command >")
    if mode==0:
        print ">> make train batch"
        batch = make_batch(TRAIN_BATCH_SIZE, TRAIN_IMAGE_PATH, TRAIN_LABEL_PATH)
        util.pickle_save(TRAIN_BATCH_PATH, batch)
    elif mode==1:
        print ">> make test batch"
        batch = make_batch(TEST_BATCH_SIZE, TEST_IMAGE_PATH, TEST_LABEL_PATH)
        util.pickle_save(TEST_BATCH_PATH, batch)
    elif mode==2:
        print ">> train mode : max_it_train = %d" % (num_of_processed)
        batch = util.pickle_load(TRAIN_BATCH_PATH)
        if batch is None:
            print "error : no train batch"
            return 0
        
        train_mode(r, batch, minibatch_size)
        save_weight(r)
    elif mode==3:
        print ">> test mode"
        debug = 0
        test_batch = util.pickle_load(TEST_BATCH_PATH)
        if test_batch is None:
            print "error : no test batch"
            return 0
        
        test_mode(r, test_batch, NUM_OF_CLASS, TEST_BATCH_SIZE, debug)
    elif mode==4:
        print ">> self-test mode"
        debug = 0
        train_array = util.pickle_load(TRAIN_BATCH_PATH)
        test_mode(r, train_array, NUM_OF_CLASS, minibatch_size, debug)
    elif mode==5:
        print ">> batch on GPU"
        start_time = time.time()
        #
        batch = util.pickle_load(TRAIN_BATCH_PATH)
        if batch is None:
            print "error : no train batch"
            return 0

        cnt = 0
        for i in range(1):
            cnt = cnt + batch_train_mode(i, r, batch, minibatch_size)
            save_weight(r)
        #
        elasped_time = time.time() - start_time
        t = format(elasped_time, "0")
        print "[total elasped time] %s" % (t)
        print "tatal update = %d" % (cnt)
    elif mode==6:
        print ">> export weight to CSV"
        wl = r.get_weight_list()
        wi_list = []
        for w in wl:
            wi_list.append( w.get_index() )

        print len(wi_list)
        util.list_to_csv("./test.cvs", wi_list)
        d_list = util.csv_to_list("./test.cvs")
        print len(d_list)
    elif mode==7:
        train_mode_new_minibatch(r)
        save_weight(r)
    elif mode==8:
        print ">> save and quit"
        save_weight(r)
    elif mode==9:
        print ">> check_weight_distribution"
        check_weight_distribution()
    else:
        print "error : mode = %d" % mode
    
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
