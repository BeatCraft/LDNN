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
    
    save_weight(r)
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
def batch_evaluate(r, batch_size, labels):
    infs = r.get_batch_inference()
    sum = 0.0
    for i in range(batch_size):
        data_class = r._batch_class[i]
        labels[data_class] = 1.0
        inf = infs[i]
        #print inf
        #
        mse =  util.cross_emtropy_error(inf, len(inf), labels, len(labels))
        #mse =  util.cross_emtropy_error_fast(inf, labels, data_class)
        #mse =  util.cross_emtropy_error_fast(inf, data_class)
        #print mse
        #
        #mse = util.mean_squared_error(inf, len(inf), labels, len(labels))
        #
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

        r.batch_propagate(w, wi+w._step, 0)
        mse_alt = batch_evaluate(r, batch_size, labels)
        if mse_alt<mse_base:
            w.set_index(wi+w._step)
            r.update_weight()
            print "  %d > %d : %f > %f" % (wi, wi+w._step , mse_base, mse_alt)
            return mse_alt, 1
        else:
            w._lock = 1
            print "  lock at initial eval. (%d) [%f]" % (wi, mse_alt)
            r.batch_propagate(None, None, 0) #
            return mse_base, 0

    if wi==core.WEIGHT_INDEX_MAX:
        w._step = -1
        print "  reversed at %d" % (wi)
    elif wi==core.WEIGHT_INDEX_MIN:
        w._step = 1
        print "  reversed at %d" % (wi)

    r.batch_propagate(w, wi+w._step, 0)
    mse_alt = batch_evaluate(r, batch_size, labels)
    if  mse_alt<mse_base:
        w.set_index(wi+w._step)
        r.update_weight()
        print "  %d > %d : %f > %f" % (wi, wi+w._step, mse_base, mse_alt)
        return mse_alt, 1
    else:
        w._lock = 1
        print "  lock (%d) [%f]" % (wi, mse_alt)

    r.batch_propagate(None, None, 0) #
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
    
    r.batch_propagate(w, wi_alt, 0)
    mse_alt = batch_evaluate(r, batch_size, labels)
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
def batch_test_mode(r, batch, batch_size):
    print ">>batch test mode (%d)" % (batch_size)
    dist = [0,0,0,0,0,0,0,0,0,0] # data_class
    rets = [0,0,0,0,0,0,0,0,0,0] # result of infs
    oks  = [0,0,0,0,0,0,0,0,0,0] # num of correct
    #
    start_time = time.time()
    r.set_batch(batch, batch_size, 28*28)
    r.batch_propagate(None, None, 0)
    infs = r.get_batch_inference()
    #
    ca = 0
    for i in range(batch_size):
        dist[r._batch_class[i]] = dist[r._batch_class[i]] + 1
        inf = infs[i]
        index = -1
        mx = max(inf)
        if mx>0.0:
            for k in range(10):
                if inf[k] == mx:
                    index = k
        #
        rets[index] = rets[index] + 1
        #
        if index==r._batch_class[i]:
            oks[index] = oks[index] +1
            ca = ca + 1
    #
    print "---------------------------------"
    print "result : %d / %d" % (ca, batch_size)
    accuracy = float(ca) / float(batch_size)
    print "accuracy : %f" % (accuracy)
    print "---------------------------------"
    print "class\t|dist\t|infs\t|ok"
    print "---------------------------------"
    for i in range(10):
        print "%d\t| %d\t| %d\t| %d"  % (i, dist[i], rets[i], oks[i])
    
    print "---------------------------------"
    #
    elapsed_time = time.time() - start_time
    t = format(elapsed_time, "0")
    print "test time = %s" % (t)
#
#
#
def batch_train_mode(it, r, batch, batch_size, data_size):
    labels = np.zeros(NUM_OF_CLASS, dtype=np.float32)
    cnt_update = 0
    r.set_batch(batch, batch_size, data_size)
    #
    start_time = time.time()
    r.batch_propagate(None, None, 0)
    mse_base = batch_evaluate(r, batch_size, labels)
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
            num_w = len(node._w_id_list)
            w_id_list = list(range(num_w))
            random.shuffle(w_id_list)
            #
            patial = num_w/5/i
            w_id_list = w_id_list[0:patial]
            #
            for p in w_id_list:
                print "[%d] %d, %d [%d, %d] layer: %d, node: %d, weight: %d : %f" % (cnt_update, it, cnt_update, cnt0, cnt1, i, k, p, mse_base)
                w_index = node.get_weight(p)
                w = weight_list[w_index]
                #
                #mse_base, c = batch_weight_random(r, batch, batch_size, labels, w, mse_base)
                #
                mse_base, c = batch_weight_shift(r, batch, batch_size, labels, w, mse_base)
                #
                cnt_update = cnt_update + c
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
    batch_size = 1000
    data_size = 28*28
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
        print ">> batch on GPU"
        start_time = time.time()
        #
        batch = util.pickle_load(TRAIN_BATCH_PATH)
        if batch is None:
            print "error : no train batch"
            return 0

        cnt = 0
        for i in range(1):
            cnt = cnt + batch_train_mode(i, r, batch, batch_size, data_size)
            save_weight(r)
            r.unlock_weight_all()
        #
        elasped_time = time.time() - start_time
        t = format(elasped_time, "0")
        print "[total elasped time] %s" % (t)
        print "tatal update = %d" % (cnt)
        save_weight(r)
    elif mode==3:
        print ">> test mode"
        debug = 0
        batch = util.pickle_load(TEST_BATCH_PATH)
        if batch is None:
            print "error : no test batch"
            return 0
    
        batch_size = TEST_BATCH_SIZE
        batch_test_mode(r, batch, batch_size)
    elif mode==4:
        print ">> self-test mode"
        debug = 0
        batch = util.pickle_load(TRAIN_BATCH_PATH)
        batch_test_mode(r, batch, batch_size)
    elif mode==5:
        print ">> ??"
        batch = util.pickle_load(TRAIN_BATCH_PATH)
        batch_test_mode(r, batch, batch_size)
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
