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
    hidden_layer_3 = r.add_layer(1, 32, 32)
    hidden_layer_4 = r.add_layer(1, 32, 32)
    output_layer = r.add_layer(2, 32, 10)
    
    if os.path.isfile(WEIGHT_INDEX_CSV_PATH):
        r.import_weight_index(WEIGHT_INDEX_CSV_PATH)
    else:
        print "init weights"
        r.init_weight()

    if my_gpu:
        r.update_weight()
    
    r.export_weight_index(WEIGHT_INDEX_CSV_PATH)
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
def evaluate(r, batch_size, labels):
    infs = r.get_inference()
    sum = 0.0
    for i in range(batch_size):
        data_class = r._batch_class[i]
        labels[data_class] = 1.0
        inf = infs[i]
        #print inf
        #
        mse = util.cross_emtropy_error_2(inf, len(inf), labels, len(labels))
        #mse = util.mean_squared_error_np(inf, len(inf), labels, len(labels))
        #if mse==np.nan:
        #    #print mse
        #    #mse = 100.0
        #    print mse
        #    sum = sum + 100.0
        #else:
        #    sum = sum + mse
    
        #mse =  util.cross_emtropy_error_fast(inf, labels, data_class)
        #mse =  util.cross_emtropy_error_fast(inf, data_class)
        #print mse
        #
        #mse = util.mean_squared_error(inf, len(inf), labels, len(labels))
        #
        sum = sum + mse
        labels[data_class] = 0.0

    #print infs[0]
    return sum/float(batch_size)
#
#
#
#def weight_random(r, batch, batch_size, labels, w, mse_base):
#    inc = 0
#    dec = 0
#    wi = w.get_index()
#    id = w.get_id()
#    if w._lock==1:
#        print "  locked"
#        return mse_base, 0
#
#    wi_alt = random.randrange(core.WEIGHT_INDEX_SIZE)
#    while wi_alt==wi:
#        wi_alt = random.randrange(core.WEIGHT_INDEX_SIZE)
#
#    #r.propagate(w, wi_alt, 0)
#    r.propagate()
#    mse_alt = evaluate(r, batch_size, labels)
#    if mse_alt<mse_base:
#        w.set_index(wi_alt)
#        r.update_weight()
#        w._lock=1
#        print "  %d > %d (%f, %f)" % (wi, wi_alt, mse_base, mse_alt)
#        return mse_alt, 1
#
#    print "  no change (%d, %d) (%f, %f)" % (wi, wi_alt, mse_base, mse_alt)
#    return mse_base, 0
#
#
#
def weight_shift(r, batch, batch_size, li, ni, ii, mse_base, labels):
    #def weight_shift(r, batch, batch_size, labels, w, mse_base):
    layer = r.getLayerAt(li)
    wi = layer.get_weight_index(ni, ii)
    wp = layer.get_weight_property(ni, ii)
    lock = layer.get_weight_lock(ni, ii)
    #wi = w.get_index()
    wi_alt = wi

    if lock==1:
        print "  locked"
        return mse_base, 0

    if wp==0:
        if wi==core.WEIGHT_INDEX_MAX:
            wp  = -1
            #layer.set_weight_property(ni, ii, -1)
        else:
            wp  = 1

        #r.propagate()
        r.propagate(li, ni, ii, wi+wp, 0)
        mse_alt = evaluate(r, batch_size, labels)
        if mse_alt<mse_base:
            #w.set_index(wi+w._step)
            #r.update_weight()
            layer.set_weight_property(ni, ii, wp)
            layer.set_weight_index(ni, ii, wi + wp)
            layer.update_weight_gpu()
            print "  %d > %d : %f > %f" % (wi, wi+wp , mse_base, mse_alt)
            return mse_alt, 1
        else:
            #w._lock = 1
            layer.set_weight_lock(ni, ii, 1)
            print "  lock at initial eval. (%d) [%f]" % (wi, mse_alt)
            return mse_base, 0

    if wi==core.WEIGHT_INDEX_MAX:
        w._step = -1
        print "  reversed at %d" % (wi)
    elif wi==core.WEIGHT_INDEX_MIN:
        w._step = 1
        print "  reversed at %d" % (wi)

    r.propagate()
    mse_alt = evaluate(r, batch_size, labels)
    if  mse_alt<mse_base:
        layer.set_weight_property(ni, ii, wp)
        layer.set_weight_index(ni, ii, wi + wp)
        layer.update_weight_gpu()
        #w.set_index(wi+w._step)
        #r.update_weight()
        print "  %d > %d : %f > %f" % (wi, wi+wp, mse_base, mse_alt)
        return mse_alt, 1
    else:
        #w._lock = 1
        layer.set_weight_lock(ni, ii, 1)
        print "  lock (%d) [%f]" % (wi, mse_alt)

    return mse_base, 0
#
#
#
def weight_shift_2(r, batch, batch_size, li, ni, ii, mse_base, labels):
    layer = r.getLayerAt(li)
    lock = layer.get_weight_lock(ni, ii)
    if lock>1:
        print "  locked(%d, %d)" % (ni, li)
        return mse_base, 0
    
    wp = layer.get_weight_property(ni, ii)
    wi = layer.get_weight_index(ni, ii)
    wi_alt = wi
    
    if wp==0:
        if wi==core.WEIGHT_INDEX_MAX:
            layer.set_weight_property(ni, ii, -1)
            wp = - 1
        else:
            layer.set_weight_property(ni, ii,  1)
            wp = 1
    
        #layer.set_weight_lock(ni, ii, 1)
    
        wi_alt = wi + wp
        r.propagate(li, ni, ii, wi+wp, 0)
        mse_alt = evaluate(r, batch_size, labels)
        print "  - %d > %d : %f > %f" % (wi, wi + wp , mse_base, mse_alt)
        if mse_alt<=mse_base:
            layer.set_weight_index(ni, ii, wi + wp)
            layer.update_weight_gpu()
            return mse_alt, 1
        else:
            print "  reversed A %d" % (wi)
            layer.set_weight_property(ni, ii, wp*-1)
            layer.set_weight_lock(ni, ii, 1)
            return mse_base, 0

    if wi==core.WEIGHT_INDEX_MAX:
        layer.set_weight_property(ni, ii, -1)
        wp = -1
        print "  reversed B %d" % (wi)
        layer.set_weight_lock(ni, ii, lock+1)
        return mse_base, 0
    elif wi==core.WEIGHT_INDEX_MIN:
        layer.set_weight_property(ni, ii, 1)
        wp = 1
        print "  reversed C %d" % (wi)
        layer.set_weight_lock(ni, ii, lock+1)
        if lock>0:
            print "  lock"
        return mse_base, 0

    wi_alt = wi + wp
    r.propagate(li, ni, ii, wi_alt, 0)
    mse_alt = evaluate(r, batch_size, labels)
    print "  = %d > %d : %f > %f" % (wi, wi_alt , mse_base, mse_alt)
    if  mse_alt<=mse_base:
        layer.set_weight_index(ni, ii, wi_alt)
        layer.update_weight_gpu()
        return mse_alt, 1
    #
    layer.set_weight_lock(ni, ii, lock+1)
    return mse_base, 0
#
#
#
def weight_shift_3(r, batch, batch_size, li, ni, ii, mse_base, labels):
    layer = r.getLayerAt(li)
    wp = layer.get_weight_property(ni, ii) # initial value is 1
    wi = layer.get_weight_index(ni, ii)
    wi_alt = wi
    
    #if wp==0:
    #    print "  locked"
    #    return mse_base, 0

    if wi==core.WEIGHT_INDEX_MAX or wi==core.WEIGHT_INDEX_MIN:
        wp = wp + 1
        layer.set_weight_property(ni, ii, wp)
        print "  reversed at (%d)" % (wi)
        return mse_base, 0

    if wp % 2==1:
        wi_alt = wi + 1
    else:
        wi_alt = wi - 1

    r.propagate(li, ni, ii, wi_alt, 0)
    mse_alt = evaluate(r, batch_size, labels)
    print "  = %d > %d : %f > %f" % (wi, wi_alt , mse_base, mse_alt)
    if  mse_alt<mse_base:
        layer.set_weight_index(ni, ii, wi_alt)
        layer.update_weight_gpu()
        return mse_alt, 1
    #
    if wp>1:
        print "  lock (%d)" % wi
        layer.set_weight_property(ni, ii, 0)

    return mse_base, 0

#
#
#
def train_mode(it, r, batch, batch_size, data_size):
    labels = np.zeros(NUM_OF_CLASS, dtype=np.float32)
    cnt_update = 0
    r.set_batch(batch, batch_size, data_size)
    #
    start_time = time.time()
    r.propagate()
    mse_base = evaluate(r, batch_size, labels)
    print mse_base
    
    cnt0 = 0
    lcntl = [0,0,0,0]
    c = r.countLayers()
    #for li in range(1, c):
    for li in range(c-1, 0, -1):
        layer = r.getLayerAt(li)
        num_node = layer._num_node
        num_w = layer._num_input
        #
        node_index_list = list(range(num_node))
        random.shuffle(node_index_list)
        lcnt = 0
        for ni in node_index_list:
            patial = num_w/5/li
            for p in range(patial):
                ii = random.randrange(num_w)
                print "%d : %d [%d] (%d, %d, %d) %f" % (it, cnt0, cnt_update, li, ni, ii, mse_base)
                mse_base, c = weight_shift_2(r, batch, batch_size, li, ni, ii, mse_base, labels)
                #
                cnt_update = cnt_update + c
                cnt0 = cnt0 + 1
                lcnt = lcnt + c
            #
        lcntl[li] = lcnt
    #
    elasped_time = time.time() - start_time
    t = format(elasped_time, "0")
    print "[elasped time] %s" % (t)
    print lcntl
    return cnt_update
#
#
#
def weight_heat(r, batch, batch_size, li, ni, ii, mse_base, labels):
    layer = r.getLayerAt(li)
    lock = layer.get_weight_lock(ni, ii)
    if lock>0:
        print "  locked(%d, %d)" % (ni, li)
        return mse_base, 0
    
    wp = layer.get_weight_property(ni, ii)
    if wp<0:
        print "  skip(%d, %d)" % (ni, li)
        return mse_base, 0

    wi = layer.get_weight_index(ni, ii)
    wi_alt = wi

    if wi==core.WEIGHT_INDEX_MAX:
        #print "  lock(%d, %d)" % (ni, li)
        #if wp==0:
        #    layer.set_weight_property(ni, ii, -1)
        #layer.set_weight_lock(ni, ii, 1) ## here, need to check
        #
        print "  reset(%d, %d)" % (ni, li)
        layer.set_weight_property(ni, ii, 0)
        return mse_base, 0

    if wi+1==core.WEIGHT_INDEX_MAX:
        wi_alt = wi+1
    else:
        wi_alt = random.randrange(wi+1, core.WEIGHT_INDEX_MAX, 1)

    r.propagate(li, ni, ii, wi_alt, 0)
    mse_alt = evaluate(r, batch_size, labels)
    print "  - %d > %d : %f > %f" % (wi, wi_alt , mse_base, mse_alt)
    #
    if mse_alt<mse_base: # <=
        layer.set_weight_property(ni, ii, 1)
        layer.set_weight_index(ni, ii, wi_alt)
        layer.update_weight_gpu()
        return mse_alt, 1
    else:
        #if wi_alt==wi+1:
        #    print "  lock(%d, %d)" % (ni, li)
        #    layer.set_weight_property(ni, ii, -1)
        #    layer.set_weight_lock(ni, ii, 1) # this need to be checked
        #    return mse_base, 0
        #
        #layer.set_weight_property(ni, ii, 1)
        print "  reset(%d, %d)" % (ni, li)
        layer.set_weight_property(ni, ii, 0)

    return mse_base, 0
#
#
#
def weight_cool(r, batch, batch_size, li, ni, ii, mse_base, labels):
    layer = r.getLayerAt(li)
    lock = layer.get_weight_lock(ni, ii)
    if lock>0:
        print "  locked(%d, %d)" % (ni, li)
        return mse_base, 0

    wp = layer.get_weight_property(ni, ii)
    if wp>0:
        print "  skip(%d, %d)" % (ni, li)
        #print "  reset(%d, %d)" % (ni, li)
        #layer.set_weight_property(ni, ii, 0)
        return mse_base, 0
    
    
    wi = layer.get_weight_index(ni, ii)
    wi_alt = wi-1

    if wi==core.WEIGHT_INDEX_MIN:
        #print "  lock(%d, %d)" % (ni, li)
        #layer.set_weight_lock(ni, ii, 1)
        print "  reset(%d, %d)" % (ni, li)
        layer.set_weight_property(ni, ii, 0)
        return mse_base, 0
    
    r.propagate(li, ni, ii, wi_alt, 0)
    mse_alt = evaluate(r, batch_size, labels)
    print "  - %d > %d : %f > %f" % (wi, wi_alt , mse_base, mse_alt)
    #
    if mse_alt<mse_base: # <=
        layer.set_weight_index(ni, ii, wi_alt)
        layer.update_weight_gpu()
        return mse_alt, 1
    else:
        #layer.set_weight_lock(ni, ii, 1)
        print "  reset(%d, %d)" % (ni, li)
        layer.set_weight_property(ni, ii, 0)
        return mse_base, 0

    return mse_base, 0
#
#
#
def train_mode_3(it, r, batch, batch_size, data_size):
    divider = 4
    #
    labels = np.zeros(NUM_OF_CLASS, dtype=np.float32)
    cnt_update = 0
    r.set_batch(batch, batch_size, data_size)
    #
    start_time = time.time()
    r.propagate()
    mse_base = evaluate(r, batch_size, labels)
    print mse_base


    #
    # heating phase
    #
    cnt = 0
    h_cnt = 0
    c = r.countLayers()
    for li in range(1, c):
        layer = r.getLayerAt(li)
        num_node = layer._num_node
        num_w = layer._num_input
        #
        node_index_list = list(range(num_node))
        random.shuffle(node_index_list)
        nc = 0
        for ni in node_index_list:
            nc = nc + 1
            w_p = num_w/divider
            if li==1:
                w_p = w_p/5

            for p in range(w_p):
                ii = random.randrange(num_w)
                print "It[%d], H: L=%d, N=%d/%d, W=%d/%d : update=%d/%d, W(%d,%d,%d), CE:%f" % (it, li, nc, num_node, p, w_p, h_cnt, cnt, li, ni, ii, mse_base)
                mse_base, ret = weight_heat(r, batch, batch_size, li, ni, ii, mse_base, labels)
                #
                cnt = cnt + 1
                h_cnt = h_cnt + ret
    #
    # cooling phase
    #
    cnt = 0
    c_cnt = 0
    for li in range(1, c):
        layer = r.getLayerAt(li)
        num_node = layer._num_node
        num_w = layer._num_input
        #
        node_index_list = list(range(num_node))
        random.shuffle(node_index_list)
        nc = 0
        for ni in node_index_list:
            nc = nc + 1
            w_p = num_w/divider
            if li==1:
                w_p = w_p/5

            for p in range(w_p):
                ii = random.randrange(num_w)
                print "It[%d], C: L=%d, N=%d/%d, W=%d/%d : update=%d/%d, W(%d,%d,%d), CE:%f" % (it, li, nc, num_node, p, w_p, c_cnt, cnt, li, ni, ii, mse_base)
                mse_base, ret = weight_cool(r, batch, batch_size, li, ni, ii, mse_base, labels)
                #
                cnt = cnt + 1
                c_cnt = c_cnt + ret
    #
    return h_cnt, c_cnt, mse_base
#
#
#
def train_mode_2(it, r, batch, batch_size, data_size):
    labels = np.zeros(NUM_OF_CLASS, dtype=np.float32)
    cnt_update = 0
    r.set_batch(batch, batch_size, data_size)
    #
    start_time = time.time()
    r.propagate()
    mse_base = evaluate(r, batch_size, labels)
    print mse_base

    cnt0 = 0
    #lcntl = [0,0,0,0,0,0,0,0,0,0]
    c = r.countLayers()
    #
    # heating phase
    #
    for li in range(1, c):
        for k in range(20):
            layer = r.getLayerAt(li)
            num_node = layer._num_node
            num_w = layer._num_input
            #
            lcnt = 0
            node_index_list = list(range(num_node))
            random.shuffle(node_index_list)
            nc = 0
            for ni in node_index_list:
                nc = nc + 1
                w_p = num_w
                if li==1:
                    w_p = num_w/20
                
                for p in range(w_p):
                    ii = random.randrange(num_w)
                    print "It[%d][%d], H: L=%d, N=%d/%d, W=%d/%d : all=%d, update=%d, W(%d,%d,%d), CE:%f" % (it, k, li, nc, num_node, p, w_p, cnt0, cnt_update, li, ni, ii, mse_base)
                    mse_base, ret = weight_heat(r, batch, batch_size, li, ni, ii, mse_base, labels)
                    #
                    cnt_update = cnt_update + ret
                    cnt0 = cnt0 + 1
                    lcnt = lcnt + ret
                #
            #
        #
        #lcntl[li] = lcnt
    #
    # cooling phase
    #
    for li in range(1, c):
        for k in range(20):
            layer = r.getLayerAt(li)
            num_node = layer._num_node
            num_w = layer._num_input
            #
            node_index_list = list(range(num_node))
            random.shuffle(node_index_list)
            lcnt = 0
            for ni in node_index_list:
                w_p = num_w /20
                if w_p<=0:
                    w_p = 2
                for p in range(num_w/20):
                    ii = random.randrange(num_w)
                    #print "(%d) C[%d] : all=%d, update=%d, W(%d,%d,%d), CE:%f" % (it, k, cnt0, cnt_update, li, ni, ii, mse_base)
                    print "It[%d][%d], C: L=%d, N=%d/%d, W=%d/%d : all=%d, update=%d, W(%d,%d,%d), CE:%f" % (it, k, li, nc, num_node, p, w_p, cnt0, cnt_update, li, ni, ii, mse_base)
                    mse_base, ret = weight_cool(r, batch, batch_size, li, ni, ii, mse_base, labels)
                    #
                    cnt_update = cnt_update + ret
                    cnt0 = cnt0 + 1
                    lcnt = lcnt + ret
                #
            #
        #
        #lcntl[li] = lcnt
    #
    #
    #
    elasped_time = time.time() - start_time
    t = format(elasped_time, "0")
    print "[elasped time] %s" % (t)
    #print lcntl
    return cnt_update
#
#
#
def test_mode(r, batch, batch_size):
    print ">>batch test mode (%d)" % (batch_size)
    dist = [0,0,0,0,0,0,0,0,0,0] # data_class
    rets = [0,0,0,0,0,0,0,0,0,0] # result of infs
    oks  = [0,0,0,0,0,0,0,0,0,0] # num of correct
    #
    start_time = time.time()
    r.set_batch(batch, batch_size, 28*28, 0)

    r.propagate(-1, -1, -1, -1, 0)
    infs = r.get_inference()
    #print infs
    #r.propagate(3, 0, 0, 1)
    #infs = r.get_inference()
    #print infs
    #
    #layer = r.getLayerAt(3)
    #layer.set_weight_index(0, 0, 1)
    #layer.update_weight_gpu()
    #r.propagate()
    #infs = r.get_inference()
    #print infs
    
    
    #
    ca = 0
    for i in range(batch_size):
        dist[r._batch_class[i]] = dist[r._batch_class[i]] + 1
        inf = infs[i]
        #print inf
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
def loop(it, r, batch, batch_size, data_size):
    h_cnt_list = []
    c_cnt_list = []
    ce_list = []
    
    limit =0.01
    #limit =0.000001
    pre_ce = 0.0
    
    for i in range(it):
        #h_cnt, c_cnt, ce = train_mode_3(i, r, batch, batch_size, data_size)
        h_cnt, c_cnt, ce = train_at_random(i, r, batch, batch_size, data_size)
        #
        h_cnt_list.append(h_cnt)
        c_cnt_list.append(c_cnt)
        ce_list.append(ce)
        r.export_weight_index(WEIGHT_INDEX_CSV_PATH)
        if pre_ce == ce:
            print "locked with local optimum"
            print "exit iterations"
            break
        #
        if ce<limit:
            print "exit iterations"
            break
        #
        pre_ce = ce
        #
        if (i+1)%10==0:
            print "RESET WP"
            r.reset_weight_property()
            r.unlock_weight_all()
        #
    k = len(h_cnt_list)
    for j in range(k):
        print "%d, %d, %d, %f," % (j, h_cnt_list[j], c_cnt_list[j], ce_list[j])
#
#
#
def loop_SA(it, r, batch, batch_size, data_size):
    labels = np.zeros(NUM_OF_CLASS, dtype=np.float32)
    w_list = []
    r.set_batch(batch, batch_size, data_size)
    #
    c = r.countLayers()
    for li in range(1, c):
        layer = r.getLayerAt(li)
        num_node = layer._num_node
        num_w = layer._num_input
        for ni in range(num_node):
            for ii in range(num_w):
                wi = layer.get_weight_index(ni, ii)
                w = core.Weight(li, ni, ii, wi)
                w_list.append(w)

    w_cnt = len(w_list)
    num_update = 100
    #print update_wi_list
    #
    r.propagate()
    mse_base = evaluate(r, batch_size, labels)
    print mse_base
    #
    for i in range(it):
        update_wi_list = random.sample(range(0, w_cnt, 1), num_update)
        for k in update_wi_list:
            w = w_list[k]
            li, ni, ii = w.get_index()
            layer = r.getLayerAt(li)
            wi = layer.get_weight_index(ni, ii)
            wi_alt = wi
            if wi==core.WEIGHT_INDEX_MAX:
                pass
            elif wi+1==core.WEIGHT_INDEX_MAX:
                wi_alt = wi+1
            else:
                wi_alt = random.randrange(wi+1, core.WEIGHT_INDEX_MAX, 1)
            #
            layer.set_weight_index(ni, ii, wi_alt)
            w.set_wi(wi_alt)
        #
        layer.update_weight_gpu()
        r.propagate()
        mse_alt = evaluate(r, batch_size, labels)
        print "[%d] %f >> %f" % (i, mse_base, mse_alt)
        if mse_alt<mse_base:
            mse_base = mse_alt
            print "    keep"
        else:
            #print "    discard"
            for k in update_wi_list:
                w = w_list[k]
                li, ni, ii = w.get_index()
                layer = r.getLayerAt(li)
                wi = w.alternate_wi()
                layer.set_weight_index(ni, ii, wi)
            #
            layer.update_weight_gpu()
        #
    # for i
#
#
#
#      train_mode_3(it, r, batch, batch_size, data_size):
def train_at_random(it, r, batch, batch_size, data_size):
    limit = 0.01
    h_cnt = 0
    c_cnt = 0
    labels = np.zeros(NUM_OF_CLASS, dtype=np.float32)
    w_list = []
    r.set_batch(batch, batch_size, data_size)
    #
    c = r.countLayers()
    for li in range(1, c):
        layer = r.getLayerAt(li)
        num_node = layer._num_node
        num_w = layer._num_input
        for ni in range(num_node):
            for ii in range(num_w):
                wi = layer.get_weight_index(ni, ii)
                w = core.Weight(li, ni, ii, wi)
                w_list.append(w)

    w_cnt = len(w_list)
    num_update = w_cnt/4 #
    #
    r.propagate()
    mse_base = evaluate(r, batch_size, labels)
    print mse_base
    #
    update_list = random.sample(range(0, w_cnt, 1), num_update)
    j = 0
    for k in update_list:
        w = w_list[k]
        li, ni, ii = w.get_index()
        mse_base, ret = weight_heat(r, batch, batch_size, li, ni, ii, mse_base, labels)
        h_cnt = h_cnt + ret
        print "[%d] H(%d), %d, %d, W(%d,%d,%d), CE:%f" % (it, num_update, j, h_cnt, li, ni, ii, mse_base)
        j = j+1
    
        if mse_base<limit:
            print "exit iterations"
            return h_cnt, c_cnt, mse_base
        
    #
    update_list = random.sample(range(0, w_cnt, 1), num_update)
    j = 0
    for k in update_list:
        w = w_list[k]
        li, ni, ii = w.get_index()
        mse_base, ret = weight_cool(r, batch, batch_size, li, ni, ii, mse_base, labels)
        c_cnt = c_cnt + ret
        print "[%d] C=%d, %d, %d, W(%d,%d,%d), CE:%f" % (it, num_update, j, c_cnt, li, ni, ii, mse_base)
        j = j+1
            
        if mse_base<limit:
            print "exit iterations"
            break
    #
    return h_cnt, c_cnt, mse_base
#
#
#
def main():
    argvs = sys.argv
    argc = len(argvs)
    #
    it = 20*10
    batch_size = 100 # 100, 500, 1000, 1500
    data_size = 28*28
    #
    # GPU
    #
    platform_id = 0
    # 0 : AMD Server
    # 1 : Intel on MBP
    # 2 : eGPU (AMD Radeon Pro 580)
    device_id = 1
    #
    my_gpu = gpu.Gpu(platform_id, device_id)
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
    print "2 : train (single)"
    print "3 : test"
    print "4 : self-test"
    print "5 : evaluate and save"
    print "6 : init WI"
    print "7 : train (loop)"
    print "8 : "
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
        print ">> train (single)"
        start_time = time.time()
        #
        batch = util.pickle_load(TRAIN_BATCH_PATH)
        if batch is None:
            print "error : no train batch"
            return 0
        #
        i = 0
        h_cnt, c_cnt, mse = train_mode_3(i, r, batch, batch_size, data_size)
        h_cnt_list.append(h_cnt)
        
        r.export_weight_index(WEIGHT_INDEX_CSV_PATH)
        print "%d, %d, %d, %f" % (i, h_cnt, c_cnt, mse)
        #
        elasped_time = time.time() - start_time
        t = format(elasped_time, "0")
        print "[total elasped time] %s" % (t)
    elif mode==3:
        print ">> test mode"
        debug = 0
        batch = util.pickle_load(TEST_BATCH_PATH)
        if batch is None:
            print "error : no test batch"
            return 0
    
        batch_size = TEST_BATCH_SIZE
        test_mode(r, batch, batch_size)
    elif mode==4:
        print ">> self-test mode"
        debug = 0
        batch = util.pickle_load(TRAIN_BATCH_PATH)
        test_mode(r, batch, batch_size)
    elif mode==5:
        print ">> evaluate and save"
        debug = 0
        batch = util.pickle_load(TEST_BATCH_PATH)
        labels = np.zeros(NUM_OF_CLASS, dtype=np.float32)
        r.set_batch(batch, batch_size, data_size)
        r.propagate()
        mse_base = evaluate(r, batch_size, labels)
        print mse_base
        #r.export_weight_index("./wi.csv")
    elif mode==6:
        print ">> init WI"
        #
    elif mode==7:
        print ">> train (loop)"
        start_time = time.time()
        batch = util.pickle_load(TRAIN_BATCH_PATH)
        if batch is None:
            print "error : no train batch"
            return 0
        #
        loop(it, r, batch, batch_size, data_size)
        #
        elasped_time = time.time() - start_time
        t = format(elasped_time, "0")
        print "[total elasped time] %s" % (t)
        pass
    elif mode==8:
        print ">> none"
        pass
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
