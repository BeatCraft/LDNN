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
#NETWORK_PATH     = "./network.pickle"
#PROCEEDED_PATH   = "./proceeded.pickle"

TRAIN_IMAGE_PATH  = "./MNIST/train-images-idx3-ubyte"
TRAIN_LABEL_PATH = "./MNIST/train-labels-idx1-ubyte"
TEST_IMAGE_PATH   = "./MNIST/t10k-images-idx3-ubyte"
TEST_LABEL_PATH   = "./MNIST/t10k-labels-idx1-ubyte"

MNIST_IMAGE_WIDTH  = 28
MNIST_IMAGE_HEIGHT = 28
MNIST_IMAGE_SIZE   = MNIST_IMAGE_WIDTH*MNIST_IMAGE_HEIGHT
NUM_OF_CLASS     = 10    # 0,1,2,3,4,5,6,7,8,9
NUM_OF_SAMPLES   = 5000  # must be 5,000 per a class
NUM_OF_TEST      = 500

TRAIN_BATCH_SIZE  = 60000
TEST_BATCH_SIZE   = 10000
IMAGE_HEADER_SIZE = 16
LABEL_HEADER_SIZE  = 8

WEIGHT_INDEX_CSV_PATH   = "./wi.csv"
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
    num_list = []
    total = 0.0
    for i in range(core.WEIGHT_INDEX_SIZE):
        key = str(i)
        num =  w_list.count(key)
        num_list.append(num)
        v = float(num)/w_total*100.0
        print "[%02d] %d : %f" % (i, num, v)
        v_list.append(v)
        total = total + v

    ave = total / float(len(v_list))
    print "average : %f" % (ave)

    for i in range(core.WEIGHT_INDEX_SIZE):
        dif = v_list[i] - ave
        print "[%02d] %f" % (i, dif)

    for i in range(core.WEIGHT_INDEX_SIZE):
        print num_list[i]
    
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
    
    if os.path.isfile(path):
        r.import_weight_index(path)
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
def evaluate(r, batch_size, labels):
    return r.get_cross_entropy()

    infs = r.get_inference()
    
    sum = 0.0
    for i in range(batch_size):
        data_class = r._batch_class[i]
        labels[data_class] = 1.0
        #print labels
        inf = infs[i]
        #print inf
        mse = util.cross_emtropy_error_2(inf, len(inf), labels, len(labels))
        sum = sum + mse
        labels[data_class] = 0.0

    return sum/float(batch_size)
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
    r.set_batch(batch, batch_size, 28*28, 10, 0)

    r.propagate(-1, -1, -1, -1, 0)
    infs = r.get_inference()
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
def weight_shift_mode(r, batch, batch_size, li, ni, ii, mse_base, labels, mode):
    layer = r.getLayerAt(li)
    wp = layer.get_weight_property(ni, ii) # default : 0
    lock = layer.get_weight_lock(ni, ii)   # default : 0
    wi = layer.get_weight_index(ni, ii)
    wi_alt = wi
    #
    if lock>0:
        print "    locked"
        return mse_base, 0
    #
    if mode>0: # heat
        if wi==core.WEIGHT_INDEX_MAX:
            print "    skip : MAX"
            layer.set_weight_property(ni, ii, 0)
            return mse_base, 0
#            if wp==mode:
#                print "    lock : MAX"
#                layer.set_weight_lock(ni, ii, 1)
#                return mse_base, 0
#            else:
#                print "    skip : MAX"
#                return mse_base, 0
            #
        #
    else:
        if wi==core.WEIGHT_INDEX_MIN:
            print "    skip : MIN"
            layer.set_weight_property(ni, ii, 0)
            return mse_base, 0
#            if wp==mode:
#                print "    lock : MIN"
#                layer.set_weight_lock(ni, ii, 1)
#                return mse_base, 0
#            else:
#                print "    skip : MIN"
#                return mse_base, 0
        #
    #
    wi_alt = wi + mode
    r.propagate(li, ni, ii, wi_alt, 0)
    mse_alt = evaluate(r, batch_size, labels)
    if  mse_alt<mse_base:
        print "    update : %d >> %d" % (wi, wi_alt)
        layer.set_weight_property(ni, ii, mode)
        layer.set_weight_index(ni, ii, wi_alt)
        layer.update_weight_gpu()
        return mse_alt, 1
        
#    elif mse_alt>mse_base:
#        if wp!=0:
#            print "    lock : REV"
#            layer.set_weight_lock(ni, ii, 1)
#            return mse_base, 0
#        else:
#            print "    skip : REV"
#            layer.set_weight_property(ni, ii, mode*-1)
#            return mse_base, 0
    #
    layer.set_weight_property(ni, ii, 0)
    print "    skip : 0"
    return mse_base, 0
#
#
#
def train(it, r, batch, batch_size, data_size, limit):
    divider = 4
    t_cnt = 0
    h_cnt = 0
    c_cnt = 0
    labels = np.zeros(NUM_OF_CLASS, dtype=np.float32)
    w_list = []
    r.set_batch(batch, batch_size, data_size, 10)
    #
    r.propagate()
    mse_base = evaluate(r, batch_size, labels)
    print mse_base
    
    #return 0, 0, 0.0
    #
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
            for p in range(w_p):
                ii = random.randrange(num_w)
                print "[%d] H=%d/%d, N(%d/%d), W(%d/%d) : W(%d,%d,%d), CE:%f" % (it, h_cnt, t_cnt, nc, num_node, p, w_p, li, ni, ii, mse_base)
                mse_base, ret = weight_shift_mode(r, batch, batch_size, li, ni, ii, mse_base, labels, 1)
                #
                h_cnt = h_cnt + ret
                t_cnt = t_cnt +1
                if mse_base<limit:
                    print "exit iterations"
                    return h_cnt, c_cnt, mse_base
                #
            #
        #
    #
    t_cnt = 0
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
            for p in range(w_p):
                ii = random.randrange(num_w)
                print "[%d] C=%d/%d, N(%d/%d), W(%d/%d) : W(%d,%d,%d), CE:%f" % (it, c_cnt, t_cnt, nc, num_node, p, w_p, li, ni, ii, mse_base)
                mse_base, ret = weight_shift_mode(r, batch, batch_size, li, ni, ii, mse_base, labels, -1)
                #
                c_cnt = c_cnt + ret
                t_cnt = t_cnt + 1
                if mse_base<limit:
                    print "exit iterations"
                    return h_cnt, c_cnt, mse_base
                #
            #
        #
    #
    return h_cnt, c_cnt, mse_base
#
#
#
def loop(it, r, batch, batch_size, data_size):
    h_cnt_list = []
    c_cnt_list = []
    ce_list = []
    
    #limit = 1.0
    #limit = 0.01
    #limit = 0.01
    limit = 0.000001
    pre_ce = 0.0
    
    for i in range(it):
        h_cnt, c_cnt, ce = train(i, r, batch, batch_size, data_size, limit)
        #
        h_cnt_list.append(h_cnt)
        c_cnt_list.append(c_cnt)
        ce_list.append(ce)
        r.export_weight_index(WEIGHT_INDEX_CSV_PATH)
        #
        save_path = "./test/wi.csv.%f" % ce
        r.export_weight_index(save_path)
        #
        if pre_ce == ce:
            print "locked with local optimum"
            print "exit iterations"
            break
        #
        if ce<limit:
            print "exit iterations"
            break
        #
        r.reset_weight_property()
        r.unlock_weight_all()
        pre_ce = ce
    #
    k = len(h_cnt_list)
    for j in range(k):
        print "%d, %d, %d, %f," % (j, h_cnt_list[j], c_cnt_list[j], ce_list[j])
    
    #
#
#
#
def init_WI(r, batch, batch_size, data_size):
    r.set_batch(batch, batch_size, data_size, 10)
    labels = np.zeros(NUM_OF_CLASS, dtype=np.float32)
    #
    for i in range(100):
        r.init_weight()
        r.update_weight()
        r.propagate()
        cr =  evaluate(r, batch_size, labels)
        print cr
        #
        save_path = "./wi/wi.csv.%f" % cr
        r.export_weight_index(save_path)
    #
#
#
#
def main():
    argvs = sys.argv
    argc = len(argvs)
    #
    it = 20*10
    batch_size = 2000# 100, 500, 1000, 1500
    data_size = 28*28
    #
    # GPU
    #
    platform_id = 0
    # 0 : AMD Server
    # 1 : Intel on MBP
    # 2 : eGPU (AMD Radeon Pro 580)
    device_id = 0
    #
    my_gpu = gpu.Gpu(platform_id, device_id)
    my_gpu.set_kernel_code()
    #
    #num_of_processed = util.pickle_load(PROCEEDED_PATH)
    #if num_of_processed is None:
    #    num_of_processed = 0
    
    path = WEIGHT_INDEX_CSV_PATH
    if argc==2:
        path = argvs[1]
    #
    r = setup_dnn(path, my_gpu)
    if r is None:
        print "fatal DNN error"
        return 0

    print "0 : make train batch"
    print "1 : make test batch"
    print "2 : "
    print "3 : test"
    print "4 : self-test"
    print "5 : evaluate"
    print "6 : init WI"
    print "7 : train (loop)"
    print "8 : weight distribution"
    print "9 : save"
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
        pass
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
        print ">> evaluate"
        debug = 0
        batch = util.pickle_load(TRAIN_BATCH_PATH)
        #labels = np.zeros(NUM_OF_CLASS, dtype=np.float32)
        r.set_batch(batch, batch_size, data_size, 10, debug)
        r.propagate()
        #mse_base = evaluate(r, batch_size, labels)
        mse_base = r.get_cross_entropy()
        print mse_base
        #r.export_weight_index("./wi.csv")
    elif mode==6:
        print ">> init WI"
        batch = util.pickle_load(TRAIN_BATCH_PATH)
        init_WI(r, batch, batch_size, data_size)
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
    elif mode==8:
        print ">> check_weight_distribution"
        check_weight_distribution()
    elif mode==9:
        print ">> save"
        r.export_weight_index(WEIGHT_INDEX_CSV_PATH)
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
