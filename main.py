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

import multiprocessing
#
#
# LDNN Modules
import core
import util
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
NETWORK_PATH     = "./network.pickle"
NUM_OF_CLASS     = 10    # 0,1,2,3,4,5,6,7,8,9
NUM_OF_SAMPLES   = 5000  # must be 5,000 per a class
NUM_OF_TEST      = 500
#
#
#
def setup_dnn(path):
    if os.path.exists(path):
        print "DNN restored."
        r = util.pickle_load(path)
        return r
    else:
        r = core.Roster()
        
        inputLayer = r.addLayer(196, 0)    # 0 : input
        hiddenLayer_1 = r.addLayer(32, 1)  # 1 : hiddeh
        hiddenLayer_2 = r.addLayer(32, 1)  # 2 : hiddeh
        outputLayer = r.addLayer(10, 2)    # 3 : output
        
        r.connectLayers(inputLayer, hiddenLayer_1)
        r.connectLayers(hiddenLayer_1, hiddenLayer_2)
        r.connectLayers(hiddenLayer_2, outputLayer)

        util.pickle_save(path, r)
        return r
    return None
#
#
#
def setup_batch(minibatch_size, it_train, it_test):
    save_data = util.pickle_load(TRAIN_BATCH_PATH)
    if save_data is None:
        train_list = scan_data(TRAIN_BASE_PATH, NUM_OF_SAMPLES, NUM_OF_CLASS)
        train_batch = make_batch(NUM_OF_CLASS, it_train, minibatch_size, train_list)
        
        test_list = scan_data(TEST_BASE_PATH, NUM_OF_TEST, NUM_OF_CLASS)
        test_batch = make_batch(NUM_OF_CLASS, it_test, minibatch_size, test_list)
        
        save_data = [0, train_batch, test_batch]
        util.pickle_save(TRAIN_BATCH_PATH, save_data)
    else:
        print "batch restored."

    return save_data
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
def scan_data(path, num, num_of_class):
    all_files = []
    for i in range(num_of_class):
        file_path = path + "%d/" % i
        print file_path
        files = []
        for f in os.listdir(file_path):
            path_name = os.path.join(file_path, f)
            file_name, file_extension = os.path.splitext(path_name)
            if file_extension == ".png":
                files.append(path_name)
                if len(files)>=num:
                    break
        #print len(files)
        all_files.append(files)
    
    return all_files
#
# batch > minibatch > block
# iteration*size_of_minibatch = NUM_OF_SAMPLES = 5000
#
def make_batch(num_of_class, iteration, size_of_minibatch, list_of_path):
    batch = []
    for i in range(iteration):
        minibatch = []
        for j in range(size_of_minibatch):
            block = []
            for label in range(num_of_class):
                #print "%d (%d, %d) %d = %d" % (label, i, j, size_of_minibatch, len(list_of_path[label]))
                path = list_of_path[label][size_of_minibatch*i+j]
                block.append(path)
            minibatch.append(block)
        batch.append(minibatch)

    return batch
#
#
#
def test_mode(r, batch, num_of_class, iteration, minibatch_size):
    print ">> self-test mode"

    it = 0
    dist = [0,0,0,0,0,0,0,0,0,0]
    stat = [0,0,0,0,0,0,0,0,0,0]
    
    for minibatch in batch:
        print "it : %d" % it
        if it>=iteration:
            break
        it = it + 1
        
        for i in range(len(minibatch)):
            samplese = minibatch[i]
            for label in range(len(samplese)):
                sample = samplese[label]
                data = util.loadData(sample)
                r.setInputData(data)
                r.propagate()
                inf = r.getInferences(1)
                if inf is None:
                    print "ERROR"
                    continue
                
                index = -1
                mx = max(inf)
                
                for k in range(num_of_class):
                    if inf[k] == mx:
                        index = k

                dist[index] = dist[index] + 1
            
                if label==index:
                    print "%d : %d : OK : %s" % (label, index, sample)
                    stat[index] = stat[index] + 1
                else:
                    print "%d : %d : NG : %s" % (label, index, sample)

    print dist
    for d in dist:
        print d

    print stat
    for s in stat:
        print s
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
def evaluate_minibatch(r, minibatch, num_of_class):
    sum_of_mse = 0.0
    num = 0
    ret = 0.0
    minibatch_size = len(minibatch)

    for i in range(minibatch_size):
        mb = minibatch[i]
        for j in range(num_of_class):
            num = num + 1
            labels = make_train_label(j, num_of_class)
            #print "[%d] %d : %s" % (i, j, mb[j])
            data = util.loadData(mb[j])
            r.setInputData(data)
            r.propagate()
            inf = r.getInferences(1)
            if inf is None:
                print "ERROR"
                print r
                sys.exit(0)
            
            ret = ret + inf[j]
            mse =  util.mean_absolute_error(inf, len(inf), labels, len(labels))
            sum_of_mse = sum_of_mse + mse
    
    ret = ret / num
    mse = sum_of_mse / num
    return mse, ret
#
#
#
def train_algorhythm_1(r, minibatch, num_of_class):
    connections = r.getConnections()
    print "connections=%d" % len(connections)
    
    cnt = 0
    inc = 0
    dec = 0
    noc = 0
    
    samples = random.sample(connections, 10)
    for con in samples:
        base_mse, base_ret = evaluate_minibatch(r, minibatch, num_of_class)
        #print "+ mae : %f, ret : %f" % (base_mse, base_ret)
        
        p0 = con.getWeightIndex()
        p1 = con.setWeightIndex(p0+1) # increase
        next_mse = 0.0
        next_ret = 0.0
        if p0!=p1: # OK to increase
            next_mse, next_ret = evaluate_minibatch(r, minibatch, num_of_class)
            if next_mse<base_mse: # OK
                inc = inc + 1
                pass
            else: # no effect by incfreasing, try to decrease
                p1 = con.setWeightIndex(p0-1)
                if p0==p1: # can't decrease, w is min.
                    noc = noc + 1
                #con.setWeightIndex(core.indexOfZero) # set to Zero
                else: # OK to decrease
                    next_mse, next_ret = evaluate_minibatch(r, minibatch, num_of_class)
                    if next_mse<base_mse: # effective
                        dec = dec + 1
                    else: # no effect, set to Zero
                        noc = noc + 1
                        #con.setWeightIndex(core.indexOfZero)
        else: # can't increase. w is max. try to decrease.
            p1 = con.setWeightIndex(p0-1)
            next_mse, next_ret = evaluate_minibatch(r, minibatch, num_of_class)
            if next_mse<base_mse: # decrease OK
                dec = dec + 1
            else: #set to Zero
                noc = noc + 1
                #con.setWeightIndex(core.indexOfZero)

    print "inc : %d, dec : %d, noc : %d, total : %d" % (inc, dec, noc, inc+dec+noc)
#
#
#
def train_algorhythm_2(r, minibatch, num_of_class):
    connections = r.getConnections()
    
    samples = random.sample(connections, 10)
    for con in samples:
        p0 = con.getWeightIndex()
        index = -1
        base_mse, base_ret = evaluate_minibatch(r, minibatch, num_of_class)
        for i in range(core.lesserWeightsLen):
            p1 = con.setWeightIndex(i)
            next_mse, next_ret = evaluate_minibatch(r, minibatch, num_of_class)
            if next_mse<base_mse:
                base_mse = next_mse
                index = i
        if index>-1:
            con.setWeightIndex((p0+index)/2)
            print "(%d + %d)/2 = %d" % (p0, index, (p0+index)/2)
        else:
            con.setWeightIndex(p0)
            print "-"

        
        #if p0==i:
        #    print "+ %02d | %f" % (i, base_mse)
        #else:
        #    print "- %02d | %f" % (i, base_mse)
#
#
#
def train_algorhythm_3(r, minibatch, num_of_class):
    connections = r.getConnections()
    
    base_mse, base_ret = evaluate_minibatch(r, minibatch, num_of_class)
    print "[ %f ]" % (base_mse)
    
    i = 0
    for con in connections:
        p0 = con.getWeightIndex()
        p1 = con.setWeightIndex(0)
        base_mse2, base_ret2 = evaluate_minibatch(r, minibatch, num_of_class)
        print "[%04d] %02d : %f" % (i, p0, base_mse2)
        i = i + 1
        p0 = con.setWeightIndex(p0)
#
#
#
def train_algorhythm_4(r, minibatch, num_of_class):
    connections = r.getConnections()
    print "connections=%d" % len(connections)
    
    inc_max = 10
    dec_max = 10
    cnt = 0
    inc = 0
    dec = 0
    noc = 0
    
    #base_mse, base_ret = evaluate_minibatch(r, minibatch, num_of_class)
    
    samples = random.sample(connections, len(connections))
    # dec loop
    for con in samples:
        if dec>=dec_max:
            break
        
        p0 = con.getWeightIndex()
        if p0==0:
            noc = noc + 1
            #print "    (%d)" % (noc)
            continue

        base_mse, base_ret = evaluate_minibatch(r, minibatch, num_of_class)
        p1 = con.setWeightIndex(p0-1)
        next_mse, next_ret = evaluate_minibatch(r, minibatch, num_of_class)
        if next_mse<base_mse: # OK
            dec = dec + 1
            #print "  - %d : %d > %d" % (dec, p0, p1)
        else:
            p1 = con.setWeightIndex(p0)
            noc = noc + 1
            #print "    (%d)" % (noc)
            continue

    # inc loop
    for con in samples:
        if inc>=inc_max:
            break
        
        p0 = con.getWeightIndex()
        if p0==core.lesserWeightsLen-1:
            noc = noc + 1
            #print "    (%d)" % (noc)
            continue
        
        base_mse, base_ret = evaluate_minibatch(r, minibatch, num_of_class)
        p1 = con.setWeightIndex(p0+1)
        next_mse, next_ret = evaluate_minibatch(r, minibatch, num_of_class)
        if next_mse<base_mse: # OK
            inc = inc + 1
                #print "  + %d : %d > %d" % (inc, p0, p1)
        else:
            p1 = con.setWeightIndex(p0)
            noc = noc + 1
            #print "    (%d)" % (noc)
            continue
    
    print "inc : %d, dec : %d, noc : %d, total : %d" % (inc, dec, noc, inc+dec+noc)
#
#
#
def train_algorhythm_5(r, minibatch, num_of_class):
    connections = r.getConnections()
    print "connections=%d" % len(connections)
    
    cnt = 0
    noc = 0 # no change
    inc = 0
    dec = 0
    inc_max = 100
    dec_max = 100
    inc_list = []
    dec_list = []
    samples_max = 400
    
    base_mse, base_ret = evaluate_minibatch(r, minibatch, num_of_class)
    samples = random.sample(connections, samples_max)
    
    for con in samples:
        p0 = con.getWeightIndex()
        
        p1 = con.setWeightIndex(p0+1)
        if p0==core.lesserWeightsLen-1:
            p1 = con.setWeightIndex(p0-1)
            
            next_mse, next_ret = evaluate_minibatch(r, minibatch, num_of_class)
            dec_list.append((con, next_mse))
        else:
            next_mse, next_ret = evaluate_minibatch(r, minibatch, num_of_class)
            if next_mse<base_mse:
                inc_list.append((con, next_mse))
            else:
                dec_list.append((con, next_mse))

        p0 = con.setWeightIndex(p0)

    print "inc : %d, dec : %d, noc : %d, total : %d" % (inc, dec, noc, inc+dec+noc)
#
#
#
def train_algorhythm_7(r, minibatch, num_of_class):
    connections = r.getConnections()
    print "connections=%d" % len(connections)
    
    inc_max = 100
    dec_max = 100
    cnt = 0
    inc = 0
    dec = 0
    noc = 0
    inc_list = []
    dec_list = []

    base_mse, base_ret = evaluate_minibatch(r, minibatch, num_of_class)
    samples = random.sample(connections, len(connections)/10)
    
    # dec loop
    for con in samples:
        if dec>=dec_max:
            break
        
        p0 = con.getWeightIndex()
        if p0==0:
            noc = noc + 1
            print " noc (%d)" % (p0)
            continue

        p1 = con.setWeightIndex(p0-1)
        next_mse, next_ret = evaluate_minibatch(r, minibatch, num_of_class)
        if next_mse<base_mse: # OK
            dec = dec + 1
            dec_list.append(con)
            print " - %d" % (p0-1)
        else:
            p1 = con.setWeightIndex(p0)
            noc = noc + 1
            print " noc (%d)" % (p0)

        con.setWeightIndex(p0)

    for con in dec_list:
        p0 = con.getWeightIndex()
        con.setWeightIndex(p0-1)

    base_mse, base_ret = evaluate_minibatch(r, minibatch, num_of_class)
    samples = random.sample(connections, len(connections)/10)

    # inc loop
    for con in samples:
        if inc>=inc_max:
            break
        
        p0 = con.getWeightIndex()
        if p0==core.lesserWeightsLen-1:
            noc = noc + 1
            print " noc (%d)" % (p0)
            continue

        p1 = con.setWeightIndex(p0+1)
        next_mse, next_ret = evaluate_minibatch(r, minibatch, num_of_class)
        if next_mse<base_mse: # OK
            inc = inc + 1
            inc_list.append(con)
            print " + %d" % (p0+1)
        else:
            p1 = con.setWeightIndex(p0)
            noc = noc + 1
            print " noc (%d)" % (p0)

        con.setWeightIndex(p0)

    for con in inc_list:
        p0 = con.getWeightIndex()
        con.setWeightIndex(p0+1)
    
    print "inc : %d, dec : %d, noc : %d, total : %d" % (inc, dec, noc, inc+dec+noc)
#
#
#
def train_algorhythm_6(r, minibatch, num_of_class):
    connections = r.getConnections()
    print "connections=%d" % len(connections)

    cnt = 0
    noc = 0 # no change
    inc = 0
    dec = 0
    inc_list = []
    dec_list = []
    samples_max = len(connections)/20 # 5%
    inc_max = samples_max/5 # 1%
    dec_max = samples_max/5
    
    base_mse, base_ret = evaluate_minibatch(r, minibatch, num_of_class)
    samples = random.sample(connections, samples_max)

    for con in samples:
        if inc>=inc_max and dec>=dec_max:
            break
    
        p0 = con.getWeightIndex()
        if p0==core.lesserWeightsLen-1:
            p1 = con.setWeightIndex(p0-1)
            next_mse, next_ret = evaluate_minibatch(r, minibatch, num_of_class)
            if next_mse<base_mse:
                dec = dec + 1
                base_mse = next_mse
                print " - (%d > %d)" % (p0, p0-1)
            elif next_mse==base_mse:
                con.setWeightIndex(p0)
                noc = noc + 1
            else:
                con.setWeightIndex(p0)
                noc = noc + 1
        else:
            p1 = con.setWeightIndex(p0+1)
            next_mse, next_ret = evaluate_minibatch(r, minibatch, num_of_class)
            if next_mse<base_mse:
                inc = inc + 1
                base_mse = next_mse
                print " + (%d > %d)" % (p0, p0+1)
            elif next_mse==base_mse:
                con.setWeightIndex(p0)
                noc = noc + 1
            else:
                con.setWeightIndex(p0-1)
                next_mse, next_ret = evaluate_minibatch(r, minibatch, num_of_class)
                base_mse = next_mse
                dec = dec + 1
                print " - (%d > %d)" % (p0, p0-1)

    print "inc : %d, dec : %d, noc : %d, total : %d" % (inc, dec, noc, inc+dec+noc)
#
#
#
def train_algorhythm_8(r, minibatch, num_of_class):
    connections = r.getConnections()
    print "connections=%d" % len(connections)
    
    inc_max = 100
    dec_max = 100
    cnt_max = 200
    cnt = 0
    inc = 0
    dec = 0
    noc = 0
    zero = 0
    inc_list = []
    dec_list = []
    zero_list = []
    
    
    base_mse, base_ret = evaluate_minibatch(r, minibatch, num_of_class)
    samples = random.sample(connections, len(connections)/10)
    
    # loop
    for con in samples:
        cnt = inc + dec + zero
        if cnt>=cnt_max:
            break
        
        p0 = con.getWeightIndex()
        if p0==0 or p0==core.lesserWeightsLen-1:
            p1 = con.setWeightIndex(core.WEIGHT_INDEX_ZERO)
            next_mse, next_ret = evaluate_minibatch(r, minibatch, num_of_class)
            if next_mse<base_mse: # OK, zero reset
                zero = zero + 1
                zero_list.append(con)
                print " zero reset"
            else:
                noc = noc + 1
                print " noc (%d)" % (p0)

            p1 = con.setWeightIndex(p0)
            continue

        p1 = con.setWeightIndex(core.WEIGHT_INDEX_MAX)#(p0+1)
        next_mse, next_ret = evaluate_minibatch(r, minibatch, num_of_class)
        if next_mse<base_mse: # OK, inc
            inc = inc + 1
            inc_list.append(con)
            print " + %d" % (p0+1)
        else:
            p1 = con.setWeightIndex(core.WEIGHT_INDEX_MIN)#(p0-1)
            next_mse, next_ret = evaluate_minibatch(r, minibatch, num_of_class)
            if next_mse<base_mse: # OK, dec
                dec= dec + 1
                dec_list.append(con)
                print " + %d" % (p0-1)
            else:
                p1 = con.setWeightIndex(p0)
                noc = noc + 1
                print " noc (%d)" % (p0)

        con.setWeightIndex(p0)


    for con in inc_list:
        p0 = con.getWeightIndex()
        con.setWeightIndex(p0+1)

    for con in dec_list:
        p0 = con.getWeightIndex()
        con.setWeightIndex(p0-1)
    
    for con in zero_list:
        con.setWeightIndex(core.WEIGHT_INDEX_ZERO)

    print "inc : %d, dec : %d, zero : %d, noc : %d, total : %d" % (inc, dec, zero, noc, inc+dec+noc+zero)
#
#
#
def train_algorhythm_9(r, minibatch, num_of_class):
    connections = r.getConnections()
    print "connections=%d" % len(connections)
    
    inc_max = 100 # len(connections)/100
    dec_max = 100 # len(connections)/100
    cnt = 0
    inc = 0
    dec = 0
    noc = 0
    zero = 0
    zero_2 = 0
    inc_list = []
    dec_list = []
    zero_list = []
    zero_list_2 = []
    
    base_mse, base_ret = evaluate_minibatch(r, minibatch, num_of_class)
    samples = random.sample(connections, len(connections)/10)
    
    # dec loop
    for con in samples:
        if dec>=dec_max:
            break
        
        p0 = con.getWeightIndex()
        if p0==core.WEIGHT_INDEX_MIN:
            p1 = con.setWeightIndex(core.WEIGHT_INDEX_ZERO)
            next_mse, next_ret = evaluate_minibatch(r, minibatch, num_of_class)
            if next_mse<base_mse: # OK
                zero = zero + 1
                zero_list.append(con)
                #print " 0 : zero reset at min"
            else:
                noc = noc + 1
                #print " noc at min (%d)" % (p0)
            
            con.setWeightIndex(p0)
            continue
        
        if p0>core.WEIGHT_INDEX_ZERO:
            p1 = con.setWeightIndex(core.WEIGHT_INDEX_ZERO)
        else:
            p1 = con.setWeightIndex(core.WEIGHT_INDEX_MIN)

        next_mse, next_ret = evaluate_minibatch(r, minibatch, num_of_class)
        if next_mse<base_mse: # OK
            dec = dec + 1
            dec_list.append(con)
            #print " - %d" % (p0-1)
        else:
            p1 = con.setWeightIndex(p0)
            noc = noc + 1
            #print " noc (%d)" % (p0)
        
        con.setWeightIndex(p0)
    
    for con in dec_list:
        p0 = con.getWeightIndex()
        con.setWeightIndex(p0-1)
    
    for con in zero_list:
        con.setWeightIndex(core.WEIGHT_INDEX_ZERO)

    print "dec : %d, zero : %d, noc : %d, total : %d" % (dec, zero, noc, dec+noc+zero)
    #
    base_mse, base_ret = evaluate_minibatch(r, minibatch, num_of_class)
    samples = random.sample(connections, len(connections)/10)

    # inc loop
    for con in samples:
        if inc>=inc_max:
            break
        
        p0 = con.getWeightIndex()
        if p0==core.WEIGHT_INDEX_MAX:
            p1 = con.setWeightIndex(core.WEIGHT_INDEX_ZERO)
            next_mse, next_ret = evaluate_minibatch(r, minibatch, num_of_class)
            if next_mse<base_mse: # OK
                zero_2 = zero_2 + 1
                zero_list_2.append(con)
                #print " 0 : zero reset at max"
            else:
                noc = noc + 1
                #print " noc at max (%d)" % (p0)
            
            con.setWeightIndex(p0)
            continue

        if p0<core.WEIGHT_INDEX_ZERO:
            p1 = con.setWeightIndex(core.WEIGHT_INDEX_ZERO)
        else:
            p1 = con.setWeightIndex(core.WEIGHT_INDEX_MAX)

        #p1 = con.setWeightIndex(p0+1)
        next_mse, next_ret = evaluate_minibatch(r, minibatch, num_of_class)
        if next_mse<base_mse: # OK
            inc = inc + 1
            inc_list.append(con)
            #print " + %d" % (p0+1)
        else:
            p1 = con.setWeightIndex(p0)
            noc = noc + 1
            #print " noc (%d)" % (p0)
        
        con.setWeightIndex(p0)
    
    for con in inc_list:
        p0 = con.getWeightIndex()
        con.setWeightIndex(p0+1)

    for con in zero_list_2:
        con.setWeightIndex(core.WEIGHT_INDEX_ZERO)

    print "inc : %d, zero : %d, noc : %d, total : %d" % (inc, zero_2, noc, inc+noc+zero_2)
#
#
#
def process_minibatch(r, minibatch, num_of_class):
    epoc = 1
    algo = 9
    for e in range(epoc):
        if algo == 1:
            train_algorhythm_1(r, minibatch, num_of_class)
        elif algo == 2:
            train_algorhythm_2(r, minibatch, num_of_class)
        elif algo == 3:
            train_algorhythm_3(r, minibatch, num_of_class)
        elif algo == 4:
            train_algorhythm_4(r, minibatch, num_of_class)
        elif algo == 5:
            train_algorhythm_5(r, minibatch, num_of_class)
        elif algo == 6:
            train_algorhythm_6(r, minibatch, num_of_class)
        elif algo == 7:
            train_algorhythm_7(r, minibatch, num_of_class)
        elif algo == 8:
            train_algorhythm_8(r, minibatch, num_of_class)
        elif algo == 9:
            train_algorhythm_9(r, minibatch, num_of_class)
#
#
#
def train_mode(r, train_batch, it_train, num_of_class, num_of_processed):
    print ">> train mode"
    print len(train_batch)
    print "train (%d, %d)" % (num_of_processed, it_train)

    start = num_of_processed
    k = 0
    for i in range(start, start+it_train, 1):
        minibatch = train_batch[i]
        start_time = time.time()
        process_minibatch(r, minibatch, num_of_class)
        elapsed_time = time.time() - start_time
        t = format(elapsed_time, "0")
        print "[%03d|%03d] %s" % (k, it_train, t)
        k = k + 1
    
    return k
#
#
#
def main():
    argvs = sys.argv
    argc = len(argvs)
    
    num_of_processed = 0
    num_of_iteration = 0
    
    list_of_path = []
    train_batch = []
    test_batch = []
    minibatch_size = 1
    max_it_train = NUM_OF_SAMPLES/minibatch_size
    max_it_test = NUM_OF_TEST/minibatch_size
    it_train = 0
    it_test = 0
    
    r = setup_dnn(NETWORK_PATH)
    if r is None:
        print "fatal DNN error"
        return 0
    
    save_data = setup_batch(minibatch_size, max_it_train, max_it_test)
    if save_data is None:
        print "fatal save_data error"
        return 0

    num_of_processed = save_data[0]
    train_batch = save_data[1]
    test_batch =  save_data[2]

    mode = get_key_input("0:train, 1:test, 2:self-test, 3:debug, 8:reset, 9:quit >")
    if mode==0:
        print ">> train mode : max_it_train = %d" % (num_of_processed)
        it_train = get_key_input("iteration > ")
        if it_train<0:
            print "error : iteration = %d" % it_train
            return 0
        
        prosecced = train_mode(r, train_batch, it_train, NUM_OF_CLASS, num_of_processed)
        num_of_processed = num_of_processed + prosecced
        save_data[0] = num_of_processed
        util.pickle_save(NETWORK_PATH, r)
        util.pickle_save(TRAIN_BATCH_PATH, save_data)
    elif mode==1:
        print ">> test mode"
        it_test = max_it_test
        test_mode(r, test_batch, NUM_OF_CLASS, it_test, minibatch_size)
    elif mode==2:
        print ">> self-test mode"
        test_mode(r, train_batch, NUM_OF_CLASS, num_of_processed, minibatch_size)
    elif mode==3:
        print ">> debug mode"
    elif mode==9:
        print ">> quit"
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
