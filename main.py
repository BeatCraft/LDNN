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

WEIGHT_INDEX_CSV_PATH   = "./wi.csv"

NUM_OF_CLASS     = 10    # 0,1,2,3,4,5,6,7,8,9
NUM_OF_SAMPLES   = 5000  # must be 5,000 per a class
NUM_OF_TEST      = 500
#
#
#
def setup_dnn(path, my_gpu):
    r = core.Roster()
    r.set_gpu(my_gpu)
        
    input_layer = r.add_layer(0, 196, 196)
    hidden_layer_1 = r.add_layer(1, 196, 32)
    hidden_layer_2 = r.add_layer(1, 32, 32)
    #
    #hidden_layer_3 = r.add_layer(1, 32, 32)
    #hidden_layer_4 = r.add_layer(1, 32, 32)
    #hidden_layer_5 = r.add_layer(1, 32, 32)
    #hidden_layer_6 = r.add_layer(1, 32, 32)
    #
    output_layer = r.add_layer(2, 32, 10)
    
    wi = util.csv_to_list(WEIGHT_INDEX_CSV_PATH)
    if len(wi)>0:
        print "restore weights"
        r.restore_weighgt(wi)
    else:
        print "init weights"
        r.init_weight()

    if my_gpu:
        r.update_gpu_weight()
    
    return r
    
#    if os.path.exists(path):
#        print "DNN restored."
#        r = util.pickle_load(path)
#        r.set_gpu(my_gpu)
#        return r
#    else:
#        r = core.Roster()
#        r.set_gpu(my_gpu)
#
#        input_layer = r.add_layer(0, 196, 196)
#        hidden_layer_1 = r.add_layer(1, 196, 32)
#        hidden_layer_2 = r.add_layer(1, 32, 32)
#        output_layer = r.add_layer(2, 32, 10)
#        r.init_weight()
#        r.update_gpu_weight()
#
        #my_gpu.set_kernel_code()
        #
        #       util.pickle_save(path, r)
        #       return r
#   return None

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
        for label in range(num_of_class):
            path = list_of_path[label][i]
            print "%d : %s" %(i, path)
            data = util.loadData(path)
            minibatch.append(data)
        
        batch.append(minibatch)

    return batch
#
#
#
def test(r, minibatch, num_of_class):
    dist = [0,0,0,0,0,0,0,0,0,0]
    stat = [0,0,0,0,0,0,0,0,0,0]

    for label in range(len(minibatch)):
        #print label
        data = minibatch[label]
        r.propagate_gpu(data)
        inf = r.get_inference_gpu()
        #print inf
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
            print "ASS HOLE"
            print mx
        
        if label==index:
            stat[index] = stat[index] + 1
         
    return [dist, stat]
#
#
#
def test_mode(r, batch, num_of_class, iteration, minibatch_size):
    print ">>test mode(%d)" % iteration
    start_time = time.time()

    it = 0
    dist = [0,0,0,0,0,0,0,0,0,0]
    stat = [0,0,0,0,0,0,0,0,0,0]
    
    for minibatch in batch:
        if it>=iteration:
            break
        print "it : %d" % it
        d, s = test(r, minibatch, num_of_class)
        #print d
        #print s
        for j in range(len(dist)):
            dist[j] = dist[j] + d[j]
            stat[j] = stat[j] + s[j]
        
        it = it + 1
    
    print dist
    for d in dist:
        print d

    print stat
    for s in stat:
        print s

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
            data = util.loadData(mb[j])
            r.propagate(data)
            #inf = r.get_inference(1)
            inf = r.get_inference_gpu()
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
def ziggle_connection(r, minibatch, num_of_class, con, dif, base_mse):
    c_id = con.get_id()
    p0 = con.get_index()
    p1 = con.set_index(p0 + dif)
    if p0==p1:
        return c_id, -1
    
    p1 = con.set_index(p0+dif)
    next_mse, next_ret = evaluate_minibatch(r, minibatch, num_of_class)
    if next_mse<base_mse: # OK
        con.set_index(p0)
        return c_id, p0+dif
    else: # noc
        con.set_index(p0)
        return c_id, -1
#
#
#
def train_algorhythm_basic(r, minibatch, num_of_class):
    connections = r.get_weight_list() #r.getConnections()
    c_num = len(connections)
    print "connections=%d" % c_num
    
    inc_max = c_num/100*2
    dec_max = c_num/100*2
    
    inc = 0
    dec = 0
    noc = 0
    
    inc_list = []
    dec_list = []
    
    base_mse, base_ret = evaluate_minibatch(r, minibatch, num_of_class)
    samples = random.sample(connections, c_num)
    
    # dec loop
    for con in samples:
        if dec>=dec_max:
            break
    
        c_id, index = ziggle_connection(r, minibatch, num_of_class, con, -1, base_mse)
        if index>=0:
            dec = dec + 1
            dec_list.append((c_id, index))
        else:
            noc = noc + 1

    for c_info in dec_list:
        con = connections[c_info[0]]
        con.set_index(c_info[1])
    
    print "dec : %d, noc : %d, total : %d" % (dec, noc, dec+noc)
    
    noc = 0
    base_mse, base_ret = evaluate_minibatch(r, minibatch, num_of_class)
    samples = random.sample(connections, c_num)
    
    # inc loop
    for con in samples:
        if inc>=inc_max:
            break
        
        c_id, index = ziggle_connection(r, minibatch, num_of_class, con, 1, base_mse)
        if index>=0:
            inc = inc + 1
            inc_list.append((c_id, index))
        else:
            noc = noc + 1

    for c_info in inc_list:
        con = connections[c_info[0]]
        con.set_index(c_info[1])
    
    print "inc : %d, noc : %d, total : %d" % (inc, noc, inc+noc)
#
#
#
def train_algorhythm_single(r, minibatch, num_of_class):
    weight_list = r.get_weight_list()
    w_num = len(weight_list)
    print "connections=%d" % w_num
    
    inc_max = w_num/100
    dec_max = w_num/100
    
    inc = 0
    dec = 0
    noc = 0
    
    base_mse, base_ret = evaluate_minibatch_gpu(r, minibatch, num_of_class)
    samples = random.sample(weight_list, w_num)
    
    # dec loop
    dif = -1
    for w in samples:
        if dec>=dec_max:
            break
        
        p0 = w.get_index()
        p1 = w.set_index(p0 + dif)
        if p0==p1:
            noc = noc + 1
            w.set_index(p0)
            continue

        next_mse, next_ret = evaluate_minibatch_gpu(r, minibatch, num_of_class)
        if next_mse<base_mse: # OK
            dec = dec + 1
            base_mse = next_mse
        else: # noc
            noc = noc + 1
            w.set_index(p0)
    
    print "dec : %d, noc : %d, total : %d" % (dec, noc, dec+noc)

    # inc loop
    base_mse, base_ret = evaluate_minibatch_gpu(r, minibatch, num_of_class)
    samples = random.sample(weight_list, w_num)
    dif = 1
    noc = 0
    for w in samples:
        if inc>=inc_max:
            break
        
        p0 = w.get_index()
        p1 = w.set_index(p0 + dif)
        if p0==p1:
            noc = noc + 1
            w.set_index(p0)
            continue
        
        next_mse, next_ret = evaluate_minibatch_gpu(r, minibatch, num_of_class)
        if next_mse<base_mse: # OK
            inc = inc + 1
            base_mse = next_mse
        else: # noc
            noc = noc + 1
            w.set_index(p0)

    print "inc : %d, noc : %d, total : %d" % (inc, noc, inc+noc)
#
#
#
def evaluate_minibatch_new(r, minibatch, num_of_class):
    sum_of_mse = 0.0
    num = 0
    ret = 0.0
    labels = [0,0,0,0,0,0,0,0,0,0]
    
    for j in range(num_of_class):
        num = num + 1
        #labels = make_train_label(j, num_of_class)
        labels[j] = 1
        data = minibatch[j] #util.loadData(mb[j])
        r.propagate(data)
        inf = r.get_inference(1)
        if inf is None:
            print "ERROR"
            print r
            sys.exit(0)
            
        ret = ret + inf[j]
        mse =  util.mean_absolute_error(inf, len(inf), labels, len(labels))
        sum_of_mse = sum_of_mse + mse
        labels[j] = 0
    
    ret = ret / num
    mse = sum_of_mse / num
    return mse, ret
#
#
#
def train_algorhythm_new(r, minibatch, num_of_class):
    weight_list = r.get_weight_list()
    w_num = len(weight_list)
    print "weights=%d" % w_num
    
    max = w_num / 100
    #max = max * 1
    noc = 0
    cnt = 0
    dec = 0
    inc = 0

    base_mse, base_ret = evaluate_minibatch_gpu(r, minibatch, num_of_class)
    samples = random.sample(weight_list, w_num/10)
    for w in samples:
        if cnt>=max:
            break
        
        p0 = w.get_index()
        p1 = p0
        if p0>core.WEIGHT_INDEX_ZERO and p0<=core.WEIGHT_INDEX_MAX:
            p1 = p1 - 1
        elif p0==core.WEIGHT_INDEX_ZERO:
            k = random.randrange(2)
            if k==0:
                p1 = p1 + 1
            else:
                p1 = p1 - 1
        elif p0<core.WEIGHT_INDEX_ZERO and p0>=core.WEIGHT_INDEX_MIN:
            p1 = p1 + 1
        
        w.set_index(p1)
        #
        r.update_gpu_weight()
        #
        next_mse, next_ret = evaluate_minibatch_gpu(r, minibatch, num_of_class)
        if next_mse<base_mse: # OK
            base_mse = next_mse
            cnt = cnt + 1
        else: # noc
            noc = noc + 1
            w.set_index(p0)
            #
            r.update_gpu_weight()
            #

    dec = cnt
    print "dec : %d, noc:%d, total : %d" % (dec, noc, dec+noc)
    
    noc = 0
    cnt = 0
    base_mse, base_ret = evaluate_minibatch_gpu(r, minibatch, num_of_class)
    samples = random.sample(weight_list, w_num/10)
    for w in samples:
        if cnt>=max:
            break
        
        p0 = w.get_index()
        p1 = p0
        if p0==core.WEIGHT_INDEX_MAX:
            p1 = p1 - 1
        elif p0>core.WEIGHT_INDEX_ZERO and p0<core.WEIGHT_INDEX_MAX:
            p1 = p1 + 1
        elif p0==core.WEIGHT_INDEX_ZERO:
            k = random.randrange(2)
            if k==0:
                p1 = p1 + 1
            else:
                p1 = p1 - 1
        elif p0<core.WEIGHT_INDEX_ZERO and p0>core.WEIGHT_INDEX_MIN:
            p1 = p1 - 1
        else: # elif p0==WEIGHT_INDEX_MIN:
            p1 = p1 + 1

        w.set_index(p1)
        #
        r.update_gpu_weight()
        #
        next_mse, next_ret = evaluate_minibatch_gpu(r, minibatch, num_of_class)
        if next_mse<base_mse: # OK
            base_mse = next_mse
            cnt = cnt + 1
        else: # noc
            noc = noc + 1
            w.set_index(p0)
            #
            r.update_gpu_weight()
            #
    inc = cnt
    print "inc : %d, noc:%d, total : %d" % (inc, noc, inc+noc)

    return dec + inc
#
#
#
def train_algorhythm_zero(r, minibatch, num_of_class):
    start_time = time.time()
    
    weight_list = r.get_weight_list()
    w_num = len(weight_list)
    max = w_num / 100
    noc = 0
    cnt = 0
    dec = 0
    inc = 0
    
    base_mse, base_ret = evaluate_minibatch_gpu(r, minibatch, num_of_class)
    samples = random.sample(weight_list, w_num)
    for w in weight_list:#samples:
        p0 = w.get_index()
        p1 = -1
        for i in range(core.lesserWeightsLen):
            w.set_index(i)
            w._layer.update_gpu_weight()
            next_mse, next_ret = evaluate_minibatch_gpu(r, minibatch, num_of_class)
            print "%f / %f" % (base_mse , next_mse)
            if next_mse<base_mse: # OK
                p1 = i
                break
        if p1<0:
            w.set_index(WEIGHT_INDEX_ZERO)
            w._layer.update_gpu_weight()
            
            
        print "%d = %d (%d)" % (cnt, i, p0)
        cnt = cnt + 1

    elapsed_time = time.time() - start_time
    t = format(elapsed_time, "0")
    print "time = %s" % (t)
#
#
#
def evaluate_gpu(r, data, data_class):
    sum_of_mse = 0.0
    num = 0
    ret = 0.0
    labels = [0,0,0,0,0,0,0,0,0,0]
    labels[data_class] = 1
    
    r.propagate_gpu(data)
    inf = r.get_inference_gpu()
    if inf is None:
        print "ERROR"
        print r
        sys.exit(0)
    
    ret = inf[data_class]
    mse =  util.mean_squared_error(inf, len(inf), labels, len(labels))
    return mse, ret
#
#
#
def evaluate_gpu_alt(r, data, data_class, w, wi_alt):
    sum_of_mse = 0.0
    num = 0
    ret = 0.0
    labels = [0,0,0,0,0,0,0,0,0,0]
    labels[data_class] = 1
    
    r.propagate_gpu_alt(data, w, wi_alt)
    inf = r.get_inference_gpu()
    #print inf
    if inf is None:
        print "ERROR"
        print r
        sys.exit(0)

    ret = inf[data_class]
    mse =  util.mean_squared_error(inf, len(inf), labels, len(labels))
    return mse, ret
#
#
#
def evaluate_minibatch_gpu(r, minibatch, num_of_class):
    sum_of_mse = 0.0
    num = float(num_of_class)
    ret = 0.0
    labels = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    
    for j in range(num_of_class):
        labels[j] = 1.0
        data = minibatch[j]
        
        r.propagate(data)
        inf = r.get_inference_gpu()
        if inf is None:
            print "ERROR"
            print r
            sys.exit(0)
        
        ret = ret + inf[j]
        mse =  util.mean_squared_error(inf, len(inf), labels, len(labels))
        sum_of_mse = sum_of_mse + mse
        labels[j] = 0.0
    
    ret = ret / num
    mse = sum_of_mse / num
    return mse, ret
#
#
#
def evaluate_minibatch_gpu_alt(r, minibatch, num_of_class, w, wi_alt):
    sum_of_mse = 0.0
    num = float(num_of_class)
    ret = 0.0
    labels = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    
    for j in range(num_of_class):
        #num = num + 1
        labels[j] = 1.0
        data = minibatch[j]
        r.propagate_gpu_alt(data, w, wi_alt)
        inf = r.get_inference_gpu()
        if inf is None:
            print "ERROR"
            print r
            sys.exit(0)
        
        ret = ret + inf[j]
        mse =  util.mean_squared_error(inf, len(inf), labels, len(labels))
        sum_of_mse = sum_of_mse + mse
        labels[j] = 0.0
    
    ret = ret / num
    mse = sum_of_mse / num
    return mse, ret
#
#
#
def train_gpu(r, data, data_class):
    weight_list = r.get_weight_list()
    w_num = len(weight_list)

    c = 0
    updates = []
    samples = random.sample(weight_list, w_num)
    for w in samples:
        base_mse, base_ret = evaluate_gpu(r, data, data_class, w, w.get_index())
        #print "%d = %d : %f" % (c, w.get_index(), base_mse)
        wi_alt = w.get_index() +1
        if wi_alt>core.WEIGHT_INDEX_MAX:
            pass
        else:
            mse, ret = evaluate_gpu_alt(r, data, data_class, w, wi_alt)
            if mse<base_mse:
                print "%d : %d = %f > %d = %f " % (w.get_id(), w.get_index(), base_mse, wi_alt, mse)
                updates.append(w.get_id())
#        for wi_alt in range(core.WEIGHT_INDEX_ZERO+2, core.lesserWeightsLen):
#           mse, ret = evaluate_gpu(r, data, data_class, w, wi_alt)
#            if mse<base_mse:
#                print "    %d : %f" % (wi_alt, mse)
        c += 1

    for i in updates:
        w = weight_list[i]
        #print "> %d = %d" % (i, w.get_index())
        w.set_index(w.get_index()+1)

    #print r.get_inference_gpu()
    r.update_gpu_weight()
#
#
#
def weight_shit(r, minibatch, num_of_class, w, base_mse, base_ret):
    inc = 0
    dec = 0
    wi = w.get_index()
    id = w.get_id()
    if wi>=core.WEIGHT_INDEX_MAX:
        #print "%d : MAX" % (id)
        #continue
        wi_alt = wi - 1
        mse_alt, ret_alt = evaluate_minibatch_gpu_alt(r, minibatch, num_of_class, w, wi_alt)
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
        mse_alt, ret_alt = evaluate_minibatch_gpu_alt(r, minibatch, num_of_class, w, wi_alt)
        if mse_alt<base_mse:
            print "     MIN(%d) : %d > %d | %f > %f" % (id, wi, wi_alt, base_mse, mse_alt)
            w.set_index(wi_alt)
            inc = inc + 1
        else:
            print "     %d : %d NOC @MIN" % (id, wi)
    else:
        wi_alt = wi + 1
        mse_alt, ret_alt = evaluate_minibatch_gpu_alt(r, minibatch, num_of_class, w, wi_alt)
        if mse_alt<base_mse:
            print "     %d : %d > %d | %f > %f" % (id, wi, wi_alt, base_mse, mse_alt)
            w.set_index(wi_alt)
            inc = inc + 1
        else:
            wi_alt = wi - 1
            mse_alt, ret_alt = evaluate_minibatch_gpu_alt(r, minibatch, num_of_class, w, wi_alt)
            if mse_alt<base_mse:
                print "     %d : %d > %d | %f > %f" % (id, wi, wi_alt, base_mse, mse_alt)
                w.set_index(wi_alt)
                dec = dec + 1
            else:
                print "     %d : %d NOC" % (id, wi)

    return inc, dec
#
#
#
def weight_scan(r, minibatch, num_of_class, w, base_mse, base_ret):
    wi = w.get_index()
    id = w.get_id()
    min = base_mse
    min_index = -1
    #min_index = core.WEIGHT_INDEX_ZERO
    
    #mse_alt, ret_alt = evaluate_minibatch_gpu_alt(r, minibatch, num_of_class, w, 0)
    min = base_mse
    for i in range(core.WEIGHT_INDEX_SIZE):
        mse_alt, ret_alt = evaluate_minibatch_gpu_alt(r, minibatch, num_of_class, w, i)
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
#def weight_shit_layer_by_layer(r, minibatch, num_of_class, w, base_mse, base_ret):
#
#
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
        base_mse, base_ret = evaluate_minibatch_gpu(r, minibatch, num_of_class)
        #
        for k in range(num):
            n = random.randint(0, layer._num_node-1)
            i = random.randint(0, layer._num_input-1)
            rwi = n*layer._num_input + i
            w = r._weight_list[wcnt + rwi]
            wi = w.get_index()
            print "%d : %d/%d (%d, %d) %d" % (index, k, num, n, i, rwi)
            #print "     %f : %f" %(core.WEIGHT_SET[wi], layer.get_weight(n, i))
            #inc, dec = weight_shit(r, minibatch, num_of_class, w, base_mse, base_ret)
            
            weight_scan(r, minibatch, num_of_class, w, base_mse, base_ret)
            
            #inc_total = inc_total + inc
            #dec_total = dec_total + dec

        # weight update
        wcnt = wcnt + layer._num_node * layer._num_input
        r.update_gpu_weight()
        # renew mse
        #base_mse, base_ret = evaluate_minibatch_gpu(r, minibatch, num_of_class)

    print "inc=%d, dec=%d" % (inc_total, dec_total)
#
#
#
def process_minibatch(r, minibatch, num_of_class):
    weight_list = r.get_weight_list()
    w_num = len(weight_list)
    base_mse, base_ret = evaluate_minibatch_gpu(r, minibatch, num_of_class)
    c = 0
    inc_total = 0
    dec_total = 0
    
    num = w_num / 10
    samples = random.sample(weight_list, num)
    for w in samples:
        print "%d / %d" % (c, num)
        inc, dec = weight_shit(r, minibatch, num_of_class, w, base_mse, base_ret)
        #weight_scan(r, minibatch, num_of_class, w, base_mse, base_ret)
        c = c + 1
        inc_total = inc_total + inc
        dec_total = dec_total + dec
    
    r.update_gpu_weight()
    print "inc=%d, dec=%d" % (inc_total, dec_total)
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
        #
        for j in range(1):
            #process_minibatch(r, minibatch, num_of_class)
            process_minibatch_layer_by_layer(r, minibatch, num_of_class)
        #
        elapsed_time = time.time() - start_time
        t = format(elapsed_time, "0")
        print "[%03d|%03d] %s" % (k, it_train, t)
        k = k + 1

    return k
#
#
#
def save_weight(r):
    wl = r.get_weight_list()
    wi_list = []
    for w in wl:
        #print w.get_index()
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

    train_batch = []
    test_batch = []
    minibatch_size = 1
    max_it_train = NUM_OF_SAMPLES/minibatch_size
    max_it_test = NUM_OF_TEST/minibatch_size
    it_train = 0
    it_test = 0
 
    #
    # GPU
    #
    my_gpu = gpu.Gpu()
    my_gpu.set_kernel_code()
    #
    #my_gpu = None
    #
    #mode = get_key_input("GPU? yes=1 >")
    #if mode==1:
    #    my_gpu = gpu.Gpu()
    #    my_gpu.set_kernel_code()
    #
    #
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
    print "5 : debug"
    print "6 : debug 2"
    print "9 : quit"
    mode = get_key_input("input command >")
    if mode==0:
        print ">> make train batch"
        train_list = scan_data(TRAIN_BASE_PATH, NUM_OF_SAMPLES, NUM_OF_CLASS)
        train_batch = make_batch(NUM_OF_CLASS, max_it_train, minibatch_size, train_list)
        train_array = np.array(train_batch)
        util.pickle_save(TRAIN_BATCH_PATH, train_array)
    elif mode==1:
        print ">> make test batch"
        test_list = scan_data(TEST_BASE_PATH, NUM_OF_TEST, NUM_OF_CLASS)
        test_batch = make_batch(NUM_OF_CLASS, max_it_test, minibatch_size, test_list)
        test_array = np.array(test_batch)
        util.pickle_save(TEST_BATCH_PATH, test_array)
    elif mode==2:
        print ">> train mode : max_it_train = %d" % (num_of_processed)
        it_train = get_key_input("iteration > ")
        if it_train<0:
            print "error : iteration = %d" % it_train
            return 0
        
        train_array = util.pickle_load(TRAIN_BATCH_PATH)
        if train_array is None:
            print "error : no train batch"
            return 0
        #
        prosecced = train_mode(r, train_array, it_train, NUM_OF_CLASS, num_of_processed)
        num_of_processed = num_of_processed + prosecced
        util.pickle_save(PROCEEDED_PATH, num_of_processed)
        # save weight here
        save_weight(r)
    elif mode==3:
        print ">> test mode"
        test_array = util.pickle_load(TEST_BATCH_PATH)
        if test_array is None:
            print "error : no test batch"
            return 0
        #
        it_test = max_it_test
        test_mode(r, test_array, NUM_OF_CLASS, it_test, minibatch_size)
    elif mode==4:
        print ">> self-test mode"
        test_mode(r, train_batch, NUM_OF_CLASS, num_of_processed, minibatch_size)
    elif mode==5:
        print ">> debug mode"
        data_list = util.loadData("./data/test/8/06755.png")
        data_array = np.array(data_list)
        #r.propagate(data_array)
        r.propagate_gpu(data_array)
        print r.get_inference_gpu()
    elif mode==6:
        wl = r.get_weight_list()
        wi_list = []
        for w in wl:
            wi_list.append( w.get_index() )

        print len(wi_list)
        util.list_to_csv("./test.cvs", wi_list)
        d_list = util.csv_to_list("./test.cvs")
        print len(d_list)
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
