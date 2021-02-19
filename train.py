#! /usr/bin/python
# -*- coding: utf-8 -*-
#

#
# LDNN : lesser's Deep Neural Network
#

import os, sys, time, math
from stat import *
import random
import copy
import math
import multiprocessing as mp
import numpy as np
import struct
import pickle

# LDNN Modules
import core
import util
import package
import gpu
#
sys.setrecursionlimit(10000)
#
#
#
class Train:
    def __init__(self, pack, r):
        self._package = pack
        self._r = r
        #
        self._cnt_e = 0
        self._cnt_i = 0
        self._cnt_k = 0
        self._weight_shift_mode = 1 # 0:cool, 1:heat
        self._loop = 1 # 1 2 4 8
        self._disable_mini_batch = 0
    
    def set_loop(self, n):
        self._loop = n
    
    def disable_mini_batch(self):
        self._disable_mini_batch = 1
    
    def reset_cnt(self):
        self._cnt_e = 0
        self._cnt_i = 0
        self._cnt_k = 0
        
    def set_mini_batch_size(self, size):
        self._mini_batch_size = size
        self._data_array = np.zeros((self._mini_batch_size, self._package._image_size), dtype=np.float32)
        self._class_array = np.zeros(self._mini_batch_size, dtype=np.int32)
    
    def set_mini_batch(self, it):
        size = self._mini_batch_size
        for i in range(size): # batch
            self._data_array[i] = self._package._train_image_batch[size*it+i]
            self._class_array[i] = self._package._train_label_batch[size*it+i]
        #
        
    def set_epoc(self, n):
        self._epoc = n

    # obsolute
    def set_weight_shift_mode(self, mode):
        self._weight_shift_mode = mode
        
    def set_layer_direction(self, d):
        self._layer_direction = d
        # 0 : input to output
        # 1 : output to input
        
    def set_limit(self, n):
        self._limit = n
    
    def set_divider(self, n):
        self._divider = n
        
    def set_iteration(self, n):
        self._it = n

    # one way
    def weight_shift(self, li, ni, ii, entropy, zero=0):
        r = self._r
        layer = r.getLayerAt(li)
        #
        lock = layer.get_weight_lock(ni, ii)   # default : 0
        if lock>0:
            return entropy, 0
        #
        wp = layer.get_weight_property(ni, ii) # default : 0
        wi = layer.get_weight_index(ni, ii)
        wi_alt = wi
        entropy_alt = entropy
        maximum = core.WEIGHT_INDEX_MAX
        minimum = core.WEIGHT_INDEX_MIN
        #
        wp_alt = wp
        if wp_alt==0:
            if wi==maximum:
                wp_alt = -1
            else:
                wp_alt = 1
            #
        else:
            if wi==maximum or wi==minimum:
                layer.set_weight_property(ni, ii, 0)
                layer.set_weight_lock(ni, ii, 1)
                print("lock at(%d)" % wi)
                return entropy, 0
            #
        #
        wi_alt = wi + wp_alt
        if r._gpu:
            r.propagate(li, ni, ii, wi_alt, 0)
            entropy_alt = r.get_cross_entropy()
        else:
            entropy_alt = r._remote.set_alt(li, ni, ii, wi_alt)
        #
        if entropy_alt<entropy:
            layer.set_weight_property(ni, ii, wp_alt)
            layer.set_weight_index(ni, ii, wi_alt)
            if r._gpu:
                layer.update_weight()
            else:
                entropy_alt = r._remote.update(li, ni, ii, wi_alt)
            #
            return entropy_alt, 1
        else:
            if wp==0:
                # reverse
                wp_alt = wp_alt*(-1)
                layer.set_weight_property(ni, ii, wp_alt)
            else:
                layer.set_weight_property(ni, ii, 0)
                layer.set_weight_lock(ni, ii, 1)
                print("lock at(%d)" % wi)
            #
        #
        return entropy, 0
        
    def weight_loop(self, entropy, layer, li, ni, zero=0):
        it = 0
        r = self._r
        limit = self._limit
        direction = self._weight_shift_mode
        epoc = self._epoc
        cnt = 0
        #
        num_w = layer.get_num_input()
        w_p = layer.get_num_update()
        #
        #
        #
        wi_list = []
        for ii in range(num_w):
            lock = layer.get_weight_lock(ni, ii)
            if lock==0:
                wi_list.append(ii)
            #
        #
        random.shuffle(wi_list)
        print("L(%d)-W(%d) : %d/%d" % (li, ni, len(wi_list), num_w))
        if w_p>len(wi_list):
            w_p = len(wi_list)
        #
        
        for p in range(w_p):
            ii = wi_list[p]
            entropy, ret = self.weight_shift(li, ni, ii, entropy, zero)
            if ret>0:
                cnt = cnt + ret
                if direction>0:
                    print("+[%d|%d|%d] L=%d, N=%d, W=%d, %d/%d, %d: CE:%f" % (self._cnt_e, self._cnt_i, self._cnt_k, li, ni, ii, p, w_p, cnt, entropy))
#                    dmsg = "+[%d|%d|%d] L=%d, N=%d, W=%d, %d/%d, %d: CE:%f" % (self._cnt_e, self._cnt_i, self._cnt_k, li, ni, ii, p, w_p, cnt, entropy)
 #                   logger.debug(dmsg)
                    
                else:
                    print("-[%d|%d|%d] L=%d, N=%d, W=%d, %d/%d, %d: CE:%f" % (self._cnt_e, self._cnt_i, self._cnt_k, li, ni, ii, p, w_p, cnt, entropy))
#                    dmsg = "-[%d|%d|%d] L=%d, N=%d, W=%d, %d/%d, %d: CE:%f" % (self._cnt_e, self._cnt_i, self._cnt_k, li, ni, ii, p, w_p, cnt, entropy)
 #                   logger.debug(dmsg)
                #
            #
            if entropy<limit:
                print("reach to the limit(%f), exit w loop" %(limit))
                break
            #
        # for p
        return entropy, cnt

    def node_loop(self, entropy, layer, li, zero=0):
        limit = self._limit
        num_node = layer.get_num_node()
        num_w = layer._num_input
        cnt = 0
        #
        node_index_list = list(range(num_node))
        random.shuffle(node_index_list)
        nc = 0
        for ni in node_index_list:
            entropy, ret = self.weight_loop(entropy, layer, li, ni, zero)
            cnt = cnt + ret
            if entropy<limit:
                print("reach to the limit(%f), exit n loop" %(limit))
                break
            #
        # for ni
        return entropy, cnt
    
    def layer_loop(self):
        r = self._r
        limit = self._limit
        reverse = self._layer_direction
        #
        entropy = 0.0
        cnt = 0
        #
        if r._gpu:
            r.propagate()
            entropy = r.get_cross_entropy()
            print(entropy)
        else:
            entropy = r._remote.evaluate()
        #
        c = r.countLayers()
        list_of_layer_index = []
        #
        for i in range(c):
            layer = r.getLayerAt(i)
#            if layer.get_learning()>0:
#                pass
#            else:
#                continue
#            #
            list_of_layer_index.append(i)
        #
        if reverse==0: # input to output
            pass
        else: # output to input
            list_of_layer_index.reverse()
        #
        for li in list_of_layer_index:
            zero = 0
            layer = r.getLayerAt(li)
            layer_type = layer.get_type()
            if layer_type==core.LAYER_TYPE_INPUT:
                continue
            elif layer_type==core.LAYER_TYPE_MAX:
                continue
            #
            entropy, ret = self.node_loop(entropy, layer, li, zero)
            cnt = cnt + ret
            if entropy<limit:
                print("reach to the limit(%f), exit l loop" %(limit))
                break
            #
        # for li
        return entropy, cnt
    
    def loop(self):
        r = self._r
        package = self._package
        mini_batch_size = self._mini_batch_size
        #num = self._it
        print("it : %d" % (self._it))
        epoc = self._epoc
        limit = self._limit # 0.000001
        package.load_batch()
        batch_size = package._train_batch_size
        data_size = package._image_size
        num_class = package._num_class
        #
        r.prepare(mini_batch_size, data_size, num_class)
        print(">>mini_batch_size(%d)" % (mini_batch_size))
        #
        start_time = time.time()
        for e in range(epoc): # epoc
            self._cnt_e = e
            for j in range(self._it): # iteration
                self._cnt_i = j
                if r._gpu:
                    self.set_mini_batch(j)
                    r.set_data(self._data_array, data_size, self._class_array, mini_batch_size)
                else:
                    r._remote.set_batch(j)
                #
                for m in range(self._loop): # 1, 2, 4, 8, 16
                    self._cnt_k = m
                    entropy, c_cnt = self.layer_loop()
                    r.export_weight(package.save_path())
                    #
                    if entropy<limit:
                        print("reach to the limit(%f), exit iterations" %(limit))
                        return
                    #
                    all = r.count_weight()
                    locked = r.count_locked_weight()
                    rate = float(locked) / float(all)
                    print("[%d|%d|%d] locked weight : %d / %d = %f" %(e, j, m, locked, all, rate))
                    #dmsg = "[%d|%d|%d] locked weight : %d / %d = %f" % (e, j, m, locked, all, rate)
                    #logger.debug(dmsg)
                    
                    if rate>0.9:
                        r.reset_weight_property()
                        r.unlock_weight_all()
                        r.reset_weight_mbid()
                        print(">>> unlock all weights")
                    #
                # for m
                if self._disable_mini_batch:
                    break
            # for j
        # for e
        elapsed_time = time.time() - start_time
        t = format(elapsed_time, "0")
        print("time = %s" % (t))
    #
    
    def get_weight_list_by_mbid(self, mbid):
        w_list = []
        r = self._r
        #
        c = r.countLayers()
        for i in range(1, c):
            layer = r.getLayerAt(i)
            for ni in range(layer._num_node):
                for ii in range(layer._num_input):
                    id = layer.get_weight_mbid(ni, ii)
                    if id==mbid:
                        w_list.append((i, ni, ii))
                    #
                #
            #
        #
        return w_list
    #
    
    def loop_alt(self):
        entropy = 1.0
        r = self._r
        package = self._package
        mini_batch_size = self._mini_batch_size
        print("it : %d" % (self._it))
        epoc = self._epoc
        limit = self._limit # 0.000001
        package.load_batch()
        batch_size = package._train_batch_size
        data_size = package._image_size
        num_class = package._num_class
        #
        r.prepare(mini_batch_size, data_size, num_class)
        print(">>mini_batch_size(%d)" % (mini_batch_size))
        #
        # set mbid here
        #
        r.assign_weight_mbid(self._it)
        #
        start_time = time.time()
        for e in range(epoc): # epoc
            self._cnt_e = e
            for j in range(self._it): # iteration / mini batch
                self._cnt_i = j
                if r._gpu:
                    self.set_mini_batch(j)
                    r.set_data(self._data_array, data_size, self._class_array, mini_batch_size)
                    r.propagate()
                    entropy = r.get_cross_entropy()
                    print(entropy)
                else:
                    r._remote.set_batch(j)
                #
                w_list = self.get_weight_list_by_mbid(j)
                print("%d : %d" % (j, len(w_list)))
                attack_num = int(len(w_list)/4) #int(len(w_list)/10)
                random.shuffle(w_list)
                for m in range(self._loop): # 1, 2, 4, 8, 16
                    for p in range(attack_num):
                        tp = w_list[p]
                        li = tp[0]
                        ni = tp[1]
                        ii = tp[2]
                        entropy, ret = self.weight_shift(li, ni, ii, entropy)
                        if ret:
                            print("[%d %d %d](%d/%d)[%d|%d|%d] %f" % (e, j, m, p, attack_num, li, ni, ii, entropy))
                        #
                    #
                    r.export_weight(package.save_path())
                    if entropy<limit:
                        print("reach to the limit(%f), exit iterations" %(limit))
                        return
                    #
                # for m
                if self._disable_mini_batch:
                    break
                #
                r.reset_weight_property()
                r.unlock_weight_all()
                #r.reset_weight_mbid()
            # for j
        # e
        elapsed_time = time.time() - start_time
        t = format(elapsed_time, "0")
        print("time = %s" % (t))
    #
    
    
    def loop_alt_2(self):
        entropy = 1.0
        r = self._r
        package = self._package
        mini_batch_size = self._mini_batch_size
        print("it : %d" % (self._it))
        epoc = self._epoc
        limit = self._limit # 0.000001
        package.load_batch()
        batch_size = package._train_batch_size
        data_size = package._image_size
        num_class = package._num_class
        #
        r.prepare(mini_batch_size, data_size, num_class)
        print(">>mini_batch_size(%d)" % (mini_batch_size))
        #
        start_time = time.time()
        for e in range(epoc): # epoc
            self._cnt_e = e
            for j in range(self._it): # iteration / mini batch
                self._cnt_i = j
                if r._gpu:
                    self.set_mini_batch(j)
                    r.set_data(self._data_array, data_size, self._class_array, mini_batch_size)
                    r.propagate()
                    entropy = r.get_cross_entropy()
                    print(entropy)
                else:
                    r._remote.set_batch(j)
                #
                w_list = self.get_weight_list_by_mbid(0)
                #
                print("%d : %d" % (j, len(w_list)))
                attack_num = int(len(w_list)/1000) #int(len(w_list)/10)
                random.shuffle(w_list)
                random.shuffle(w_list)
                for m in range(self._loop): # 1, 2, 4, 8, 16
                    for p in range(attack_num):
                        tp = w_list[p]
                        li = tp[0]
                        ni = tp[1]
                        ii = tp[2]
                        entropy, ret = self.weight_shift(li, ni, ii, entropy)
                        if ret:
                            print("[%d %d %d](%d/%d)[%d|%d|%d] %f" % (e, j, m, p, attack_num, li, ni, ii, entropy))
                        #
                    #
                    r.export_weight(package.save_path())
                    if entropy<limit:
                        print("reach to the limit(%f), exit iterations" %(limit))
                        return
                    #
                # for m
                if self._disable_mini_batch:
                    break
                #
                r.reset_weight_property()
                r.unlock_weight_all()
                #r.reset_weight_mbid()
            # for j
        # e
        elapsed_time = time.time() - start_time
        t = format(elapsed_time, "0")
        print("time = %s" % (t))
    #
#
#
#
    def loop_alt_3(self):
        entropy = 1.0
        r = self._r
        package = self._package
        mini_batch_size = self._mini_batch_size
        print("it : %d" % (self._it))
        epoc = self._epoc
        limit = self._limit # 0.000001
        package.load_batch()
        batch_size = package._train_batch_size
        data_size = package._image_size
        num_class = package._num_class
        #
        r.prepare(mini_batch_size, data_size, num_class)
        print(">>mini_batch_size(%d)" % (mini_batch_size))
        #
        
        #
        fc_layers = []
        cnn_layers = []
        c = r.count_layers()
        for i in range(c):
            layer = r.get_layer_at(i)
            type = layer.get_type()
            if type==core.LAYER_TYPE_HIDDEN or type==core.LAYER_TYPE_OUTPUT:
                fc_layers.append(i)
            elif type==core.LAYER_TYPE_CONV_4:
                cnn_layers.append(i)
            #
        #
        #print(len(fc_layers))
        #print(len(cnn_layers))
                
        start_time = time.time()
        for e in range(epoc): # epoc
            self._cnt_e = e
            for j in range(self._it): # iteration / mini batch
                self._cnt_i = j
                #
                self.set_mini_batch(j)
                r.set_data(self._data_array, data_size, self._class_array, mini_batch_size)
                r.propagate()
                entropy = r.get_cross_entropy()
                print(entropy)
                #
                w_list = []
                for i in fc_layers:
                    layer = r.getLayerAt(i)
                    for ni in range(layer._num_node):
                        for ii in range(layer._num_input):
                            w_list.append((i, ni, ii))
                        #
                    #
                #
                #print(len(w_list))
                random.shuffle(w_list)
                random.shuffle(w_list)
                attack_num = int(len(w_list)/1000)
                for m in range(self._loop):
                    for p in range(attack_num):
                        tp = w_list[p]
                        li = tp[0]
                        ni = tp[1]
                        ii = tp[2]
                        entropy, ret = self.weight_shift(li, ni, ii, entropy)
                        if ret:
                            print("[%d %d %d](%d/%d)[%d|%d|%d] %f" % (e, j, m, p, attack_num, li, ni, ii, entropy))
                        #
                    #
                    r.export_weight(package.save_path())
                    if entropy<limit:
                        print("reach to the limit(%f), exit iterations" %(limit))
                        return
                    #
                # for m
                if self._disable_mini_batch:
                    break
                #
                r.reset_weight_property()
                r.unlock_weight_all()
            # for j
            for j in range(self._it): # iteration / mini batch
                self._cnt_i = j
                #
                w_list = []
                for i in cnn_layers:
                    layer = r.getLayerAt(i)
                    for ni in range(layer._num_node):
                        for ii in range(layer._num_input):
                            w_list.append((i, ni, ii))
                        #
                    #
                #
                random.shuffle(w_list)
                random.shuffle(w_list)
                attack_num = int(len(w_list)/10)
                for m in range(self._loop):
                    for p in range(attack_num):
                        tp = w_list[p]
                        li = tp[0]
                        ni = tp[1]
                        ii = tp[2]
                        entropy, ret = self.weight_shift(li, ni, ii, entropy)
                        if ret:
                            print("[%d %d %d](%d/%d)[%d|%d|%d] %f" % (e, j, m, p, attack_num, li, ni, ii, entropy))
                        #
                    #
                    r.export_weight(package.save_path())
                    if entropy<limit:
                        print("reach to the limit(%f), exit iterations" %(limit))
                        return
                    #
                # for m
                if self._disable_mini_batch:
                    break
                #
                r.reset_weight_property()
                r.unlock_weight_all()
            # for j
        # e
        elapsed_time = time.time() - start_time
        t = format(elapsed_time, "0")
        print("time = %s" % (t))
    #
    
    def simple_weight_shift(self, i, entropy, attack_i):
        w = self._w_list[attack_i]
        li = w[0]
        ni = w[1]
        ii = w[2]
        print("(%d, %d, %d) %d" % (li, ni, ii, attack_i))
        
        r = self._r
        layer = r.getLayerAt(li)
        lock = layer.get_weight_lock(ni, ii)   # default : 0
        if lock>0:
            print("[%d] locked" % (i))
            return entropy, 0
        #
        wp = layer.get_weight_property(ni, ii) # default : 0
        wi = layer.get_weight_index(ni, ii)
        wi_alt = wi
        entropy_alt = entropy
        maximum = core.WEIGHT_INDEX_MAX
        minimum = core.WEIGHT_INDEX_MIN
        #
        wp_alt = wp
        if wp_alt==0:
            if wi==maximum:
                wp_alt = -1
            else:
                wp_alt = 1
            #
        else:
            if wi==maximum or wi==minimum:
                layer.set_weight_property(ni, ii, 0)
                layer.set_weight_lock(ni, ii, 1)
                print("[%d] lock_1(%d)" % (i, wi))
                return entropy, 0
            #
        #
        wi_alt = wi + wp_alt
        r.propagate(li, ni, ii, wi_alt, 0)
        entropy_alt = r.get_cross_entropy()
        if entropy_alt<entropy:
            layer.set_weight_property(ni, ii, wp_alt)
            layer.set_weight_index(ni, ii, wi_alt)
            layer.update_weight()
            return entropy_alt, 1
        else:
            if wp==0:
            # reverse
                wp_alt = wp_alt*(-1)
                layer.set_weight_property(ni, ii, wp_alt)
                print("[%d] reverse(%d)" % (i, wp_alt))
            else:
                layer.set_weight_property(ni, ii, 0)
                layer.set_weight_lock(ni, ii, 1)
                print("[%d] lock_2(%d)" % (i, wi))
            #
        #
        return entropy, 0
    
    def make_w_list(self):
        self._w_list  = []
        r = self._r
        c = r.countLayers()
        for li in range(1, c):
            layer = r.getLayerAt(li)
            type = layer.get_type()
            if type==core.LAYER_TYPE_HIDDEN or type==core.LAYER_TYPE_OUTPUT or type==core.LAYER_TYPE_CONV_4:
                for ni in range(layer._num_node):
                    for ii in range(layer._num_input):
                        self._w_list.append((li, ni, ii))
                    #
                #
            #
        #
        return len(self._w_list)

    def simple_loop(self):
        #entropy = 1.0
        loop_n = 1
        r = self._r
        package = self._package
        mini_batch_size = self._mini_batch_size
        print("mini_batch_size=%d" % (mini_batch_size))
        #
        package.load_batch()
        batch_size = package._train_batch_size
        data_size = package._image_size
        num_class = package._num_class
        r.prepare(mini_batch_size, data_size, num_class)
        self.set_mini_batch(0)
        r.set_data(self._data_array, data_size, self._class_array, mini_batch_size)
        #
        w_num = self.make_w_list()
        #
        attack_num = int(w_num/10*3)
        r.propagate()
        ce = r.get_cross_entropy()
        print("CE=%f" % (ce))
        print("num=%d" % (w_num))
        #
        #
        cnt = 0
        for n in range(loop_n):
            # reset
            if n>0 and n%3==0:
                r.reset_weight_property()
                r.unlock_weight_all()
                r.reset_weight_mbid()
            #
            for i in range(attack_num):
                attack_i = random.randrange(attack_num)
                ce, k = self.simple_weight_shift(i, ce, attack_i)
                cnt = cnt + k
                print("[%d][%d] %f (%d) %d" %(i, attack_i, ce, k, cnt))
            #
            r.export_weight(package.save_path())
        #
        return 0


    def loop_hb(self):
        #loop_n = 1
        r = self._r
        package = self._package
        mini_batch_size = self._mini_batch_size
        print("mini_batch_size=%d" % (mini_batch_size))
        #
        package.load_batch()
        batch_size = package._train_batch_size
        data_size = package._image_size
        num_class = package._num_class
        r.prepare(mini_batch_size, data_size, num_class)
        self.set_mini_batch(0)
        r.set_data(self._data_array, data_size, self._class_array, mini_batch_size)
        #
        r.propagate()
        ce = r.get_cross_entropy()
        print("CE = %f" % (ce))
        #
        debug = 1
        r.back_propagate(self._class_array, debug)
        #return 0
        #
        #
        #
        c = r.countLayers()
        li = c - 1
        ni = 0
        ii = 1
        output_layer = r.get_layer_at(c-1)
        hidden_1 = r.get_layer_at(c-2)
        hidden_2 = r.get_layer_at(c-3)
        rr = 0.0001
        
        for k in range(100):
            for ni in range(output_layer._num_node):
                for ii in range(output_layer._num_input):
                    wi = output_layer.get_weight_index(ni, ii)
                    #print wi
                    wi_alt = wi
                    e = output_layer.dw[ni][ii]
                    if e>0:
                        if wi==0:
                            pass
                        else:
                            wi_alt = wi - 1
                            output_layer._weight_matrix[ni][ii] = output_layer._weight_matrix[ni][ii] - rr
                            hidden_1._weight_matrix[ni][ii] = hidden_1._weight_matrix[ni][ii] - rr
                            hidden_2._weight_matrix[ni][ii] = hidden_2._weight_matrix[ni][ii] - rr
                        #
                    elif e<0:
                        if wi==core.WEIGHT_INDEX_MAX:
                            pass
                        else:
                            wi_alt = wi + 1
                            output_layer._weight_matrix[ni][ii] = output_layer._weight_matrix[ni][ii] + rr
                            hidden_1._weight_matrix[ni][ii] = hidden_1._weight_matrix[ni][ii] + rr
                            hidden_2._weight_matrix[ni][ii] = hidden_2._weight_matrix[ni][ii] + rr
                        #
                    #
                    #output_layer.set_weight_index(ni, ii, wi_alt)
                    #r.propagate(li, ni, ii, wi_alt, 0)
                    #ce_alt = r.get_cross_entropy()
                    #print("%d > %d : CE = %f, %f, %f" % (wi, wi_alt, ce_alt, ce-ce_alt, e))
                #
            #
            output_layer.update_weight()
            r.propagate()
            ce_2 = r.get_cross_entropy()
            print("[%d] CE = %f, %f" % (k, ce_2, ce-ce_2))
            r.back_propagate(self._class_array, 0)
        #
        r.export_weight(package.save_path())
        return 0
        
    def loop_hb2(self):
        r = self._r
        package = self._package
        mini_batch_size = self._mini_batch_size
        print("mini_batch_size=%d" % (mini_batch_size))
        #
        package.load_batch()
        batch_size = package._train_batch_size
        data_size = package._image_size
        num_class = package._num_class
        r.prepare(mini_batch_size, data_size, num_class)
        self.set_mini_batch(0)
        r.set_data(self._data_array, data_size, self._class_array, mini_batch_size)
        #
        r.propagate()
        ce = r.get_cross_entropy()
        print("CE = %f" % (ce))
        #
        debug = 1
        r.back_propagate(self._class_array, debug)
        #
        #
        #
        c = r.countLayers()
        li = c - 1
        ni = 1
        ii = 0



        good = 0
        #
        #
        #
        for k in range(500):
            #for li in reversed(range(c)):
            for q in range(1):
                li = random.randrange(c)
                layer = r.get_layer_at(li)
                type = layer.get_type()
                #print("%d = %d" % (li, type))
                if type==core.LAYER_TYPE_INPUT or type==core.LAYER_TYPE_CONV_4 or type==core.LAYER_TYPE_MAX:
                    continue
                #
                #print layer.dx
#                for ni in range(layer._num_node):
                for p in range(1):
                    ni = random.randrange(layer._num_node)
                    #
                    dy = layer.dy[ni]
                    #print dy
                    dx = layer.dx[ni]
#                    dw_array = layer.dw[ni] #[ii]
#                    if dy<0:
#                        ii = np.argmin(dw_array)
#                    else:
#                        ii = np.argmax(dw_array)
                    #
                    #print("dy=%f, dx=%f, dw=%f" % (dy, dx, dw_array[ii]))
                    wi = layer.get_weight_index(ni, ii)
                    wi_alt = wi
                    if dy<0:
                        if wi<core.WEIGHT_INDEX_MAX:
                            wi_alt = wi + 1
                        #
                    elif dy>0:
                        if wi>core.WEIGHT_INDEX_MIN:
                            wi_alt = wi - 1
                        #
                    #
                    layer.set_weight_index(ni, ii, wi_alt)
                #
                layer.update_weight()
            #
            r.propagate()
            ce_alt = r.get_cross_entropy()
            print("[%d] CE = %f, %f" % (k, ce_alt, ce-ce_alt))
            if ce-ce_alt<0:
                good = good + 1
            #
            ce = ce_alt
            r.back_propagate(self._class_array, 0)
        #
        print good
        r.export_weight(package.save_path())
        return


