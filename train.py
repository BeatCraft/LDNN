#! /usr/bin/python
# -*- coding: utf-8 -*-
#

#
# LDNN : lesser's Deep Neural Network
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
import cPickle
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
#
#
class Train:
    def __init__(self, package, r):
        self._package = package
        self._r = r
        #
        self._cnt_e = 0
        self._cnt_i = 0
        self._cnt_k = 0
    
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

    def weight_shift(self, li, ni, ii, entropy, zero=0):
        r = self._r
        mode = self._weight_shift_mode
        #
        layer = r.getLayerAt(li)
        wp = layer.get_weight_property(ni, ii) # default : 0
        lock = layer.get_weight_lock(ni, ii)   # default : 0
        wi = layer.get_weight_index(ni, ii)
        wi_alt = wi
        #
        if lock>0:
            return entropy, 0
        #
        if wp!=mode and wp!=0:
            return entropy, 0
        #
        if mode>0: # heat
            maximum = core.WEIGHT_INDEX_MAX
            if zero==1:
                maximum = len(core.WEIGHT_SET_CNN)-1
                if wi==maximum:
                    layer.set_weight_property(ni, ii, 0)
                    return entropy, 0
                #
            #
            if wi==maximum:
                layer.set_weight_property(ni, ii, 0)
                layer.set_weight_index(ni, ii, wi-1)
                if r._gpu:
                    layer.update_weight()
                    r.propagate()
                    entropy = r.get_cross_entropy()
                else:
                    entropy = r._remote.update(li, ni, ii, wi-1)
                #
                return entropy, 1
            #
        else: # cool
            minimum = core.WEIGHT_INDEX_MIN
            if zero==1:
                minimum = 0
                if wi==minimum:
                    layer.set_weight_property(ni, ii, 0)
                    return entropy, 0
                #
            #
            if wi==minimum:
                layer.set_weight_property(ni, ii, 0)
                layer.set_weight_index(ni, ii, wi+1)
                if r._gpu:
                    layer.update_weight()
                    r.propagate()
                    entropy = r.get_cross_entropy()
                else:
                    entropy = r._remote.update(li, ni, ii, wi+1)
                #
                return entropy, 1
            #
        #
        wi_alt = wi + mode
        entropy_alt = entropy
        if r._gpu:
            r.propagate(li, ni, ii, wi_alt, 0)
            entropy_alt = r.get_cross_entropy()
        else:
            entropy_alt = r._remote.set_alt(li, ni, ii, wi_alt)
        #
        if  entropy_alt<entropy:
            layer.set_weight_property(ni, ii, mode)
            layer.set_weight_index(ni, ii, wi_alt)
            if r._gpu:
                layer.update_weight()
            else:
                entropy_alt = r._remote.update(li, ni, ii, wi_alt)
            #
            return entropy_alt, 1
        #
        layer.set_weight_property(ni, ii, 0)
        #
        return entropy, 0

    def weight_loop(self, entropy, layer, li, ni, zero=0):
        it = 0
        r = self._r
        limit = self._limit
        #divider = self._divider
        #divider = 1
        direction = self._weight_shift_mode
        epoc = self._epoc
        cnt = 0
        num_w = layer.get_num_input()#_num_input
        #
        w_p = layer.get_num_update()#num_w
#        if num_w>divider:
#            w_p = num_w/divider
#        else:
#            w_p = num_w/4
#            if w_p<1:
#                w_p = 3
#            #
        #
        #
        #
        for p in range(w_p):
            ii = random.randrange(num_w)
            entropy, ret = self.weight_shift(li, ni, ii, entropy, zero)
            if ret>0:
                cnt = cnt + ret
                if direction>0:
                    print "+[%d|%d|%d] L=%d, N=%d, W=%d, %d/%d, %d: CE:%f" % (self._cnt_e, self._cnt_i, self._cnt_k, li, ni, ii, p, w_p, cnt, entropy)
                else:
                    print "-[%d|%d|%d] L=%d, N=%d, W=%d, %d/%d, %d: CE:%f" % (self._cnt_e, self._cnt_i, self._cnt_k, li, ni, ii, p, w_p, cnt, entropy)
                #
            #
            if entropy<limit:
                print "reach to the limit(%f), exit w loop" %(limit)
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
                print "reach to the limit(%f), exit n loop" %(limit)
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
            print entropy
        else:
            entropy = r._remote.evaluate()
        #
        c = r.countLayers()
        list_of_layer_index = []
        #
        for i in range(c):
            layer = r.getLayerAt(i)
#            layer_type = layer.get_type()
#            if layer_type==core.LAYER_TYPE_INPUT:
#                continue
#            elif layer_type==core.LAYER_TYPE_OUTPUT:
#                #continue
#                pass
#            elif layer_type==core.LAYER_TYPE_HIDDEN:
#                #continue
#                pass
#            elif layer_type==core.LAYER_TYPE_CONV:
#                continue
#            elif layer_type==core.LAYER_TYPE_POOL:
#                continue
#            elif layer_type==core.LAYER_TYPE_CONV_2D:
#                continue
#                pass
#            else:
#                continue
            #
            if layer.get_learning()>0:
                pass
            else:
                continue
            #
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
            if layer_type==core.LAYER_TYPE_CONV_2D:
                zero = 1
            #
            entropy, ret = self.node_loop(entropy, layer, li, zero)
            cnt = cnt + ret
            if entropy<limit:
                print "reach to the limit(%f), exit l loop" %(limit)
                break
            #
        # for li
        return entropy, cnt
    
    def loop(self):
        r = self._r
        package = self._package
        mini_batch_size = self._mini_batch_size
        #num = self._it
        print "it : %d" % (self._it)
        epoc = self._epoc
        limit = self._limit # 0.000001
        package.load_batch()
        batch_size = package._train_batch_size
        data_size = package._image_size
        num_class = package._num_class
        #
        r.prepare(mini_batch_size, data_size, num_class)
        print ">>mini_batch_size(%d)" % (mini_batch_size)
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
                for m in range(2): #4
                    self._cnt_k = m
                    self.set_weight_shift_mode(1)
                    entropy, h_cnt = self.layer_loop()
                    self.set_weight_shift_mode(-1)
                    entropy, c_cnt = self.layer_loop()
                    r.export_weight_index(package._wi_csv_path)
                    #
                    if entropy<limit:
                        print "reach to the limit(%f), exit iterations" %(limit)
                        return
                    #
                #
                #r.reset_weight_property()
            #
        #
        elapsed_time = time.time() - start_time
        t = format(elapsed_time, "0")
        print "time = %s" % (t)
    #
    
    def loop_alt(self):
        print "< experimental train loop >"
        r = self._r
        package = self._package
        mini_batch_size = self._mini_batch_size
        epoc = self._epoc
        limit = self._limit#0.000001
        package.load_batch()
        batch_size = package._train_batch_size
        data_size = package._image_size
        num_class = package._num_class
        it = self._it
        #
        # core training
        r.prepare(mini_batch_size, data_size, num_class)
        self.set_mini_batch(0)
        r.set_data(self._data_array, data_size, self._class_array, mini_batch_size)
        #
        # self test
#        group = []
#        r.propagate(-1, -1, -1, -1, 0)
#        answes = r.get_answer()
#        for i in range(mini_batch_size):
#            if answes[i] == self._class_array[i]:
#                pass
#            else:
#                group.append(i)
            #
        #
        group = []
        #
        for i in range(1, it):
            self.set_mini_batch(i)
            r.set_data(self._data_array, data_size, self._class_array, mini_batch_size)
            r.propagate(-1, -1, -1, -1, 0)
            answes = r.get_answer()
            for i in range(mini_batch_size):
                if answes[i] == self._class_array[i]:
                    pass
                else:
                    group.append(i)
                #
            #
        #
        print len(group)
        #
        # train extra
        esize = 400
        bsize = mini_batch_size + esize
        print bsize
        r.prepare(bsize, data_size, num_class)
        data_array = np.zeros((bsize, data_size), dtype=np.float32)
        class_array = np.zeros(bsize, dtype=np.int32)
        #
        for i in range(mini_batch_size):
            data_array[i] = package._train_image_batch[i]
            class_array[i] = package._train_label_batch[i]
            #index = group[i]
            #data_array[i] = package._train_image_batch[index]
            #class_array[i] = package._train_label_batch[index]
        #
        start = mini_batch_size
        end = bsize
        k = 0
        for i in range(start, end):
            index = group[k]
            data_array[i] = package._train_image_batch[index]
            class_array[i] = package._train_label_batch[index]
            k = k + 1
        #
        r.set_data(data_array, data_size, class_array, bsize)
        for m in range(4):
            self._cnt_k = m
            self.set_weight_shift_mode(1)
            entropy, h_cnt = self.layer_loop()
            self.set_weight_shift_mode(-1)
            entropy, c_cnt = self.layer_loop()
            r.export_weight_index(package._wi_csv_path)
            #
            if entropy<limit:
                print "reach to the limit(%f), exit iterations" %(limit)
                return
            #
        #
#
#
#



