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
import pickle
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
        self._weight_shift_mode = 1 # 0:cool, 1:heat
        self._loop = 4 # 1 2 4 8
    
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

    # one way
    def weight_shift_5(self, li, ni, ii, entropy, zero=0):
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
        

    def weight_shift_4(self, li, ni, ii, entropy, zero=0):
        r = self._r
        layer = r.getLayerAt(li)
        wp = layer.get_weight_property(ni, ii) # default : 0
        lock = layer.get_weight_lock(ni, ii)   # default : 0
        wi = layer.get_weight_index(ni, ii)
        wi_alt = wi
        entropy_alt = entropy
        maximum = core.WEIGHT_INDEX_MAX
        minimum = core.WEIGHT_INDEX_MIN
        #
        if lock>0:
            return entropy, 0
        #
        if wp==0:
            if wi==maximum:
                wi_alt = wi - 1
                if r._gpu:
                    r.propagate(li, ni, ii, wi_alt, 0)
                    entropy_alt = r.get_cross_entropy()
                else:
                    entropy_alt = r._remote.set_alt(li, ni, ii, wi_alt)
                #
                if entropy_alt<entropy:
                    layer.set_weight_property(ni, ii, -1)
                    layer.set_weight_index(ni, ii, wi_alt)
                    if r._gpu:
                        layer.update_weight()
                    else:
                        entropy_alt = r._remote.update(li, ni, ii, wi_alt)
                    #
                    return entropy_alt, 1
                #
                #layer.set_weight_lock(ni, ii, 1)
                #layer.set_weight_property(ni, ii, 0)
                return entropy_alt, 0
            elif wi==minimum:
                wi_alt = wi + 1
                if r._gpu:
                    r.propagate(li, ni, ii, wi_alt, 0)
                    entropy_alt = r.get_cross_entropy()
                else:
                    entropy_alt = r._remote.set_alt(li, ni, ii, wi_alt)
                #
                if entropy_alt<entropy:
                    layer.set_weight_property(ni, ii, 1)
                    layer.set_weight_index(ni, ii, wi_alt)
                    if r._gpu:
                        layer.update_weight()
                    else:
                        entropy_alt = r._remote.update(li, ni, ii, wi_alt)
                    #
                    return entropy_alt, 1
                #
                #layer.set_weight_lock(ni, ii, 1)
                return entropy_alt, 0
            else: # minimum < wi < maximum
                wi_alt = wi + 1
                if r._gpu:
                    r.propagate(li, ni, ii, wi_alt, 0)
                    entropy_alt = r.get_cross_entropy()
                else:
                    entropy_alt = r._remote.set_alt(li, ni, ii, wi_alt)
                #
                if entropy_alt<entropy:
                    layer.set_weight_property(ni, ii, 1)
                    layer.set_weight_index(ni, ii, wi_alt)
                    if r._gpu:
                        layer.update_weight()
                    else:
                        entropy_alt = r._remote.update(li, ni, ii, wi_alt)
                    #
                    #if wi_alt==maximum:
                    #    layer.set_weight_lock(ni, ii, 1)
                    #
                    return entropy_alt, 1
                #
                wi_alt = wi - 1 ##
                if r._gpu:
                    r.propagate(li, ni, ii, wi_alt, 0)
                    entropy_alt = r.get_cross_entropy()
                else:
                    entropy_alt = r._remote.set_alt(li, ni, ii, wi_alt)
                #
                if entropy_alt<entropy:
                    layer.set_weight_property(ni, ii, -1)
                    layer.set_weight_index(ni, ii, wi_alt)
                    if r._gpu:
                        layer.update_weight()
                    else:
                        entropy_alt = r._remote.update(li, ni, ii, wi_alt)
                    #
                    #if wi_alt==minimum:
                    #    layer.set_weight_lock(ni, ii, 1)
                    #
                    return entropy_alt, 1
                #
                layer.set_weight_lock(ni, ii, 1)
                return entropy_alt, 0
            #
        elif wp>0:
            if wi==maximum:
                layer.set_weight_property(ni, ii, 0)
                return entropy_alt, 0
            #
            wi_alt = wi + 1
            if r._gpu:
                r.propagate(li, ni, ii, wi_alt, 0)
                entropy_alt = r.get_cross_entropy()
            else:
                entropy_alt = r._remote.set_alt(li, ni, ii, wi_alt)
            #
            if entropy_alt<entropy:
                layer.set_weight_property(ni, ii, 1)
                layer.set_weight_index(ni, ii, wi_alt)
                if r._gpu:
                    layer.update_weight()
                else:
                    entropy_alt = r._remote.update(li, ni, ii, wi_alt)
                #
                #if wi_alt==maximum:
                #    layer.set_weight_lock(ni, ii, 1)
                #
                return entropy_alt, 1
            #
            layer.set_weight_lock(ni, ii, 1)
            layer.set_weight_property(ni, ii, 0)
            return entropy_alt, 0
        else: #  wp<0
            if wi==minimum:
                layer.set_weight_property(ni, ii, 0)
                return entropy_alt, 0
            #
            wi_alt = wi - 1
            if r._gpu:
                r.propagate(li, ni, ii, wi_alt, 0)
                entropy_alt = r.get_cross_entropy()
            else:
                entropy_alt = r._remote.set_alt(li, ni, ii, wi_alt)
            #
            if entropy_alt<entropy:
                layer.set_weight_property(ni, ii, -1)
                layer.set_weight_index(ni, ii, wi_alt)
                if r._gpu:
                    layer.update_weight()
                else:
                    entropy_alt = r._remote.update(li, ni, ii, wi_alt)
                #
                #if wi_alt==minimum:
                #    layer.set_weight_lock(ni, ii, 1)
                #
                return entropy_alt, 1
            #
            layer.set_weight_lock(ni, ii, 1)
            layer.set_weight_property(ni, ii, 0)
            return entropy_alt, 0
        #
        return entropy, 0
    
    def weight_shift_3(self, li, ni, ii, entropy, zero=0):
        r = self._r
        layer = r.getLayerAt(li)
        wp = layer.get_weight_property(ni, ii) # default : 0
        lock = layer.get_weight_lock(ni, ii)   # default : 0
        wi = layer.get_weight_index(ni, ii)
        wi_alt = wi
        entropy_alt = entropy
        maximum = core.WEIGHT_INDEX_MAX
        minimum = core.WEIGHT_INDEX_MIN
        #
        if lock>0:
            return entropy, 0
        #
        if wp==0:
            if wi==maximum:
                wi_alt = wi - 1
                if r._gpu:
                    r.propagate(li, ni, ii, wi_alt, 0)
                    entropy_alt = r.get_cross_entropy()
                else:
                    entropy_alt = r._remote.set_alt(li, ni, ii, wi_alt)
                #
                if entropy_alt<entropy:
                    layer.set_weight_property(ni, ii, -1)
                    layer.set_weight_index(ni, ii, wi_alt)
                    if r._gpu:
                        layer.update_weight()
                    else:
                        entropy_alt = r._remote.update(li, ni, ii, wi_alt)
                    #
                    return entropy_alt, 1
                #
                layer.set_weight_lock(ni, ii, 1)
                return entropy_alt, 0
            elif wi==minimum:
                wi_alt = wi + 1
                if r._gpu:
                    r.propagate(li, ni, ii, wi_alt, 0)
                    entropy_alt = r.get_cross_entropy()
                else:
                    entropy_alt = r._remote.set_alt(li, ni, ii, wi_alt)
                #
                if entropy_alt<entropy:
                    layer.set_weight_property(ni, ii, 1)
                    layer.set_weight_index(ni, ii, wi_alt)
                    if r._gpu:
                        layer.update_weight()
                    else:
                        entropy_alt = r._remote.update(li, ni, ii, wi_alt)
                    #
                    return entropy_alt, 1
                #
                layer.set_weight_lock(ni, ii, 1)
                return entropy_alt, 0
            else:
                wi_alt = wi + 1
                if r._gpu:
                    r.propagate(li, ni, ii, wi_alt, 0)
                    entropy_alt = r.get_cross_entropy()
                else:
                    entropy_alt = r._remote.set_alt(li, ni, ii, wi_alt)
                #
                if entropy_alt<entropy:
                    layer.set_weight_property(ni, ii, 1)
                    layer.set_weight_index(ni, ii, wi_alt)
                    if r._gpu:
                        layer.update_weight()
                    else:
                        entropy_alt = r._remote.update(li, ni, ii, wi_alt)
                    #
                    if wi_alt==maximum:
                        layer.set_weight_lock(ni, ii, 1)
                    #
                    return entropy_alt, 1
                #
                wi_alt = wi - 1 ##
                if r._gpu:
                    r.propagate(li, ni, ii, wi_alt, 0)
                    entropy_alt = r.get_cross_entropy()
                else:
                    entropy_alt = r._remote.set_alt(li, ni, ii, wi_alt)
                #
                if entropy_alt<entropy:
                    layer.set_weight_property(ni, ii, -1)
                    layer.set_weight_index(ni, ii, wi_alt)
                    if r._gpu:
                        layer.update_weight()
                    else:
                        entropy_alt = r._remote.update(li, ni, ii, wi_alt)
                    #
                    if wi_alt==minimum:
                        layer.set_weight_lock(ni, ii, 1)
                    #
                    return entropy_alt, 1
                #
                layer.set_weight_lock(ni, ii, 1)
                return entropy_alt, 0
            #
        elif wp>0:
            wi_alt = wi + 1
            if r._gpu:
                r.propagate(li, ni, ii, wi_alt, 0)
                entropy_alt = r.get_cross_entropy()
            else:
                entropy_alt = r._remote.set_alt(li, ni, ii, wi_alt)
            #
            if entropy_alt<entropy:
                layer.set_weight_property(ni, ii, 1)
                layer.set_weight_index(ni, ii, wi_alt)
                if r._gpu:
                    layer.update_weight()
                else:
                    entropy_alt = r._remote.update(li, ni, ii, wi_alt)
                #
                if wi_alt==maximum:
                    layer.set_weight_lock(ni, ii, 1)
                #
                return entropy_alt, 1
            #
            layer.set_weight_lock(ni, ii, 1)
            return entropy_alt, 0
        else:
            wi_alt = wi - 1
            if r._gpu:
                r.propagate(li, ni, ii, wi_alt, 0)
                entropy_alt = r.get_cross_entropy()
            else:
                entropy_alt = r._remote.set_alt(li, ni, ii, wi_alt)
            #
            if entropy_alt<entropy:
                layer.set_weight_property(ni, ii, -1)
                layer.set_weight_index(ni, ii, wi_alt)
                if r._gpu:
                    layer.update_weight()
                else:
                    entropy_alt = r._remote.update(li, ni, ii, wi_alt)
                #
                if wi_alt==minimum:
                    layer.set_weight_lock(ni, ii, 1)
                #
                return entropy_alt, 1
            #
            layer.set_weight_lock(ni, ii, 1)
            return entropy_alt, 0
        #
        return entropy, 0

    def weight_shift_2(self, li, ni, ii, entropy, zero=0):
        r = self._r
        mode = self._weight_shift_mode
        #
        layer = r.getLayerAt(li)
        wp = layer.get_weight_property(ni, ii) # default : 0
        lock = layer.get_weight_lock(ni, ii)   # default : 0
        wi = layer.get_weight_index(ni, ii)
        wi_alt = wi
        entropy_alt = entropy
        maximum = core.WEIGHT_INDEX_MAX
        minimum = core.WEIGHT_INDEX_MIN
        #
        if lock>0:
            return entropy, 0
        #
        if wp<0:
            pass
        elif wi==maximum:
            layer.set_weight_property(ni, ii, 0)
            return entropy_alt, 0
#            wi_alt = wi - 1
#            if r._gpu:
#                layer.set_weight_property(ni, ii, 0)
#                layer.set_weight_index(ni, ii, wi_alt)
#                layer.update_weight()
#                r.propagate()
#                entropy_alt = r.get_cross_entropy()
#            else:
#                entropy_alt = r._remote.update(li, ni, ii, wi_alt)
#            #
#            return entropy_alt, 1 # !!
        else:
            wi_alt = wi + 1
            if r._gpu:
                r.propagate(li, ni, ii, wi_alt, 0)
                entropy_alt = r.get_cross_entropy()
            else:
                entropy_alt = r._remote.set_alt(li, ni, ii, wi_alt)
            #
            if entropy>entropy_alt: # better
                layer.set_weight_property(ni, ii, 1)
                layer.set_weight_index(ni, ii, wi_alt)
                if r._gpu:
                    layer.update_weight()
                else:
                    entropy_alt = r._remote.update(li, ni, ii, wi_alt)
                #
                return entropy_alt, 1
            else:
                layer.set_weight_property(ni, ii, 0)
            #
        #
        if wp>0:
            pass
        elif wi==minimum:
            layer.set_weight_property(ni, ii, 0)
            return entropy_alt, 0
#            wi_alt = wi - 1
#            if r._gpu:
#                layer.set_weight_property(ni, ii, 0)
#                layer.set_weight_index(ni, ii, wi_alt)
#                layer.update_weight()
#                r.propagate()
#                entropy_alt = r.get_cross_entropy()
#            else:
#                entropy_alt = r._remote.update(li, ni, ii, wi_alt)
            #
#            return entropy_alt, 1 # !!
        else:
            wi_alt = wi - 1
            if r._gpu:
                r.propagate(li, ni, ii, wi_alt, 0)
                entropy_alt = r.get_cross_entropy()
            else:
                entropy_alt = r._remote.set_alt(li, ni, ii, wi_alt)
            #
            if entropy>entropy_alt: # better
                layer.set_weight_property(ni, ii, -1)
                layer.set_weight_index(ni, ii, wi_alt)
                if r._gpu:
                    layer.update_weight()
                else:
                    entropy_alt = r._remote.update(li, ni, ii, wi_alt)
                #
                return entropy_alt, 1
            else:
                layer.set_weight_property(ni, ii, 0)
            #
        #
        return entropy_alt, 0
        
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
        direction = self._weight_shift_mode
        epoc = self._epoc
        cnt = 0
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
        random.shuffle(wi_list)
        print("w : %d" % (len(wi_list)))
        if w_p>len(wi_list):
            w_p = len(wi_list)
        #
        #
        #
        for p in range(w_p):
            #ii = random.randrange(num_w)
            ii = wi_list[p]
            entropy, ret = self.weight_shift_5(li, ni, ii, entropy, zero)
            if ret>0:
                cnt = cnt + ret
                if direction>0:
                    print("+[%d|%d|%d] L=%d, N=%d, W=%d, %d/%d, %d: CE:%f" % (self._cnt_e, self._cnt_i, self._cnt_k, li, ni, ii, p, w_p, cnt, entropy))
                else:
                    print("-[%d|%d|%d] L=%d, N=%d, W=%d, %d/%d, %d: CE:%f" % (self._cnt_e, self._cnt_i, self._cnt_k, li, ni, ii, p, w_p, cnt, entropy))
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
            if layer_type==core.LAYER_TYPE_INPUT:
                continue
            elif layer_type==core.LAYER_TYPE_POOL:
                continue
            #
#            if layer_type==core.LAYER_TYPE_CONV_2D:
#                zero = 1
#            #
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
                for m in range(self._loop): #1, 2, 4, 8, 16
                    self._cnt_k = m
                    #self.set_weight_shift_mode(1)
                    entropy, c_cnt = self.layer_loop()
                    #
#                    self.set_weight_shift_mode(1)
#                    entropy, h_cnt = self.layer_loop()
#                    self.set_weight_shift_mode(-1)
#                    entropy, c_cnt = self.layer_loop()
                    r.export_weight_index(package._wi_csv_path)
                    #
                    if entropy<limit:
                        print("reach to the limit(%f), exit iterations" %(limit))
                        return
                    #
                    all = r.count_weight()
                    locked = r.count_locked_weight()
                    rate = float(locked) / float(all)
                    print("locked weight : %d / %d = %f" %(locked, all, rate))
                    if rate>0.9:
                        r.reset_weight_property()
                        r.unlock_weight_all()
                        print(">>> unlock all weights")
                    #
                    # this won't work. better use sum of updated weights
                    #
                #
            #
        #
        elapsed_time = time.time() - start_time
        t = format(elapsed_time, "0")
        print("time = %s" % (t))
    #
#
#
#



