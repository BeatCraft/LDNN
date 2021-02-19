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
    
    def weight_shift(self, i, entropy, attack_i):
        w = self._w_list[attack_i]
        li = w[0]
        ni = w[1]
        ii = w[2]
        #print("(%d, %d, %d) %d" % (li, ni, ii, attack_i))
        
        r = self._r
        layer = r.getLayerAt(li)
        lock = layer.get_weight_lock(ni, ii)   # default : 0
        if lock>0:
            print("    [%d] locked" % (i))
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
                print("    [%d] lock_1(%d)" % (i, wi))
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
                print("    [%d] reverse(%d)" % (i, wp_alt))
            else:
                layer.set_weight_property(ni, ii, 0)
                layer.set_weight_lock(ni, ii, 1)
                print("    [%d] lock_2(%d)" % (i, wi))
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

    def weight_ops(self, attack_i, mode):
        w = self._w_list[attack_i]
        li = w[0]
        ni = w[1]
        ii = w[2]
        #
        r = self._r
        layer = r.getLayerAt(li)
        wi = layer.get_weight_index(ni, ii)
        wi_alt = wi
        maximum = core.WEIGHT_INDEX_MAX
        minimum = core.WEIGHT_INDEX_MIN
        #
        if mode>0: # heat
            if wi<maximum:
                wi_alt = wi + 1
                
            #
        else:
            if wi>minimum:
                wi_alt = wi - 1
            #
        #
        return wi, wi_alt


    def multi_attack(self, ce, mode=1):
        r = self._r
        pack = self._package
        #
        loop_n = 20
        w_num = self.make_w_list()
        attack_num = w_num/1000 # 0.1%
        #
        attack_list = []
        for i in range(attack_num*10):
            if i>=attack_num:
                break
            #
            attack_i = random.randrange(w_num)
            w = self._w_list[attack_i]
            li = w[0]
            ni = w[1]
            ii = w[2]
            wi, wi_alt = self.weight_ops(attack_i, mode)
            if wi!=wi_alt:
                attack_list.append((attack_i, wi, wi_alt))
            #
        #
        for wt in attack_list:
            attack_i = wt[0]
            wi = wt[1]
            wi_alt = wt[2]
            w = self._w_list[attack_i]
            li = w[0]
            ni = w[1]
            ii = w[2]
            #
            layer = r.get_layer_at(li)
            layer.set_weight_index(ni, ii, wi_alt)
        #
        
        c =r.countLayers()
        for li in range(c):
            layer = r.get_layer_at(li)
            layer.update_weight()
        #
        r.propagate()
        ce_alt = r.get_cross_entropy()
        if ce_alt<ce:
            ce = ce_alt
        else:
            for wt in attack_list:
                attack_i = wt[0]
                wi = wt[1]
                wi_alt = wt[2]
                w = self._w_list[attack_i]
                li = w[0]
                ni = w[1]
                ii = w[2]
                #
                layer = r.get_layer_at(li)
                layer.set_weight_index(ni, ii, wi)
            #
            for li in range(c):
                layer = r.get_layer_at(li)
                layer.update_weight()
            #
        #
        return ce
    
    def single_attack(self, ce):
        r = self._r
        pack = self._package
        #
        loop_n = 20
        w_num = self.make_w_list()
        attack_num = int(w_num/10*3)
        print("w : %d / %d" % (attack_num, w_num))
        cnt = 0
        for n in range(loop_n):
            # reset
            if n>0 and n%5==0:
                r.reset_weight_property()
                r.unlock_weight_all()
                r.reset_weight_mbid()
            #
            for i in range(attack_num):
                attack_i = random.randrange(attack_num)
                ce_alt, k = self.weight_shift(i, ce, attack_i)
                cnt = cnt + k
                if k>0:
                    print("o : %d, %d, %d : %f (%f) (%d)" %(n, i, attack_i, ce, ce-ce_alt, cnt))
                else:
                    print("x : %d, %d, %d : %f (%f) (%d)" %(n, i, attack_i, ce, ce-ce_alt, cnt))
                #
                ce = ce_alt
                
            #
            r.export_weight(pack.save_path())
        #

    def loop(self):
        r = self._r
        pack = self._package
        mini_batch_size = self._mini_batch_size
        print("mini_batch_size=%d" % (mini_batch_size))
        #
        pack.load_batch()
        batch_size = pack._train_batch_size
        data_size = pack._image_size
        num_class = pack._num_class
        r.prepare(mini_batch_size, data_size, num_class)
        self.set_mini_batch(0)
        r.set_data(self._data_array, data_size, self._class_array, mini_batch_size)
        r.propagate()
        ce = r.get_cross_entropy()
        print("CE=%f" % (ce))
        #
        #self.single_attack(ce)
        for j in range(10):
            for i in range(500):
                ce = self.multi_attack(ce, 1)
                print("%d : H : %d : %f" % (j, i, ce))
            #
            for i in range(500):
                ce = self.multi_attack(ce, 0)
                print("%d : C : %d : %f" % (j, i, ce))
            #
            r.export_weight(pack.save_path())
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


