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
    def __init__(self, pack, r, size):
        self._package = pack
        self._r = r
        self._batch_size = size
        #
    
    def make_w_list(self):
        self._w_list  = []
        r = self._r
        c = r.count_layers()
        for li in range(1, c):
            layer = r.get_layer_at(li)
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
        layer = r.get_layer_at(li)
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

    def make_attack_list(self, div, mode):
        w_num = len(self._w_list)
        attack_num = int(w_num/100*div) # 1% min
        if attack_num<1:
            attack_num = 0
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
        return attack_list

    def undo_attack(self, attack_list):
        r = self._r
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
        c = r.count_layers()
        for li in range(c):
            layer = r.get_layer_at(li)
            layer.update_weight()
        #
        
    def multi_attack(self, ce, mode=1, kt=0):
        r = self._r
        pack = self._package
        #
        loop_n = 20
        #w_num = self.make_w_list()
        attack_list = self.make_attack_list(kt, mode)
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
        
        c = r.count_layers()
        for li in range(c):
            layer = r.get_layer_at(li)
            layer.update_weight()
        #
        r.propagate()
        ce_alt = r.get_cross_entropy()
        ret = 0
        if ce_alt<ce:
            ce = ce_alt
            ret = 1
        else:
            self.undo_attack(attack_list)
        #
        return ce, ret

    def loop(self):
        r = self._r
        pack = self._package
        data_size = pack._image_size
        num_class = pack._num_class
        batch_size = self._batch_size
        print("batch_size=%d" % (batch_size))
        #
        pack.load_batch()
        data_array = np.zeros((self._batch_size, self._package._image_size), dtype=np.float32)
        labels = np.zeros((batch_size, num_class), dtype=np.float32)
        r.prepare(batch_size, data_size, num_class)
        #
        for j in range(batch_size):
            data_array[j] = self._package._train_image_batch[j]
            k = pack._train_label_batch[j]
            labels[j][k] = 1.0
        #
        r.set_data(data_array, data_size, labels, batch_size, 1)
        r.propagate()
        ce = r.get_cross_entropy()
        print("CE=%f" % (ce))
        #
        #
        #
        it = 50
        w_num = self.make_w_list()
        level = 0
        l_max = int(math.log2(w_num/100)) + 1
        l_cnts = [0] * l_max
        step = 1
#        kt = [1, 0.1, 0.01, 0.001, 0.01, 0.1]
        #kt = [1, 0.5, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 0.0009765625,]
        # 1*2**(-k)
        
        for j in range(it):
            div = 1.0/float(2**(level))
            cnt = 0
            for i in range(100):
                ce, ret = self.multi_attack(ce, 1, div)
                cnt = cnt + ret
                print("%d : H : %d : %f, %f" % (j, i, ce, div)
            #
            for i in range(100):
                ce, ret = self.multi_attack(ce, 0, div)
                cnt = cnt + ret
                print("%d : C : %d : %f, %f" % (j, i, ce, div)
            #
#            if step:
#                if level==l_max-1:
#                    step
            
            
            
            

#            for i in range(100):
#                ce = self.multi_attack(ce, 1, kt[k])
#                print("%d : H : %d : %f, %f" % (j, i, ce, kt[k]))
#            #
#            for i in range(100):
#                ce = self.multi_attack(ce, 0, kt[k])
#                print("%d : C : %d : %f, %f" % (j, i, ce, kt[k]))
#            #
#            if k==len(kt)-1:
#                k = 0
#            else:
#                k = k+1
#            #
            r.export_weight(pack.save_path())
        #
        return 0

