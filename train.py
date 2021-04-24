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
        #self._batch_size = size
        #
    
    def mpi_evaluate(self, com, rank, size):
        self._r.propagate()
        ce = self._r.get_cross_entropy()
        #print("[%d] ce_avg = %f" % (rank, ce))
        ce_list = com.gather(ce, root=0)
        sum = 0.0
        if rank==0:
            for i in ce_list:
                sum = sum + i
            #
            avg = sum/float(size)
        #
        if rank==0:
            ce_avg_list = [avg]*size
        else:
            ce_avg_list = None
        #
        ce_avg_list = com.scatter(ce_avg_list, root=0)
        self._ce_avg = ce_avg_list
        return self._ce_avg
        
    def make_w_list(self):
        #self._w_list  = []
        w_list  = []
        r = self._r
        c = r.count_layers()
        for li in range(1, c):
            layer = r.get_layer_at(li)
            type = layer.get_type()
            if type==core.LAYER_TYPE_HIDDEN or type==core.LAYER_TYPE_OUTPUT or type==core.LAYER_TYPE_CONV_4:
                for ni in range(layer._num_node):
                    for ii in range(layer._num_input):
                        w_list.append((li, ni, ii))
                    #
                #
            #
        #
        return w_list

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
    
    def multi_attack(self, ce, mode=1, div=0):
        r = self._r
        #pack = self._package
        #
        loop_n = 20
        #w_num = self.make_w_list()
        attack_list = self.make_attack_list(div, mode)
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
        
    def mpi_multi_attack(self, com, rank, size, ce, mode=1, div=0):
        r = self._r
        loop_n = 20
        #
        if rank==0:
            attack_list = self.make_attack_list(div, mode)
        else:
            attack_list = []
        #
        attack_list = com.bcast(attack_list, root=0)
        w_num = len(self._w_list)
        #print("rank=%d, w_num=%d" % (rank, w_num))
        #
        for wt in attack_list:
            attack_i = wt[0]
            wi = wt[1]
            wi_alt = wt[2]
            w = self._w_list[attack_i]
            li = w[0]
            ni = w[1]
            ii = w[2]
            layer = r.get_layer_at(li)
            layer.set_weight_index(ni, ii, wi_alt)
        #
        c = r.count_layers()
        for li in range(c):
            layer = r.get_layer_at(li)
            layer.update_weight()
        #
        ce_alt = self.mpi_evaluate(com, rank, size)
#        r.propagate()
#        ce_alt = r.get_cross_entropy()
        #
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
        #
        r.propagate()
        ce = r.get_cross_entropy()
        print("CE=%f" % (ce))
        #
        it = 50
        self._w_list = self.make_w_list()
        w_num = len(self._w_list)
        #
        level = 0
        l_min = 0
        l_max = int(math.log(w_num/100, 2)) + 1
        l_cnts = [1] * l_max
        mode = 1
        for j in range(it):
            div = 1.0/float(2**(level))
            cnt = 0
            for i in range(100):
                ce, ret = self.multi_attack(ce, 1, div)
                cnt = cnt + ret
                print("%d : H : %d : %f, %d (%d, %d) %d" % (j, i, ce, level, l_min, l_max, cnt))
            #
            for i in range(100):
                ce, ret = self.multi_attack(ce, 0, div)
                cnt = cnt + ret
                print("%d : C : %d : %f, %d (%d, %d) %d" % (j, i, ce, level, l_min, l_max, cnt))
            #
            l_cnts[level] = cnt
            if level == l_max-1:
                mode = -1
            elif level == l_min:
                if cnt==0:
                    if l_min==l_max-2:
                        pass
                    else:
                        l_min = l_min + 1
                    #
                #
                mode = 1
            #
            level = level + mode
            r.export_weight(pack.save_path())
        #
        return 0

    def mpi_loop(self, com, rank, size):
        r = self._r
        pack = self._package
        #
        ce = self.mpi_evaluate(com, rank, size)
        print("ce=%f" % (ce))
        it = 50
        #
        if rank==0:
            self._w_list = self.make_w_list()
        else:
            self._w_list = []
        #
        self._w_list = com.bcast(self._w_list, root=0)
        w_num = len(self._w_list)
        print("rank=%d, w_num=%d" % (rank, w_num))
        #
        level = 0
        l_min = 0
        l_max = int(math.log(w_num/100, 2)) + 1
        l_cnts = [1] * l_max
        mode = 1
        for j in range(it):
            div = 1.0/float(2**(level))
            cnt = 0
            for i in range(100):
                ce, ret = self.mpi_multi_attack(com, rank, size, ce, 1, div)
                cnt = cnt + ret
                if rank==0:
                    print("H [%d:%d] ce=%f, div=%d, Lv=%d(%d, %d) %d" % (j, i, ce, div, level, l_min, l_max, cnt))
                else:
                    pass
                #
            #
            for i in range(100):
                ce, ret = self.mpi_multi_attack(com, rank, size, ce, 0, div)
                cnt = cnt + ret
                if rank==0:
                    print("C [%d:%d] ce=%f, div=%d, Lv=%d(%d, %d) %d" % (j, i, ce, div, level, l_min, l_max, cnt))
                else:
                    pass
                #
            #
            l_cnts[level] = cnt
            if level == l_max-1:
                mode = -1
            elif level == l_min:
                if cnt==0:
                    if l_min==l_max-2:
                        pass
                    else:
                        l_min = l_min + 1
                    #
                #
                mode = 1
            #
            level = level + mode
            if rank==0:
                r.export_weight(pack.save_path())
            else:
                pass
            #
        #
        return 0
