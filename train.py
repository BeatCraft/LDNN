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
    
    def mpi_evaluate(self, mpi=0, com=None, rank=0, size=0):
        self._r.propagate()
        ce = self._r.get_cross_entropy()
        #print("[%d] ce_avg = %f" % (rank, ce))
        if mpi==0:
            return ce
        #
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

    def find_layer_by_type(self, type_list=None):
        if type_list is None:
            type_list = [core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT, core.LAYER_TYPE_CONV_4]
        #
        l_list  = []
        #
        r = self._r
        c = r.count_layers()
        for li in range(1, c):
            layer = r.get_layer_at(li)
            type = layer.get_type()
            for t in type_list:
                if type!=t:
                    continue
                else:
                    l_list.append((li))
                #
            #
        #
        return l_list
        
    def find_cnn_layer(self):
        l_list  = []
        r = self._r
        c = r.count_layers()
        for li in range(1, c):
            layer = r.get_layer_at(li)
            type = layer.get_type()
            if type==core.LAYER_TYPE_CONV_4:
                nc = layer.get_num_node()   # number of filters
                ic = layer.get_num_input()  # size of a filter
                l_list.append((li, nc, ic))
            #
        #
        return l_list
        

    def make_w_list(self, type_list=None):
        if type_list is None:
            type_list = [core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT, core.LAYER_TYPE_CONV_4]
        #
        w_list  = []
        r = self._r
        c = r.count_layers()
        for li in range(1, c):
            layer = r.get_layer_at(li)
            type = layer.get_type()
            
            for t in type_list:
                if type!=t:
                    continue
                #
                for ni in range(layer._num_node):
                    for ii in range(layer._num_input):
                        w_list.append((li, ni, ii))
                    #
                #
            #
        #
        return w_list

    def weight_ops(self, attack_i, mode, w_list):
        w = w_list[attack_i]
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

    def make_attack_cnn_list(self, w_list, attack_num):
        w_num = len(w_list)
        n_num = w_num/attack_num
        attack_list = []
        attack_i = random.randrange(n_num)
        w = w_list[attack_i]
        li = w[0]
        ni = w[1]
        #
        r = self._r
        layer = r.get_layer_at(li)
        #
        for ii in range(attack_num):
            wi = layer.get_weight_index(ni, ii)
            wi_alt = random.randrange(core.WEIGHT_INDEX_SIZE)
            attack_list.append((attack_i+i, wi, wi_alt))
        #
        return attack_list

    def make_attack_list(self, div, mode, w_list):
        w_num = len(w_list)
        attack_num = int(w_num/div)
        if attack_num<1:
            attack_num = 1
        #
        attack_list = []
        for i in range(attack_num*10):
            if i>=attack_num:
                break
            #
            attack_i = random.randrange(w_num)
            w = w_list[attack_i]
            li = w[0]
            ni = w[1]
            ii = w[2]
            wi, wi_alt = self.weight_ops(attack_i, mode, w_list)
            if wi!=wi_alt:
                attack_list.append((li, ni, ii, wi, wi_alt))
            #
        #
        return attack_list

    def undo_attack(self, attack_list):
        r = self._r
        for wt in attack_list:
            li = wt[0]
            ni = wt[1]
            ii = wt[2]
            wi = wt[3]
            wi_alt = wt[4]
            #
            layer = r.get_layer_at(li)
            layer.set_weight_index(ni, ii, wi)
        #
        c = r.count_layers()
        for li in range(c):
            layer = r.get_layer_at(li)
            layer.update_weight()
        #

    def mpi_multi_attack(self, ce, w_list, mode=1, div=0, mpi=0, com=None, rank=0, size=0):
        r = self._r
        #
        if mpi:
            if rank==0:
                attack_list = self.make_attack_list(div, mode, w_list)
            else:
                attack_list = []
            #
            attack_list = com.bcast(attack_list, root=0)
        else:
            attack_list = self.make_attack_list(div, mode, w_list)
        #
        w_num = len(w_list)
        for wt in attack_list:
            li = wt[0]
            ni = wt[1]
            ii = wt[2]
            wi = wt[3]
            wi_alt = wt[4]
            layer = r.get_layer_at(li)
            layer.set_weight_index(ni, ii, wi_alt)
        #
        c = r.count_layers()
        for li in range(c):
            layer = r.get_layer_at(li)
            layer.update_weight()
        #
        ce_alt = self.mpi_evaluate(mpi, com, rank, size)
        ret = 0
        if ce_alt<ce:
            ce = ce_alt
            ret = 1
        else:
            self.undo_attack(attack_list)
        #
        return ce, ret
    
    def mpi_w_loop(self, c, n, d, ce, w_list, lv_min, lv_max, label, mpi=0, com=None, rank=0, size=0):
        level = lv_min
        mode = 1
        r = self._r
        pack = self._package
        #
        for j in range(n):
            #
            div = float(d)*float(2**(level))
            #
            cnt = 0
            for i in range(100):
                ce, ret = self.mpi_multi_attack(ce, w_list, 1, div, mpi, com, rank, size)
                cnt = cnt + ret
                if mpi:
                    if rank==0:
                        print("[%d|%s] %d : H : %d : %f, %d (%d, %d) %d (%d, %d)" %
                            (c, label, j, i, ce, level, lv_min, lv_max, cnt, 2**(level), int(len(w_list)/div)))
                    #
                else:
                    print("[%d|%s] %d : H : %d : %f, %d (%d, %d) %d (%d, %d)" %
                        (c, label, j, i, ce, level, lv_min, lv_max, cnt, 2**(level), int(len(w_list)/div)))
                #
            #
            for i in range(100):
                ce, ret = self.mpi_multi_attack(ce, w_list, 0, div, mpi, com, rank, size)
                cnt = cnt + ret
                if mpi:
                    if rank==0:
                        print("[%d|%s] %d : C : %d : %f, %d (%d, %d) %d (%d, %d)" %
                            (c, label, j, i, ce, level, lv_min, lv_max, cnt, 2**(level), int(len(w_list)/div)))
                    #
                else:
                    print("[%d|%s] %d : C : %d : %f, %d (%d, %d) %d (%d, %d)" %
                        (c, label, j, i, ce, level, lv_min, lv_max, cnt, 2**(level), int(len(w_list)/div)))
                #
            #
            if mode==1:
                if level==lv_max:
                    if level==lv_min:
                        mode = 0
                    else:
                        mode = -1
                elif level<lv_max-1:
                    if cnt==0:
                        if level==lv_min:
                            lv_min = lv_min + 1
                        #
                    #
                #
                level = level + mode
            elif mode==-1:
                if level==lv_min:
                    if level==lv_max:
                        mode = 0
                    else:
                        mode = 1
                        if cnt==0:
                            lv_min = lv_min + 1
                        #
                    #
                #
                level = level + mode
            #
            self.mpi_save(pack.save_path(), mpi, com, rank, size)
        #
        return ce, lv_min, lv_max
    
    
    def mpi_save(self, path, mpi=0, com=None, rank=0, size=0):
        r = self._r
        if mpi:
            if rank==0:
                print("mpi_save(%s) : mpi=%d, rank=%d" % (path, mpi, rank))
                r.export_weight(path)
            else:
                pass
            #
        else:
            r.export_weight(path)
        #

    def mpi_loop(self, n=1, mpi=0, com=None, rank=0, size=0):
        w_list = None
        ce = 0.0
        ret = 0
        r = self._r
        pack = self._package
        ce = self.mpi_evaluate(mpi, com, rank, size)
        print("CE starts with %f" % ce)
        #
#        self.mpi_save(pack.save_path(), mpi, com, rank, size)
#        return 0
        
        #
        if mpi:
            if rank==0:
                w_list = self.make_w_list([core.LAYER_TYPE_CONV_4, core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT])
            else:
                w_list = []
            #
            w_list = com.bcast(w_list, root=0)
            w_num = len(w_list)
            print("rank=%d, w=%d" % (rank, w_num))
        else:
            w_list = self.make_w_list([core.LAYER_TYPE_CONV_4, core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT])
            w_num = len(w_list)
        #
        d = 100
        lv_min = 0
        lv_max = int(math.log(w_num/d, 2)) + 1
        #
        for i in range(n):
            ce, lv_min, lv_min = self.mpi_w_loop(i, 500, d, ce, w_list, lv_min, lv_max, "all", mpi, com, rank, size)
            #self.mpi_save(pack.save_path(), mpi, com, rank, size)
            #
        #
        return 0
        
    def mpi_cnn_filter_loop(self, n=1, mpi=0, com=None, rank=0, size=0):
        fc_w_list = None
        cnn_w_list = None
        ce = 0.0
        ret = 0
        r = self._r
        pack = self._package
        ce = self.mpi_evaluate(mpi, com, rank, size)
        #
        if mpi:
            if rank==0:
                cnn_layers = self.find_cnn_layer()
                cnn_l_num = len(cnn_layers)
                print(cnn_layers)
            else:
                pass
            #
        else:
            cnn_layers = self.find_cnn_layer()
            cnn_l_num = len(cnn_layers)
            print(cnn_layers)
            k = random.randrange(cnn_l_num) # for multi cnn
            li = cnn_layers[k][0]
            nc = cnn_layers[k][1] # num of filters
            ic = cnn_layers[k][2] # size of a filter
            layer = r.get_layer_at(li)
            ni = random.randrange(nc)
            print("li=%d, ni=%d" % (li, ni))
        #
        
        for i in range(n):
            if mpi:
                if rank==0:
                    li = random.randrange(cnn_l_num) # for multi cnn
                    layer = r.get_layer_at(li)
                    nc = cnn_layers[li][1] # num of filters
                    ic = cnn_layers[li][2] # size of a filter
                    ni = random.randrange(nc)
                    attack_list = []
                    for ii in range(ic):
                        wi = layer.get_weight_index(ni, ii)
                        wi_alt = random.randrange(core.WEIGHT_INDEX_SIZE)
                        attack_list.append((li, ni, ii, wi, wi_alt))
                    #
                else:
                    attack_list = []
                #
                attack_list = com.bcast(attack_list, root=0)
                for wt in attack_list:
                    li = wt[0]
                    ni = wt[1]
                    ii = wt[2]
                    wi = wt[3]
                    wi_alt = wt[4]
                    layer.set_weight_index(ni, ii, wi_alt)
                #
                c = r.count_layers()
                for li in range(c):
                    layer = r.get_layer_at(li)
                    layer.update_weight()
                #
                ce_alt = self.mpi_evaluate(mpi, com, rank, size)
                if ce_alt<ce:
                    ce = ce_alt
                else:
                    self.undo_attack(attack_list)
                #
            else:
#                li = random.randrange(cnn_l_num) # for multi cnn
#                layer = r.get_layer_at(li)
#                nc = cnn_layers[li][1] # num of filters
#                ic = cnn_layers[li][2] # size of a filter
#                ni = random.randrange(nc)
                attack_list = []
                for ii in range(ic):
                    wi = layer.get_weight_index(ni, ii)
                    wi_alt = random.randrange(core.WEIGHT_INDEX_SIZE)
                    attack_list.append((li, ni, ii, wi, wi_alt))
                    layer.set_weight_index(ni, ii, wi_alt)
                #
                #
                layer.update_weight()
                ce_alt = self.mpi_evaluate(mpi, com, rank, size)
                print("%d : %f > %f" % (i, ce, ce_alt))
                if ce_alt<ce:
                    ce = ce_alt
                else:
                    self.undo_attack(attack_list)
                #
            #
        # for
        self.mpi_save(pack.save_path(), mpi, com, rank, size)
        return 0
        
    def mpi_cnn_loop(self, n=1, mpi=0, com=None, rank=0, size=0):
        fc_w_list = None
        cnn_w_list = None
        ce = 0.0
        ret = 0
        r = self._r
        pack = self._package
        ce = self.mpi_evaluate(mpi, com, rank, size)
        #
        if mpi:
            if rank==0:
                fc_w_list = self.make_w_list([core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT])
                cnn_w_list = self.make_w_list([core.LAYER_TYPE_CONV_4])
            else:
                fc_w_list = []
                cnn_w_list = []
            #
            fc_w_list = com.bcast(fc_w_list, root=0)
            cnn_w_list = com.bcast(cnn_w_list, root=0)
            fc_w_num = len(fc_w_list)
            cnn_w_num = len(cnn_w_list)
            print("rank=%d, fc=%d, cnn=%d" % (rank, fc_w_num, cnn_w_num))
        else:
            fc_w_list = self.make_w_list([core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT])
            fc_w_num = len(fc_w_list)
            cnn_w_list = self.make_w_list([core.LAYER_TYPE_CONV_4])
            cnn_w_num = len(cnn_w_list)
            print("fc=%d, cnn=%d" % (fc_w_num, cnn_w_num))
        #
        fc_d = 100
        fc_lv_min = 0
        fc_lv_max = int(math.log(fc_w_num/fc_d, 2)) + 1
        cnn_d = 1
        cnn_lv_min = 0
        cnn_lv_max = int(math.log(cnn_w_num/cnn_d, 2)) + 1
        #
        for i in range(n):
            ce, fc_lv_min, fc_lv_min = self.mpi_w_loop(i, 500, fc_d, ce, fc_w_list, fc_lv_min, fc_lv_max, "fc", mpi, com, rank, size)
            self.mpi_save(pack.save_path(), mpi, com, rank, size)
            #
            ce, cnn_lv_min, cnn_lv_max = self.mpi_w_loop(i, 100, cnn_d, ce, cnn_w_list, cnn_lv_min, cnn_lv_max, "cnn", mpi, com, rank, size)
            self.mpi_save(pack.save_path(), mpi, com, rank, size)
        #
        return 0
