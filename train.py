#! /usr/bin/python
# -*- coding: utf-8 -*-
#

#
# LDNN : lesser's Deep Neural Network
#

import os, sys, time, math
import random
import numpy as np
from PIL import Image

#
# LDNN Modules
#
import util
import core
import gpu
#
sys.setrecursionlimit(10000)
#
#
#
class Train:
    def __init__(self, r, com=None, rank=-1, size=-1):
        self._r = r
        self.mode = 0
        #self.mode_w = 0
        self.mode_e = 0
        self.mse_idex = -1
        
        if com:
            if rank<0 or size<0:
                return
            #
            self.mpi = True
            self.com = com
            self.rank = rank
            self.size = size
        #
        
    def set_batch(self, data_size, num_class, train_data_batch, train_label_batch,  batch_size, batch_offset):
        self._batch_size = batch_size
        self._data_size = data_size
        #
        r = self._r
        r.prepare(batch_size, data_size, num_class)
        r.set_batch(data_size, num_class, train_data_batch, train_label_batch,  batch_size, batch_offset)
    
    def set_path(path):
        self._path = path
    
    def evaluate(self):#, mode=0, idx=0):
        r = self._r
        ce = 0.0
        
        if self.mode_e==0: # normal
            ce = r.evaluate()
        elif self.mode_e==1: # layer MSE
            r.propagate()
            layer = r.get_layer_at(mse_idex) # 2, 4
            ent = layer.mse(0)
        elif self.mode_e==2: # MSE for autoencoder
            r.propagate()
            r._gpu.mse(r.output._gpu_output, r.input._gpu_output, r._gpu_entropy, self._data_size, self._batch_size)
            r._gpu.copy(r._batch_cross_entropy, r._gpu_entropy)
            ce = np.sum(r._batch_cross_entropy)/np.float32(self._batch_size)
        elif self.mode_e==3: # MSE for regression
            r.propagate()
            r._gpu.mse(r.output._gpu_output, r._gpu_labels, r._gpu_entropy, self._data_size, self._batch_size)
            r._gpu.copy(r._batch_cross_entropy, r._gpu_entropy)
            ce = np.sum(r._batch_cross_entropy)/np.float32(self._batch_size)
        else:
            print("Train::evaluate() N/A")
        #
        if self.mpi==False:
            return ce
        #
        
        ce_list = self.com.gather(ce, root=0)
        sum = 0.0
        if self.rank==0:
            for i in ce_list:
                sum = sum + i
            #
            avg = sum/float(self.size)
        #
        if self.rank==0:
            ce_avg_list = [avg]*self.size
        else:
            ce_avg_list = None
        #
        ce_avg = self.com.scatter(ce_avg_list, root=0)
        return ce_avg
    
    def make_w_list(self, type_list=None):
        r = self._r
        if type_list is None:
            type_list = [core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT, core.LAYER_TYPE_CONV_4]
        #
        w_list  = []
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
                        #wi = layer.get_weight_index(ni, ii)
                        #wi_alt = wi
                        #mark = 0
                        #w_list.append((li, ni, ii, wi, wi_alt, mark))
                        w_list.append(layer.getWeight(ni, ii))
                    #
                #
            #
        #
        return w_list
       
    def make_attack_list(self, attack_num, mode, w_list):
        r = self._r
        w_num = len(w_list)
        
        attack_list = []
        #while len(attack_list)<attack_num:
        for i in range(attack_num):
            attack_i = random.randrange(w_num)
            w = w_list[attack_i]
            while 1:
                w.wi_alt = random.randint(core.WEIGHT_INDEX_MIN, core.WEIGHT_INDEX_MAX)
                if w.wi_alt!=w.wi:
                    attack_list.append(attack_i)
                    break
                #
            #
        #
        return attack_list

    def undo_attack(self, w_list, attack_list):
        r = self._r
       
        llist = []
        for i in attack_list:
            w = w_list[i]
            layer = r.get_layer_at(w.li)
            layer.set_weight_index(w.ni, w.ii, w.wi)
            w.wi_alt = w.wi
            if w.li in llist:
                pass
            else:
                llist.append(w.li)
            #
        #

        for li in llist:
            layer = r.get_layer_at(li)
            layer.update_weight()
        #
        
    def multi_attack(self, ce, w_list, mode=1, div=0):
        r = self._r
        attack_list = self.make_attack_list(div, mode, w_list)
       
        llist = []
        w_num = len(attack_list)
        for i in attack_list:
            w = w_list[i]
            layer = r.get_layer_at(w.li)
            layer.set_weight_index(w.ni, w.ii, w.wi_alt)
            if w.li in llist:
                pass
            else:
                llist.append(w.li)
            #
        #

        for li in llist:
            layer = r.get_layer_at(li)
            layer.update_weight()
        #
        
        ce_alt = self.evaluate()
        ret = 0

        if ce_alt<=ce:
            ce = ce_alt
            ret = 1
            for i in attack_list:
                w = w_list[i]
                w.wi = w.wi_alt
            #
        else:
            self.undo_attack(w_list, attack_list)
        #
        return ce, ret
            
    def acceptance(self, ce1, ce2, temperature):
        delta = ce2 - ce1
        if delta<=0:
            return 1
        elif delta>ce1*0.005:
            return 0
        #

        A = np.exp(-delta/float(temperature)) / 100
        if np.random.random()<A:
            print(delta, "\t", temperature, A)
            return 1
        #
        return 0
        
    #
    # with acceptance control
    #
    def multi_attack_t(self, ce, w_list, attack_num, temperature):
        r = self._r
        
        attack_list = self.make_attack_list(attack_num, 0, w_list)
        llist = []
        w_num = len(attack_list)
        
        for i in attack_list:
            w = w_list[i]
            layer = r.get_layer_at(w.li)
            #if layer._type==core.LAYER_TYPE_CONV_4:
            #    layer.reset_weight_index(w.ni, w.ii)
            #else:
            #    layer.set_weight_index(w.ni, w.ii, w.wi_alt)
            #
            layer.set_weight_index(w.ni, w.ii, w.wi_alt)
            #print(i, w.wi, w.wi_alt)
            if w.li in llist:
                pass
            else:
                llist.append(w.li)
            #
        #
        #print(llist)
        
        for li in llist:
            layer = r.get_layer_at(li)
            layer.update_weight()
        #
        
        ce_alt = self.evaluate()
        ret = 0
        ret = self.acceptance(ce, ce_alt, temperature)
        #if ce_alt<=ce:
        #    if ce_alt<0:
        #        ret = -1
        #    else:
        #        ret = 1
        #    #
        #else: # acceptance control
        #    delta = ce_alt - ce
        #    hit = self.random_hit(delta, temperature)
        #    if hit==1:
        #        ret = 1
        #    else:
        #        ret = 0
        #    #
        #
        if ret<=0:
            self.undo_attack(w_list, attack_list)
        else:
            #print(ret, ce, ce_alt)
            ce = ce_alt
            for i in attack_list:
                w = w_list[i]
                w.wi = w.wi_alt
            #
        #
        return ce, ret

    #
    # a loop with a temperature
    #
    def loop_sa_t(self, loop, ce, temperature, attack_num, w_list, wtype, debug=0):
        r = self._r
        
        w_num = len(w_list)
        ret = 0
        part = 0 # hit count
        num = 0 # loop count
        min = 200
        hist = []
        atk_flag = True
        
        while atk_flag:
            ce, ret = self.multi_attack_t(ce, w_list, attack_num, temperature)
            if ret<0:
                return -1
            #
            num += 1
            part += ret
            hist.append(ret)
            if len(hist)>min:
                hist = hist[1:]
            #
            s = 0 # hit conunt in the last 200
            for a in hist:
                s += a
            #
            rate = float(s)/float(num)
            print("[%d:%d] %d [%s:%d] (%d, %d) %06f, ce:"
                    % (loop, temperature, num, wtype, attack_num, part, s, rate), ce)
            if num>min:
                if num>10000 or rate<0.01:
                    atk_flag = False
                #
            #
        # while
        
        return ce


    def logathic_loop(self, loop, w_list, wtype):
        r = self._r
        
        w_num = len(w_list)
        lv_min = 0
        lv_max = int(math.log(w_num/100, 2)) + 1 # 1 % max
        ce = self.evaluate()
        
        for temperature in range(lv_max, -1, -1):
            attack_num = 2**temperature
            ce = self.loop_sa_t(loop, ce, temperature, attack_num, w_list, wtype)
            if ce<0:
                break
            #
        #
        r.save()

    def multi_attack_sa(self, ce, w_list, attack_num, temperature):
        r = self._r
        
        if self.mpi==False or self.rank==0:
            attack_list = self.make_attack_list(attack_num, 0, w_list)
            alt_list = []
            for i in attack_list:
                w = w_list[i]
                alt_list.append(w.wi_alt)
            #
        else:
            attack_list = []
            alt_list = []
        #
        
        if self.mpi==True:
            attack_list = self.com.bcast(attack_list, root=0)
            alt_list = self.com.bcast(alt_list, root=0)
            
            if self.rank==0:
                pass
            else:
                idx = 0
                for i in attack_list:
                    #w = w_list[i]
                    w_list[i].wi_alt = alt_list[idx]
                    idx += 1
                #
            #
        #
        
        llist = []
        w_num = len(attack_list)
        for i in attack_list:
            w = w_list[i]
            layer = r.get_layer_at(w.li)
            layer.set_weight_index(w.ni, w.ii, w.wi_alt)
            if w.li in llist:
                pass
            else:
                llist.append(w.li)
            #
        #
        for li in llist:
            layer = r.get_layer_at(li)
            layer.update_weight()
        #
        
        ce_alt = self.evaluate()
        if self.mpi==False or self.rank==0:
            ret = self.acceptance(ce, ce_alt, temperature)
        else:
            ret = 0
        #
        if self.mpi==True:
            ret = self.com.bcast(ret, root=0)
        #
        
        if ret<=0:
            self.undo_attack(w_list, attack_list)
        else:
            ce = ce_alt
            for i in attack_list:
                #w = w_list[i]
                w_list[i].wi = w.wi_alt
            #
        #
        return ce, ret
    
    def loop_sa(self, w_list, wtype, debug=0):
        r = self._r
        
        w_num = len(w_list)
        attack_num = 1
        temperature = 100.0
        total = 10000
        #cnt = 0
        
        ce = r.evaluate()
        print(ce)
        
        while temperature>0.1:
            num = 0
            while num<(total):
                ce, ret = self.multi_attack_sa(ce, w_list, attack_num, temperature)
                if ret<0:
                    print(ret, "ce => zero")
                    return # -1
                #
                if self.mpi==False or self.rank==0:
                    print("[%d/%d]"%(num, total), "T=%f"%(temperature), "\t", ret, "\t", ce)
                #
                num += 1
            #
            temperature = temperature*0.95
            if self.mpi==False or self.rank==0:
                r.save()
            #
        #
#
#
#
