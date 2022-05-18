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
    def __init__(self, r):
        self._r = r
        self.mode = 0
        self.mode_w = 0
        self.mode_e = 0
        self.mse_idex = -1
        
    def set_batch(self, data_size, num_class, train_data_batch, train_label_batch,  batch_size, batch_offset):
        self._bacth_size = batch_size
        self._data_size = data_size
        #
        r = self._r
        r.prepare(batch_size, data_size, num_class)
        r.set_batch(data_size, num_class, train_data_batch, train_label_batch,  batch_size, batch_offset)
    
    def set_path(path):
        self._path = path
    
    def evaluate(self):#, mode=0, idx=0):
        r = self._r

        if self.mode_e==0 or self.mode_e==1:
            ce = r.evaluate()
            return ce
        elif self.mode_e==2 or self.mode_e==3:
            r.propagate()
            layer = r.get_layer_at(mse_idex) # 2, 4
            ent = layer.mse(0)
            return ent
        elif self.mode_e==4:
            r.propagate()
            r._gpu.mse(r.output._gpu_output, r.input._gpu_output, r._gpu_entropy, self._data_size, self._bacth_size)
            r._gpu.copy(r._batch_cross_entropy, r._gpu_entropy)
            ce = np.sum(r._batch_cross_entropy)/np.float32(batch_size)
            return ce
        #
        
        return 0.0
    
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
        while len(attack_list)<attack_num:
            attack_i = random.randrange(w_num)
            w = w_list[attack_i]
            while 1:
                w.wi_alt = random.randint(core.WEIGHT_INDEX_MIN, core.WEIGHT_INDEX_MAX)
                if w.wi_alt!=w.wi:
                    attack_list.append(attack_i)
                    break
                #
            #
            #if mode>0: # heat
            #    if w.wi<core.WEIGHT_INDEX_MAX:
            #        #w.wi_alt = w.wi + 1
            #        w.wi_alt = random.randint(w.wi, core.WEIGHT_INDEX_MAX)
            #        attack_list.append(attack_i)
            #    #
            #else:
            #    if w.wi>core.WEIGHT_INDEX_MIN:
            #        #w.wi_alt = w.wi - 1
            #        w.wi_alt = random.randint(core.WEIGHT_INDEX_MIN, w.wi)
            #        attack_list.append(attack_i)
            #    #
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
            w.wi_alt = w.wi # ?
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
        #
        # need to implement acceptance control here??
        #
        #de = (ce - ce_alt)
        #flg = math.exp(de/float(div)) < random.random()
        #/float(div)
        #ac = 2**(-de/float(div))
        #ac = math.e**(de/float(div))
        # log(div, 2) #math.e**(-de/div)
        #c = math.e**(-de)
        #ac = math.e**(de)
        #ac = np.exp(de)
        #print(ac)
        #k = 1.38 * (10**-23)
        #print(np.exp(-ac/k*math.log(div, 2)))
        if ce_alt<=ce:
            ce = ce_alt
            ret = 1
            for i in attack_list:
                w = w_list[i]
                w.wi = w.wi_alt
            #
        else:
            #if flg:
            #    print("####", math.exp(de/float(div)))
            #    ce = ce_alt
            #    ret = 1
            #    for i in attack_list:
            #        w = w_list[i]
            #        w.wi = w.wi_alt
            #    #
            #
            #print(de, ac)
            self.undo_attack(w_list, attack_list)
        #
        return ce, ret
        
        #if self.mode==0 or self.mode==1 or self.mode==4:
        #    if ce_alt<ce:
        #        ce = ce_alt
        #        ret = 1
        #    else:
        #        self.undo_attack(attack_list)
        #elif self.mode==2 or self.mode==3:
        #    if ce_alt>ce:
        #        ce = ce_alt
        #        ret = 1
        #    else:
        #        self.undo_attack(attack_list)
        #    #
        ##
        #
        #return ce, ret

    def loop_sa(self, w_list, wtype, m, n=1, atk=50, atk_r = 0.05):
        r = self._r
        
        ce = self.evaluate()
        ret = 0
        w_num = len(w_list)
        d = 100
        lv_min = 0
        lv_max = int(math.log(w_num/d, 2)) + 1
        total = 0
        
        for j in range(n):
            for lv in range(lv_min, lv_max+1):
                div = 2**lv
                total = 0
                for i in range(atk - int(atk*atk_r*lv)):
                    ce, ret = self.multi_attack(ce, w_list, 1, div)
                    total += ret
                    print(m, wtype, "[", j, "] lv", lv, div, "i", i, "ce", ce, total)
                #
            #
            
            for lv in range(lv_max, -1, -1):
                div = 2**lv
                total = 0
                #for i in range(atk - int(atk*atk_r*lv)):
                for i in range(atk):
                    ce, ret = self.multi_attack(ce, w_list, 0, div)
                    total += ret
                    print(m, wtype, "[", j, "] lv", lv, div, "i", i, "ce", ce, total)
                #
            #
            r.save()
        #
        return 0
        
    def loop_sa2(self, w_list, wtype, m, n=1, atk=50, atk_r = 0.05):
        r = self._r
        
        ce = self.evaluate()
        ret = 0
        w_num = len(w_list)
        d = 100
        lv_min = 0
        lv_max = int(math.log(w_num/d, 2)) + 1
        total = 0
        
        for j in range(n):
            for lv in range(lv_max, -1, -1):
                div = 2**lv
                total = 0
                #for i in range(atk - int(atk*atk_r*lv)):
                for i in range(atk):
                    ce, ret = self.multi_attack(ce, w_list, 0, div)
                    total += ret
                    print(m, wtype, "[", j, "] lv", lv, div, "i", i, "ce", ce, total)
                #
            #
            r.save()
        #
        return 0
        
    def loop_sa3(self, w_list, wtype, m, n=1, atk=50, atk_r = 0.05):
        r = self._r
        
        ce = self.evaluate()
        ret = 0
        w_num = len(w_list)
        d = 100
        lv_min = 0
        lv_max = int(math.log(w_num/d, 2)) + 1
        total = 0
        
        for j in range(n):
            for lv in range(lv_max, -1, -1):
                div = 2**lv
                total = 0
                part = 0
                i = 0
                k = 0
                flag = 1
                while flag:
                    ce, ret = self.multi_attack(ce, w_list, 0, div)
                    total += ret
                    part += ret
                    print(m, wtype, "[", j, "] lv", lv, div, "(", i, ")", "ce", ce, total)
                    #
                    if k>atk:
                        tr = float(total)/float(i)
                        pr = float(part)/float(k)
                        if tr<0.05 and pr<0.05:
                            flag = 0
                            #
                            # need to refreash
                            #
                        else:
                            if i>=atk*20:
                                flag = 0
                            #
                        #
                    #
                    i += 1
                    k += 1
                # while
                #i = 0
                #j = 0
            #
            r.save()
        #
        return 0
    
    
    def multi_attack_sa4(self, ce, w_list, mode=1, div=0, pbty=0):
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
            # acceptance control
            delta = abs(ce_alt - ce)
            if pbty>3: # 2, 3?
                diff = math.log10(delta) - math.log10(ce)
                limit = -1.0# - 1.0/(1.0+math.log(div, 2))
                print(diff, limit)
                if (diff < limit):
                    print("RESET", diff, limit)
                    ce = ce_alt
                    ret = 1
                    pbty = 0
                #
            #
        #
        if ret==0:
            self.undo_attack(w_list, attack_list)
        #
        return ce, ret, pbty
    
    def loop_sa4(self, w_list, wtype, n=1, atk=1000, atk_r = 0.01):
        r = self._r
        
        ce = self.evaluate()
        ret = 0
        w_num = len(w_list)
        d = 100
        lv_min = 0
        lv_max = int(math.log(w_num/d, 2)) + 1
        
        pbty = 0

        for j in range(n):
            total = 0
            for lv in range(lv_max, -1, -1):
                div = 2**lv
                part = 0
                for i in range(atk):
                    ce, ret, pbty = self.multi_attack_sa4(ce, w_list, 0, div, pbty)
                    total += ret
                    part += ret
                    print(wtype, "[", j, "]", wtype, lv,"/", lv_max, "|", div, "(", i, ")", "ce", ce, part, total, pbty)
                    #
                #
                rate = float(part)/float(atk)
                if rate<atk_r:
                    lv_max -= 1
                    if lv_max<3:
                        lv_max = 3
                        pbty += 1
                    #
                #
            #
            r.save()
        #
        return 0
        

    
