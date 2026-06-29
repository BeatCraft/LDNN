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
        self.mode_e = 0
        self.mse_idex = -1
        
        self.delta_num = 0.0
        self.delta_sum = 0.0
        self.delta_avg = 0.5
        
        if com:
            if rank<0 or size<0:
                return
            #
            self.mpi = True
            self.com = com
            self.rank = rank
            self.size = size
        else:
            self.mpi = False
        #
        self.w_list = []
        self.w_list_momentum = []
        self.w_lists = []
        
    def set_path(path):
        self._path = path
    
    def make_w_list_momentum(self):
        w_list_momentum = []
        for w in self.w_list:
            if w.momentum==0:
                pass
            else:
                w_list_momentum.append(w)
            #
        #
        return w_list_momentum
    
    def make_w_list_by_layer(self):
        r = self._r
        w_lists  = []
        c = r.count_layers()
        type_list = [core.LAYER_TYPE_CONV, core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT]
                    
        #for li in range(1, c):
        for li in range(c):
            layer = r.get_layer_at(li)
            type = layer.get_type()
            if type in type_list:
                w_list = []
                for ni in range(layer._num_node):
                    for ii in range(layer._num_input):
                        w_list.append(layer.get_weight(ni, ii))
                    #
                #
                w_lists.append(w_list)
            else:
                w_lists.append([])
            #
            #print(li, len(w_lists[li]))
        #
        return w_lists
        
    def make_w_list(self, type_list=None):
        r = self._r
        if type_list is None:
            type_list = [core.LAYER_TYPE_CONV, core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT]
        #
        w_list  = []
        c = r.count_layers()
        for li in range(1, c):
            layer = r.get_layer_at(li)
            type = layer.get_type()
            #for t in type_list:
            #    if type!=t:
            #        continue
            #    #
            if type in type_list:
                for ni in range(layer._num_node):
                    for ii in range(layer._num_input):
                        w_list.append(layer.get_weight(ni, ii))
                    #
                #
            #
        #
        return w_list

    def make_w_list_by_index(self, idx_list):
        r = self._r
        #
        w_list  = []
        for li in idx_list:
            layer = r.get_layer_at(li)
            cw = layer.count_weight()
            if cw > 0:
                for ni in range(layer._num_node):
                    for ii in range(layer._num_input):
                        w_list.append(layer.get_weight(ni, ii))
                    #
                #
            #
        #
        return w_list
    
    def make_attack_list(self, w_num, attack_num):
        attack_list = []
        while len(attack_list)<attack_num:
            widx = random.randint(0, w_num-1)
            w = self.w_list[widx]
            wi = w.wi
            attack_list.append((widx, wi))
        #
        return attack_list
    
    def attack(self, attack_list):
        for ws in attack_list:
            widx = ws[0]
            w = self.w_list[widx]
            layer = self._r.get_layer_at(w.li)
            if layer._type==core.LAYER_TYPE_HIDDEN or layer._type==core.LAYER_TYPE_OUTPUT:
                if self._r.wi_mode==3:
                    wi_alt = core.wi_8020()
                else:
                    wi_alt = random.randint(0, len(core.WEIGHT_SET)-1)
                #
            else:
                wi_alt = random.randint(0, len(core.CNN_WEIGHT_SET)-1)
            #
            w.wi = wi_alt
            layer.set_weight_index(w.ni, w.ii, wi_alt)
        #
        self._r.update_weight()
    
    def undo_attack(self, attack_list, w_list):
        for ws in attack_list:
            widx = ws[0]
            wi = ws[1]
            w = w_list[widx]
            w.wi = wi
            layer = self._r.get_layer_at(w.li)
            layer.set_weight_index(w.ni, w.ii, wi)
            w_list[widx].wi = wi
        #
        self._r.update_weight()

    def main_simple_loop(self, idx, loop, ce, loop_max, attack_num, save=0, debug=0, bi=0):
        r = self._r
        w_num = len(self.w_list)
        num = 0
        num_pre = 0
        hit = 0
        hit_pre = 0
        
        while num<loop_max:
            attack_list = self.make_attack_list(w_num, attack_num)
        
            # attack
            for ws in attack_list:
                widx = ws[0]
                w = self.w_list[widx]
                #wi_alt = random.randint(0, len(core.WEIGHT_SET)-1)
                layer = r.get_layer_at(w.li)
                if layer._type==core.LAYER_TYPE_HIDDEN or layer._type==core.LAYER_TYPE_OUTPUT:
                    if r.wi_mode==3:
                        wi_alt = core.wi_8020()
                    elif r.wi_mode==7:
                        wi_alt = core.wi_8020_3bit()
                    else:
                        wi_alt = random.randint(0, len(core.WEIGHT_SET)-1)
                    #
                else:
                    wi_alt = random.randint(0, len(core.CNN_WEIGHT_SET)-1)
                #
                w.wi = wi_alt
                layer.set_weight_index(w.ni, w.ii, wi_alt)
            #
            r.update_weight()
            
            ce_alt = r.evaluate(0)
            #if ce_alt<=ce: # keep
            if ce_alt<=ce: # keep
                ce = ce_alt
                ret = 1
                hit = hit + 1
            else: # undo
                self.undo_attack(attack_list, self.w_list)
            #
            if num>0 and num % 100 == 0:
                num_pre = num
                hit_pre = hit
                r.save()
            #
            num += 1
        #
        hit_rate = hit / loop_max
        print(bi, attack_num, "hit rate:", hit, "/", loop_max, "=", hit_rate, "ce=", ce)
        r.save()
        return ce, hit_rate

    def main_challenge_loop(self, ce, loop_max, attack_num, one=False, save=0, debug=0, bi=0):
        r = self._r
        w_num = len(self.w_list)
        num = 0
        num_pre = 0
        hit = 0
        hit_pre = 0
        sum_ce = 0.0

        while num<loop_max:
            attack_list = self.make_attack_list(w_num, attack_num)
        
            # attack
            for ws in attack_list:
                widx = ws[0]
                w = self.w_list[widx]
                layer = r.get_layer_at(w.li)
                if layer._type==core.LAYER_TYPE_HIDDEN or layer._type==core.LAYER_TYPE_OUTPUT:
                    if r.wi_mode==3:
                        wi_alt = core.wi_8020()
                    elif r.wi_mode==7:
                        #wi_alt = core.wi_8020_3bit()
                        #wi_alt = core.wi_std2()
                        #wi_alt = random.randrange(core.WEIGHT_INDEX_SIZE)
                        wi_alt = core.wi_std_11()
                    else:
                        wi_alt = random.randint(0, len(core.WEIGHT_SET)-1)
                    #
                else:
                    wi_alt = random.randint(0, len(core.CNN_WEIGHT_SET)-1)
                #
                w.wi = wi_alt
                layer.set_weight_index(w.ni, w.ii, wi_alt)
            #
            r.update_weight()
            
            ce_alt = r.evaluate(0)
            #if ce_alt<=ce: # keep
            if ce_alt<ce: # keep
                sum_ce += (ce - ce_alt)
                ce = ce_alt
                ret = 1
                hit = hit + 1
            else: # undo
                self.undo_attack(attack_list, self.w_list)
            #
            if num>0 and num % 100 == 0:
                num_pre = num
                hit_pre = hit
                r.save()
            #
            num += 1
            if one==True and hit>0:
                break
            #
        #
        hit_rate = hit / num #loop_max
        print(bi, attack_num, "hit rate:", hit, "/", num, "ce=", ce)
        #r.save()
        return ce, hit_rate

    def train_slope3(self, b, my_gpu, r, wmode, batch_size, ce, n, qfc, qcnn, attack_max_list, undo=False):
        r.slope(0)
        
        cnt = 0
        conv_cnt = 0
        lc = r.count_layers()
        if len(attack_max_list)==lc:
            pass
        else:
            print("error :: attack_max_list=%d, lc=%d" % (len(attack_max_list), lc))
            return 0.0
        #
        
        attack_cnt = [0]*lc
        attack_list_fc = []
        attack_list_cnn = []
        th_list = self.get_th_list2(r, qfc, qcnn)
    
        for li in range(lc):
            l = r.get_layer_at(li)
            if l.wcnt==0:
                continue
            #
            
            th = th_list[li]
            cand = np.argwhere(np.abs(l.dW) >= th)
            widx_list = list(range(len(cand)))
            random.shuffle(widx_list)
            cnt = 0
            for k in widx_list:
                ii, ni = cand[k]
                if cnt>attack_max_list[li]-1:
                    break
                #
                
                w = l.get_weight(ni, ii)
                wi = w.wi
                type = w.type
                if type==core.LAYER_TYPE_CONV:
                    kmax = core.CNN_WEIGHT_INDEX_MAX
                    kmin = core.CNN_WEIGHT_INDEX_MIN
                else:
                    kmax = core.WEIGHT_INDEX_MAX
                    kmin = core.WEIGHT_INDEX_MIN
                #
                
                g = l.dW[ii][ni]
                if g<0.0: # ++
                    if wi==kmax:
                        #print("kmax", type, li)
                        pass
                    else:
                        w.wi_alt = w.wi
                        w.wi = wi + 1
                        if type==core.LAYER_TYPE_CONV:
                            attack_list_cnn.append(w)
                        else:
                            attack_list_fc.append(w)
                        #
                        cnt += 1
                    #
                elif g>0.0: # --
                    if wi==kmin:
                        #print("kmin", type, li)
                        pass
                    else:
                        w.wi_alt = w.wi
                        w.wi = wi - 1
                        #attack_list_cnn.append(w)
                        if type==core.LAYER_TYPE_CONV:
                            attack_list_cnn.append(w)
                        else:
                            attack_list_fc.append(w)
                        #
                        cnt += 1
                    #
                else:
                    print("ZERO : no slope, must be error")
                #
            # for k
        # for li
        if len(attack_list_fc) + len(attack_list_cnn)==0:
            print("skip : none in attack_list")
            return 0
        #

        # cnn
        for w in attack_list_cnn:
            li = w.li
            l = r.get_layer_at(li)
            l.set_weight_index(w.ni, w.ii, w.wi) # attack
        #
        r.update_weight()
        
        ce_alt = r.evaluate(0)
        if ce_alt>ce: # undo
            if undo:
                print("[%d] *CNN(%d)" % (n, len(attack_list_cnn)), ce, "(", ce_alt, "), UNDO")
                for w in attack_list_cnn:
                    li = w.li
                    l = r.get_layer_at(w.li)
                    w.wi = w.wi_alt
                    l.set_weight_index(w.ni, w.ii, w.wi)
                #
                r.update_weight()
            #
            else:
                print("[%d] CNN(%d)" % (n, len(attack_list_cnn)), ce, "=>", ce_alt)
                ce = ce_alt
            #
        else:
            print("[%d] CNN(%d)" % (n, len(attack_list_cnn)), ce, "->", ce_alt)
            ce = ce_alt
        #

        # fc
        for w in attack_list_fc:
            li = w.li
            l = r.get_layer_at(li)
            l.set_weight_index(w.ni, w.ii, w.wi) # attack
        #
        r.update_weight()
        
        ce_alt = r.evaluate(0)
        if ce_alt>ce: # undo
            if undo:
                print("[%d] *FC(%d)" % (n, len(attack_list_fc)), ce, "(", ce_alt, "), UNDO")
                for w in attack_list_fc:
                    li = w.li
                    l = r.get_layer_at(w.li)
                    w.wi = w.wi_alt
                    l.set_weight_index(w.ni, w.ii, w.wi)
                #
                r.update_weight()
            #
            else:
                print("[%d] FC(%d)" % (n, len(attack_list_fc)), ce, "=>", ce_alt)
                ce = ce_alt
            #
        else:
            print("[%d] FC(%d)" % (n, len(attack_list_fc)), ce, "->", ce_alt)
            ce = ce_alt
        #
        
        return ce
        
    def train_slope2(self, b, my_gpu, r, wmode, batch_size, ce, n, qfc, qcnn, attack_max_list, undo=False):
        r.slope(0)
        
        cnt = 0
        conv_cnt = 0
        lc = r.count_layers()
        if len(attack_max_list)==lc:
            pass
        else:
            print("error :: attack_max_list=%d, lc=%d" % (len(attack_max_list), lc))
            return 0.0
        #
        attack_cnt = [0]*lc
        attack_list_fc = []
        attack_list_cnn = []
        th_list = self.get_th_list2(r, qfc, qcnn)
        #print(th_list)
    
        for li in range(lc):
            l = r.get_layer_at(li)
            if l.wcnt==0:
                continue
            #
            
            th = th_list[li]
            cand = np.argwhere(np.abs(l.dW) >= th)
            widx_list = list(range(len(cand)))
            random.shuffle(widx_list)
            cnt = 0
            for k in widx_list:
                ii, ni = cand[k]
                if cnt>attack_max_list[li]-1:
                    break
                #
                
                w = l.get_weight(ni, ii)
                wi = w.wi
                type = w.type
                if type==core.LAYER_TYPE_CONV:
                    kmax = core.CNN_WEIGHT_INDEX_MAX
                    kmin = core.CNN_WEIGHT_INDEX_MIN
                else:
                    kmax = core.WEIGHT_INDEX_MAX
                    kmin = core.WEIGHT_INDEX_MIN
                #
                
                g = l.dW[ii][ni]
                if g<0.0: # ++
                    if wi==kmax:
                        #print("kmax", type, li)
                        pass
                    else:
                        w.wi_alt = w.wi
                        w.wi = wi + 1
                        if type==core.LAYER_TYPE_CONV:
                            attack_list_cnn.append(w)
                        else:
                            attack_list_fc.append(w)
                        #
                        cnt += 1
                    #
                elif g>0.0: # --
                    if wi==kmin:
                        #print("kmin", type, li)
                        pass
                    else:
                        w.wi_alt = w.wi
                        w.wi = wi - 1
                        #attack_list_cnn.append(w)
                        if type==core.LAYER_TYPE_CONV:
                            attack_list_cnn.append(w)
                        else:
                            attack_list_fc.append(w)
                        #
                        cnt += 1
                    #
                else:
                    print("ZERO : no slope, must be error")
                #
            # for k
        # for li
        if len(attack_list_fc) + len(attack_list_cnn)==0:
            print("skip : none in attack_list")
            return 0
        #

        # cnn
        for w in attack_list_cnn:
            li = w.li
            l = r.get_layer_at(li)
            l.set_weight_index(w.ni, w.ii, w.wi) # attack
        #
        r.update_weight()
        
        ce_alt = r.evaluate(0)
        if ce_alt>ce: # undo
            if undo:
                print("[%d] *CNN(%d)" % (n, len(attack_list_cnn)), ce, "(", ce_alt, "), UNDO")
                for w in attack_list_cnn:
                    li = w.li
                    l = r.get_layer_at(w.li)
                    w.wi = w.wi_alt
                    l.set_weight_index(w.ni, w.ii, w.wi)
                #
                r.update_weight()
            #
            else:
                print("[%d] CNN(%d)" % (n, len(attack_list_cnn)), ce, "=>", ce_alt)
                ce = ce_alt
            #
        else:
            print("[%d] CNN(%d)" % (n, len(attack_list_cnn)), ce, "->", ce_alt)
            ce = ce_alt
        #

        # fc
        for w in attack_list_fc:
            li = w.li
            l = r.get_layer_at(li)
            l.set_weight_index(w.ni, w.ii, w.wi) # attack
        #
        r.update_weight()
        
        ce_alt = r.evaluate(0)
        if ce_alt>ce: # undo
            if undo:
                print("[%d] *FC(%d)" % (n, len(attack_list_fc)), ce, "(", ce_alt, "), UNDO")
                for w in attack_list_fc:
                    li = w.li
                    l = r.get_layer_at(w.li)
                    w.wi = w.wi_alt
                    l.set_weight_index(w.ni, w.ii, w.wi)
                #
                r.update_weight()
            #
            else:
                print("[%d] FC(%d)" % (n, len(attack_list_fc)), ce, "=>", ce_alt)
                ce = ce_alt
            #
        else:
            print("[%d] FC(%d)" % (n, len(attack_list_fc)), ce, "->", ce_alt)
            ce = ce_alt
        #
        
        return ce

    def train_slope(self, b, my_gpu, r, wmode, batch_size, attack_num, ce, n, undo=False):
        r.slope(0)
        cnt = 0
        attack_list = []
        th_list = []
        
        conv_cnt = 0
    
        lc = r.count_layers()
        for li in range(0, lc):
            layer = r.get_layer_at(li)
            type = layer.get_type()
            if type not in (core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_CONV, core.LAYER_TYPE_OUTPUT):
                th_list.append(np.float32(0.0))
            else:
                th = np.quantile(np.abs(layer.dW), 0.95)
                th_list.append(th)
            #
        #
    
        while cnt<attack_num:
            k = random.randint(0, len(self.w_list)-1)
            w = self.w_list[k]
            li = w.li
            ni = w.ni
            ii = w.ii
            l = r.get_layer_at(li)
            wi = w.wi
            type = w.type
            if type==core.LAYER_TYPE_CONV:
                kmax = core.CNN_WEIGHT_INDEX_MAX
                kmin = core.CNN_WEIGHT_INDEX_MIN
            else:
                kmax = core.WEIGHT_INDEX_MAX
                kmin = core.WEIGHT_INDEX_MIN
            #
        
            th = th_list[li]
            g = l.dW[ii][ni]
            if abs(g)<th:
                cnt += 1
                continue
            #
        
            if g<0.0: # ++
                if wi==kmax:
                    pass
                else:
                    w.wi_alt = w.wi
                    w.wi = wi + 1
                    l.set_weight_index(w.ni, w.ii, wi+1) # attack
                    attack_list.append(w)
                #
            elif g>0.0: # --
                if wi==kmin:
                    pass
                else:
                    w.wi_alt = w.wi
                    w.wi = wi - 1
                    l.set_weight_index(w.ni, w.ii, wi-1) # attack
                    attack_list.append(w)
                #
            #
            
            if type==core.LAYER_TYPE_CONV:
                conv_cnt += 1
            #
            cnt += 1
        # while
    
        if len(attack_list)==0:
            print("skip")
            return 0
        #
    
        r.update_weight()
        ce_alt = r.evaluate(0)
        if ce_alt>ce: # undo
            if undo:
                print("[%d](%d/%d)" % (n, len(attack_list), attack_num), ce, "(", ce_alt, "), UNDO")
                for w in attack_list:
                    li = w.li
                    l = r.get_layer_at(w.li)
                    w.wi = w.wi_alt
                    l.set_weight_index(w.ni, w.ii, w.wi)
                #
                r.update_weight()
            #
            else:
                print("[%d](%d/%d)" % (n, len(attack_list), attack_num), ce, "=>", ce_alt)
                ce = ce_alt
            #
        else:
            print("[%d](%d/%d)" % (n, len(attack_list), attack_num), ce, "->", ce_alt, conv_cnt)
            ce = ce_alt
            
            from collections import Counter
            print("layer update:", Counter(w.li for w in attack_list))
        #
        return ce

    def get_th_list2(self, r, qfc, qcnn):
        th_list = []
        
        lc = r.count_layers()
        for li in range(lc):
            layer = r.get_layer_at(li)
            type = layer.get_type()
            #
            if type==core.LAYER_TYPE_HIDDEN or type==core.LAYER_TYPE_OUTPUT:
                th = np.quantile(np.abs(layer.dW), qfc)
                th_list.append(th)
            elif type==core.LAYER_TYPE_CONV:
                th = np.quantile(np.abs(layer.dW), qcnn)
                th_list.append(th)
            else:
                th_list.append(np.float32(0.0))
            #
        #
        return th_list
        
    def get_th_list(self, r, batch_size):
        th_list = []
        
        lc = r.count_layers()
        for li in range(0, lc):
            layer = r.get_layer_at(li)
            type = layer.get_type()
            if type not in (core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_CONV, core.LAYER_TYPE_OUTPUT):
                th_list.append(np.float32(0.0))
            else:
                th = np.quantile(np.abs(layer.dWd/np.float32(batch_size)), 0.95)
                th_list.append(th)
            #
        #
        return th_list

    def slope_batch_attack(self, r, batch_size, attack_num, ce, th_list):
        cnt = 0
        attack_list = []
        while cnt<attack_num:
            k = random.randint(0, len(self.w_list)-1)
            w = self.w_list[k]
            li = w.li
            ni = w.ni
            ii = w.ii
            l = r.get_layer_at(li)
            wi = w.wi
            type = w.type
            if type==core.LAYER_TYPE_CONV:
                kmax = core.CNN_WEIGHT_INDEX_MAX
                kmin = core.CNN_WEIGHT_INDEX_MIN
            else:
                kmax = core.WEIGHT_INDEX_MAX
                kmin = core.WEIGHT_INDEX_MIN
            #
        
            th = th_list[li]
            g = l.dWd[ii][ni]
            if abs(g)<th:
                cnt += 1
                continue
            #
        
            if g<0.0: # ++
                if wi==kmax:
                    pass
                else:
                    w.wi_alt = w.wi
                    w.wi = wi + 1
                    l.set_weight_index(w.ni, w.ii, wi+1) # attack
                    print("-", w.value(), g)
                    attack_list.append(w)
                #
            elif g>0.0: # --
                if wi==kmin:
                    pass
                else:
                    w.wi_alt = w.wi
                    w.wi = wi - 1
                    l.set_weight_index(w.ni, w.ii, wi-1) # attack
                    print("+", w.value(), g)
                    attack_list.append(w)
                #
            #
            cnt += 1
        # while
        
        print(len(attack_list))
        r.update_weight()
        #ce_alt = r.evaluate(0)
        
    
    def zero_dWd(self, r):
        lc = r.count_layers()
        for li in range(lc):
            layer = r.get_layer_at(li)
            if layer.get_type() in (
                core.LAYER_TYPE_HIDDEN,
                core.LAYER_TYPE_CONV,
                core.LAYER_TYPE_OUTPUT
            ):
                layer.dWd[:] = np.float32(0.0)
            #
        # for
    

        
    
