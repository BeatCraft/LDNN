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
        
    def set_path(path):
        self._path = path
    
    def make_w_list_momentum(self):
        w_list_momentum = []
        for w in self.w_list:
            if w.momentum==0:
                pass
            else:
                #print(w.momentum)
                w_list_momentum.append(w)
            #
        #
        return w_list_momentum
    
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
        
            for t in type_list:
                if type!=t:
                    continue
                #
                for ni in range(layer._num_node):
                    for ii in range(layer._num_input):
                        #w_list.append(layer.getWeight(ni, ii))
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
                        #w_list.append(layer.getWeight(ni, ii))
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
            #w = self.w_list[widx]
            w = w_list[widx]
            w.wi = wi
            layer = self._r.get_layer_at(w.li)
            layer.set_weight_index(w.ni, w.ii, wi)
            #self.w_list[widx].wi = wi
            w_list[widx].wi = wi
        #
        self._r.update_weight()

    def undo_attack_reset_momentum(self, attack_list):
        for ws in attack_list:
            widx = ws[0]
            wi = ws[1]
            w = self.w_list[widx]
            w.wi = wi
            layer = self._r.get_layer_at(w.li)
            layer.set_weight_index(w.ni, w.ii, wi)
            self.w_list[widx].wi = wi
            self.w_list[widx].momentum = 0
        #
        self._r.update_weight()
        
    def attack_set_momentum(self, attack_list):
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
            
            #
            # momentum
            #
            momentum = w.wi - wi_alt
            #print("momentum: wi_alt - w.wi :", wi_alt, w.wi, momentum)
            if momentum>0:
                w.momentum = 1
            elif momentum<0:
                w.momentum = -1
            else:
                w.momentum = 0
            #
            # momentum
            #
            
            w.wi = wi_alt
            layer.set_weight_index(w.ni, w.ii, wi_alt)
        #
        self._r.update_weight()
        
    def main_simple_loop(self, idx, loop, ce, loop_max, attack_num, save=0, debug=0):
        r = self._r
        w_num = len(self.w_list)
        num = 0
        num_pre = 0
        hit = 0
        hit_pre = 0
        
        while num<loop_max:
            attack_list = self.make_attack_list(w_num, attack_num)
            #attack_list = []
            #while len(attack_list)<attack_num:
            #    widx = random.randint(0, w_num-1)
            #    w = self.w_list[widx]
            #    wi = w.wi
            #    attack_list.append((widx, wi))
            ##
        
            # attack
            for ws in attack_list:
                widx = ws[0]
                w = self.w_list[widx]
                #wi_alt = random.randint(0, len(core.WEIGHT_SET)-1)
                layer = r.get_layer_at(w.li)
                if layer._type==core.LAYER_TYPE_HIDDEN or layer._type==core.LAYER_TYPE_OUTPUT:
                    if r.wi_mode==3:
                        wi_alt = core.wi_8020()
                        #wi_alt = random.randint(0, len(core.WEIGHT_SET)-1)
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
            if ce_alt<=ce: # keep
                #print(idx, loop, "[%d/%d]"%(num, loop_max), attack_num, "\t", ce, ">", ce_alt)
                #print("[%d/%d]" % (num, loop_max), attack_num, "\t", ce, ">", ce_alt)
                print("[%d][%d, %d/%d] %d : " % (idx, loop, num, loop_max, attack_num), ce, ">", ce_alt)
                ce = ce_alt
                ret = 1
                hit = hit + 1
            else: # undo
                #print(idx, loop, "[%d/%d]"%(num, loop_max), attack_num, "\t", ce)
                #print("[%d/%d]" % (num, loop_max), attack_num, "\t", ce)
                print("[%d][%d, %d/%d] %d : " % (idx, loop, num, loop_max, attack_num), ce)
                self.undo_attack(attack_list, self.w_list)
                                
                #for ws in attack_list:
                #    widx = ws[0]
                #    wi = ws[1]
                #    w = self.w_list[widx]
                #    w.wi = wi
                #    layer = r.get_layer_at(w.li)
                #    layer.set_weight_index(w.ni, w.ii, wi)
                ##
                #r.update_weight()
            #
            if num>0 and num % 100 == 0:
                print("hit rate:", hit - hit_pre, "/ 100 = ", float((hit - hit_pre)/(100)))
                print("hit rate:", hit, "/", loop_max, "=", float(hit/loop_max))
                num_pre = num
                hit_pre = hit
                r.save()
            #
            num += 1
        #
        print("hit rate:", hit, "/", loop_max, "=", float(hit/loop_max))
        r.save()
        return ce

    def momentum_loop(self, idx, loop, loop_max, attack_num, save=0, debug=0):
        r = self._r
        w_num = len(self.w_list)
        num = 0
        hit = 0
        ce = r.evaluate(0)
        
        #
        # attack
        #
        while num<loop_max:
            attack_list = self.make_attack_list(w_num, attack_num)
            self.attack_set_momentum(attack_list)
            ce_alt = r.evaluate(0)
            if ce_alt<=ce: # keep
                print(idx, "%d, [%d/%d]" % (loop, num, loop_max), attack_num, "\t", ce, ">", ce_alt)
                ce = ce_alt
                ret = 1
                hit = hit + 1

            else: # undo
                print(idx, "%d, [%d/%d]" % (loop, num, loop_max), attack_num, "\t", ce)
                self.undo_attack_reset_momentum(attack_list)
            #
            num = num + 1
        #


    def make_attack_list_momentum(self, w_num, attack_num):
        attack_list = []
        while len(attack_list)<attack_num:
            widx = random.randint(0, w_num-1)
            w = self.w_list_momentum[widx]
            wi = w.wi
            attack_list.append((widx, wi))
        #
        return attack_list
        
    def reset_mommentum(self):
        r = self._r
        for w in self.w_list:
            w.momentum = 0
        #
        
    def auto_momentum_loop(self, idx, loop, loop_max, attack_num, save=0, debug=0):
        #print("train::auto_momentum_loop()")
        r = self._r
        num = 0
        hit = 0
        
        self.w_list_momentum = self.make_w_list_momentum()
        #print(self.w_list_momentum)
        w_num = len(self.w_list_momentum)
        if w_num>0:
            pass
        else:
            return 0
        #
        
        while num<loop_max:
            #
            # attack
            #
            attack_list = self.make_attack_list_momentum(w_num, attack_num)
            ce = r.evaluate(0)
        
            for ws in attack_list:
                widx = ws[0]
                w = self.w_list_momentum[widx]
                wi = w.wi
                momentum = w.momentum
                wi_alt = w.wi + momentum
                if momentum>0:
                    if wi_alt<=core.WEIGHT_INDEX_MAX:
                        w.wi = wi_alt
                        layer = r.get_layer_at(w.li)
                        layer.set_weight_index(w.ni, w.ii, w.wi)
                    #
                else:
                    if wi_alt>=0:
                        w.wi = wi_alt
                        layer = r.get_layer_at(w.li)
                        layer.set_weight_index(w.ni, w.ii, w.wi)
                    #
                #
            #
            r.update_weight()
            ce_alt = r.evaluate(0)

            if ce_alt<=ce: # keep
                print("*", idx, "%d, [%d/%d]" % (loop, num, loop_max), attack_num, "\t", ce, ">", ce_alt)
                ce = ce_alt
                ret = 1
                hit = hit + 1
            else: # undo
                print("*", idx, "%d, [%d/%d]"%(loop, num, loop_max), attack_num, "\t", ce)
                self.undo_attack(attack_list, self.w_list_momentum)
            #
            num = num + 1
        #
        self.reset_mommentum()
        
        return 1
