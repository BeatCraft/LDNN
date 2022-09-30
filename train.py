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
        #self.rank = 0
        
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

        if self.mode_e==0 or self.mode_e==1:
            ce = r.evaluate()
            return ce
        elif self.mode_e==2 or self.mode_e==3:
            r.propagate()
            layer = r.get_layer_at(mse_idex) # 2, 4
            ent = layer.mse(0)
            return ent
        elif self.mode_e==4: # MSE for autoencoder
            r.propagate()
            r._gpu.mse(r.output._gpu_output, r.input._gpu_output, r._gpu_entropy, self._data_size, self._batch_size)
            r._gpu.copy(r._batch_cross_entropy, r._gpu_entropy)
            ce = np.sum(r._batch_cross_entropy)/np.float32(self._batch_size)
            return ce
        elif self.mode_e==5: # MSE for regression
            #print("MSE for regression")
            r.propagate()
            r._gpu.mse(r.output._gpu_output, r._gpu_labels, r._gpu_entropy, self._data_size, self._batch_size)
            r._gpu.copy(r._batch_cross_entropy, r._gpu_entropy)
            ce = np.sum(r._batch_cross_entropy)/np.float32(self._batch_size)
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

    def loop_sa5(self, idx, w_list, wtype, loop=1, min=100):#, atk=1000, atk_r = 0.01):
        r = self._r
        
        ce = self.evaluate()
        ret = 0
        w_num = len(w_list)
        d = 100
        lv_min = 0
        lv_max = int(math.log(w_num/d, 2)) + 1
        
        pbty = 0
        total = 0
        for lv in range(lv_max, -1, -1):
            div = 2**lv
            part = 0
            atk_flag = True
            num = 0
            hist = []
            while atk_flag:
                ce, ret, pbty = self.multi_attack_sa4(ce, w_list, 0, div, pbty)
                num += 1
                total += ret
                part += ret
                hist.append(ret)
                if len(hist)>min:
                    hist = hist[1:]
                #
                s = 0
                for a in hist:
                    s += a
                #
                rate = float(s)/float(len(hist))
                print("[", idx, "]", wtype, lv,"/", lv_max, "|", div, "(", num, ")", "ce", ce, part, total, "|", s, "/", len(hist), "=", rate)
                #
                if num>min:
                    if num>10000 or rate<0.01:
                        atk_flag = False
                    #
                #
            #
            r.save()
        #
        return ce
    
    def random_hit(self, delta, temperature):
        #A = np.exp(-delta/(temperature+0.000001)) / 1000.0
        #return 0
        A = np.exp(-np.abs(delta)/float(temperature)) / 1000.0
        if random.random() < A:
            print(delta, temperature, A)
            #return 1
        #
        
        #hit = random.choices([0, 1], k=1, weights=[1-A, A])
        #if hit[0]==1:
        #    print(A)
        #    return 1
        return 0
    
    def loop_sa_20(self, loop, w_list, debug=0):
        r = self._r
        ce = self.evaluate()
        print(ce)
        
        min=200
        w_num = len(w_list)
        ret = 0
        t_max = 100
        for t in range(t_max, 0, -1):
            total = 0
            rec = [0] * (t_max+1)
            hist = []

            atk_flag = True
            part = 0
            num = 0
            while atk_flag:
                atk_num = random.randint(1, t)
                attack_list = self.make_attack_list(atk_num, 0, w_list)
                alt_list = []
                for i in attack_list:
                    w =w_list[i]
                    alt_list.append(w.wi_alt)
                #
   
                idx = 0
                for i in attack_list:
                    w = w_list[i]
                    w.wi_alt = alt_list[idx]
                    idx += 1
                #
                
                llist = []
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
                    ret = 1
                    ce = ce_alt
                    for i in attack_list:
                        w = w_list[i]
                        w.wi = w.wi_alt
                    #
                else:
                    delta = ce_alt - ce
                    hit = self.random_hit(delta, t)
                    if hit==1:
                        ret = 1
                        ce = ce_alt
                    else:
                        ret = 0
                        self.undo_attack(w_list, attack_list)
                    #
                #
                
                num += 1
                total += ret
                part += ret
                hist.append(ret)
                if len(hist)>min:
                    hist = hist[1:]
                #
                s = 0
                for a in hist:
                    s += a
                #
                rate = float(s)/float(num)
                print("[%d] %d/%d (%d) [fc:%03d] ce %09f (%d, %d, %d) %f" %(loop, t, t_max, num, atk_num, ce, part, total, s, rate))
                if num>min:
                    if num>10000 or rate<0.01:
                        atk_flag = False
                    #
                #
            # while
            rec[t] = part
        # for t

        r.save()
        if debug==1:
            log = "%d, %s" % (loop+1, '{:.10g}'.format(ce))
            output("./log.csv", log)
            spath = "./wi/wi-%04d.csv" % (loop+1)
            r.save_as(spath)
        #
        return ce
        
        
        
        
        
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

        if ce_alt<=ce:
            ret = 1
        else: # acceptance control
            delta = ce_alt - ce
            hit = self.random_hit(delta, temperature)
            if hit==1:
                ret = 1
            else:
                ret = 0
            #
        #
        if ret==0:
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

    def make_attack_list_d(self, attack_num, temperature, fc_list, cnn_list):
        r = self._r
        
        fc_num = len(fc_list)
        cnn_num = len(cnn_list)
        #w_num = len(w_list)
        #cnn_ratio = float(cnn_num)/float(cnn_num)
        #fc_ratio = 1 - cnn_ratio
        fc_cnt = 0
        cnn_cnt = 0

        attack_list = []
        t_max = random.randint(0, attack_num)
        for k in range(t_max):
            attack_i = random.randrange(fc_num)
            w = fc_list[attack_i]
            fc_cnt += 1
            while 1:
                w.wi_alt = random.randint(core.WEIGHT_INDEX_MIN, core.WEIGHT_INDEX_MAX)
                if w.wi_alt!=w.wi:
                    attack_list.append([0, attack_i])
                    break
                #
            #
        #

        t_max = random.randint(0, 5)#temperature)
        for k in range(t_max):
            attack_i = random.randrange(cnn_num)
            w = cnn_list[attack_i]
            cnn_cnt += 1
            while 1:
                w.wi_alt = random.randint(core.WEIGHT_INDEX_MIN, core.WEIGHT_INDEX_MAX)
                if w.wi_alt!=w.wi:
                    attack_list.append([1, attack_i])
                    break
                #
            #
        #        

        #print(fc_cnt, cnn_cnt, len(attack_list))
        return attack_list

    def undo_attack_d(self, fc_list, cnn_list, attack_list):
        r = self._r

        llist = []
        for aw in attack_list:
            w_type = aw[0]
            i = aw[1]
            if w_type==0:
                w = fc_list[i]
            else:
                w = cnn_list[i]
            #
            #w = w_list[i]
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

    def multi_attack_d(self, ce, fc_list, cnn_list, attack_num, temperature):
        r = self._r

        attack_list = self.make_attack_list_d(attack_num, temperature, fc_list, cnn_list)
        llist = []
        w_num = len(attack_list)


        for aw in attack_list:
            w_type = aw[0]
            i = aw[1]
            if w_type==0: # fc
                w = fc_list[i]
            else:
               w = cnn_list[i]
            #
            #w = w_list[i]
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
            for aw in attack_list:
                w_type = aw[0]
                i = aw[1]
                #w = w_list[i]
                if w_type==0:
                    w = fc_list[i]
                else:
                    w = cnn_list[i]
                #
                w.wi = w.wi_alt
            #
        else:
            # acceptance control
            delta = ce_alt - ce
            hit = self.random_hit(delta, temperature)
            if hit==1:
                ret = 1
                ce = ce_alt
            else:
                ret = 0
            #
        #
        if ret==0:
            self.undo_attack_d(fc_list, cnn_list, attack_list)
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
            print("[%d:%d] %d [%s:%d] ce %09f (%d, %d) %06f"
                    % (loop, temperature, num, wtype, attack_num, ce, part, s, rate))
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
        #
        r.save()

    def loop_sa_d(self, loop, ce, temperature, attack_num, fc_list, cnn_list, wtype, debug=0):
        r = self._r

        #w_num = len(w_list)
        ret = 0
        part = 0 # hit count
        num = 0 # loop count
        min = 200
        hist = []
        atk_flag = True

        while atk_flag:
            ce, ret = self.multi_attack_d(ce, fc_list, cnn_list, attack_num, temperature)
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
            print("[%d:%d] %d [%s:%d] ce %09f (%d, %d) %06f"
                    % (loop, temperature, num, wtype, attack_num, ce, part, s, rate))
            if num>min:
                if num>10000 or rate<0.01:
                    atk_flag = False
                #
            #
        # while

        return ce

    def loop_sa_alt(self, loop, ce, temperature, fc_list, fc_attack_num, cnn_list, cnn_attack_num, wtype, debug=0):
        r = self._r
        
        ret = 0
        cnt = 0
        for i in range(100):
            ce, ret = self.multi_attack_t(ce, cnn_list, cnn_attack_num, temperature)
            cnt += ret
            print("[%d:%d] cnn [%d] %d : ce %f" % (loop, i, cnn_attack_num, cnt, ce))
        #

        cnt = 0
        for i in range(100):
            ce, ret = self.multi_attack_t(ce, fc_list, fc_attack_num, temperature)
            cnt += ret
            print("[%d:%d] fc [%d] %d : ce %f" % (loop, i, fc_attack_num, cnt, ce))
        #

        return ce
#
#
#
    
    
