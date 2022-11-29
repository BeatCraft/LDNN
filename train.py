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
            type_list = [core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT, core.LAYER_TYPE_CONV]
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
                        w_list.append(layer.getWeight(ni, ii))
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
                        w_list.append(layer.getWeight(ni, ii))
                    #
                #
            #
        #
        return w_list
    
    def shift_weight(self, w, wi, mode=0, type=-1):
        #print("shift_weight", wi, type)
        wi_alt = wi
        wi_min = 0
        wi_max = 1
        
        if type==core.LAYER_TYPE_HIDDEN or type==core.LAYER_TYPE_OUTPUT:
            wi_max = core.WEIGHT_INDEX_MAX
        elif type==core.LAYER_TYPE_CONV:
            wi_max = core.CNN_WEIGHT_INDEX_MAX
        #
        
        if type==core.LAYER_TYPE_CONV:
            if wi>2 or wi_alt>2:
                print("AAA(%d, %d, %d), wi=%d, alt=%d, max=%d (%d)"
                        % (w.li, w.ni, w.ii, wi, wi_alt, wi_max, w.mark))
            #
        #
        
        if mode==0: # random
            wi_alt = random.randint(wi_min, wi_max)
            if type==core.LAYER_TYPE_CONV and wi>2:
                print("FUCK!!")
                print(" (%d, %d, %d)" %(w.li, w.ni, w.ii) )
                print(" wi=%d, alt=%d, max=%d (%d)" % (wi, wi_alt, wi_max, w.mark))
            #
        elif mode==1: # neighbor
            k = random.random()

            if k<0.5: # increase
                if wi==wi_max:
                    wi_alt = random.randint(wi_min, wi_max-1)
                else:
                    wi_alt = wi + 1
                    if type==core.LAYER_TYPE_CONV and wi_alt>2:
                        print("FUCK!!")
                        print(" (%d, %d, %d)" %(w.li, w.ni, w.ii) )
                        print(" wi=%d, alt=%d, max=%d (%d) %f" % (wi, wi_alt, wi_max, w.mark, k))
                    #
                #
            else: # decrease
                if wi==wi_min:
                    wi_alt = random.randint(wi_min+1, wi_max)
                else:
                    wi_alt = wi - 1
                #
                if type==core.LAYER_TYPE_CONV and wi_alt>2:
                    print("FUCK!!")
                    print(" (%d, %d, %d)" %(w.li, w.ni, w.ii) )
                    print(" wi=%d, alt=%d, max=%d (%d) %f" % (wi, wi_alt, wi_max, w.mark, k))
                #
            #
        #
        
        if type==core.LAYER_TYPE_CONV:
            if wi>2 or wi_alt>2:
                print("KKK(%d, %d, %d), wi=%d, alt=%d, max=%d (%d)" % (w.li, w.ni, w.ii, wi, wi_alt, wi_max, w.mark))
            #
        #
        w.mark = w.mark + 1
        return wi_alt
    
    def make_attack_list(self, attack_num, mode, w_list):
        r = self._r
        w_num = len(w_list)
        
        attack_list = []
        #while len(attack_list)<attack_num:
        for i in range(attack_num):
            attack_i = random.randrange(w_num)
            w = w_list[attack_i]
            type = w.type
            if type==core.LAYER_TYPE_CONV:
                if w.wi>2 or w.wi_alt>2:
                    print("CCC(%d, %d, %d), wi=%d, alt=%d, (%d)" %
                        (w.li, w.ni, w.ii, w.wi, w.wi_alt, w.mark))
                #
            #
            
            # mode : 0=random, 1=neighbor
            w.wi_alt = self.shift_weight(w, w.wi, mode, type)
            if type==3 and w.wi_alt>2:
                print("KOKO:", type, w.wi, mode, w.wi_alt, w.ni, w.ii, w.mark)
            #
            attack_list.append(attack_i)
        #
        for idx in attack_list:
            aw = w_list[idx]
            if aw.type==3 and aw.wi_alt>2:
                print("DDDDD")
                print(" (%d, %d, %d)" %(aw.li, aw.ni, aw.ii) )
                print(" wi=%d, alt=%d (%d)" % (aw.wi, aw.wi_alt, w.mark))
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
            
    def acceptance(self, ce1, ce2, temperature, onoff=1):
        delta = ce2 - ce1
        if delta<=0:
            return 1
        #
        if onoff==0:
            return 0
        #
        if delta>self.delta_avg:
            return 0
        #

        A = np.exp(-delta/float(temperature)) / 100
        if np.random.random()<A:
            print(delta, "\t", temperature, A)
            return 1
        #
        return 0

    def multi_attack_sa(self, ce, w_list, attack_num, temperature, asw=1):
        r = self._r
        #attack_list = []
        if self.mpi==False or self.rank==0:
            mode = 1 # 0:random, 1:neighbor
            attack_list = self.make_attack_list(attack_num, mode, w_list)
            for a in attack_list:
                w = w_list[a]
                if w.type==core.LAYER_TYPE_CONV and w.wi>2:
                    print("*** (%d, %d, %d), wi=%d, alt=%d (%d)" %
                        (w.li, w.ni, w.ii, w.wi, w.wi_alt, w.mark))
                #
            #
            
            #print(len(attack_list))
            #k = attack_list[0]
            #print(w_list[k].wi, w_list[k].wi_alt)
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
                    #print(i, idx, len(w_list))
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
            #
            if w.type==core.LAYER_TYPE_CONV and w.wi>2:
                print("$$$ (%d, %d, %d), wi=%d, alt=%d (%d)" % (w.li, w.ni, w.ii, w.wi, w.wi_alt, w.mark))
                #
            #
            
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
            ret = self.acceptance(ce, ce_alt, temperature, asw)
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
                w_list[i].wi = w.wi_alt
            #
        #
        return ce, ret
    
    def loop_sa(self, idx, ce, w_list, wtype, temperature, total, minimum=0.1, debug=0, asw=1):
        r = self._r
        
        w_num = len(w_list)
        attack_num = 1
        ce = r.evaluate()
        print("ce:", ce)
        self.delta_avg = ce * 0.1
        t_min = minimum # 1.0, 0.1, 0.01
        lp = 0
        
        while temperature>t_min:
            num = 0
            self.delta_avg = ce*0.1
            while num<(total):
                ce, ret = self.multi_attack_sa(ce, w_list, attack_num, temperature, asw)
                if ce<0:
                    print(ret, "ce => zero")
                    return # -1
                #
                if self.mpi==False or self.rank==0:
                    print(idx, "[%d/%d]"%(num, total), "T=%f"%(temperature), "(%d)"%(lp), "\t", ret, "\t", ce)
                #
                num += 1
                self.delta_avg = ce*0.1
            #
            lp = lp +1
            temperature = temperature*0.95 #0.90, 0.95
            #if self.mpi==False or self.rank==0:
            #    r.save()
            #
        #
        return ce
        #if self.mpi==False or self.rank==0:
        #    r.save()
        #
        
    def loop_logathic_sa(self, idx, w_list, wtype, asw=1):
        r = self._r
        
        w_num = len(w_list)
        lv_min = 0
        lv_max = int(math.log(w_num/100, 2)) + 1 # 1 % max
        ce = self.evaluate()
        
        min = 200
        for temperature in range(lv_max, 0, -1):
            attack_num = 2 ** temperature
            hist = []
            atk_flag = True
            part = 0 # hit count
            num = 0  # loop count
            while atk_flag:
                ce, ret = self.multi_attack_sa(ce, w_list, attack_num, temperature, asw)
                if ret<0:
                    print("exit from while :", ce, ret)
                    #atk_flag = False
                    break
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
                if self.mpi==False or self.rank==0:
                    print("[%d:%d] %d [%s:%d] (%d, %d) %06f, ce:"
                        % (idx, temperature, num, wtype, attack_num, part, s, rate), ce)
                #
                if num>min:
                    if num>10000 or rate<0.01:
                        atk_flag = False
                    #
                #
            # while
            if ce<=0:
                print("exit from for")
                break
            #
        #
        if self.mpi==False or self.rank==0:
            r.save()
        #
    
    def shift_weight_2(self, type, wi, mode=0):
        wi_alt = wi
        wi_min = 0
        wi_max = 1
        
        if type==core.LAYER_TYPE_HIDDEN:
            wi_max = core.WEIGHT_INDEX_MAX
        if type==core.LAYER_TYPE_OUTPUT:
            wi_max = core.WEIGHT_INDEX_MAX
        elif type==core.LAYER_TYPE_CONV:
            wi_max = core.CNN_WEIGHT_INDEX_MAX
            if wi_alt>wi_max:
                print("error :", wi)
            #
        #
        
        if mode==0: # random
            wi_alt = random.randint(wi_min, wi_max)
        elif mode==1: # neighbor
            k = random.random()
            if k<0.5: # increase
                if wi==wi_max:
                    wi_max = wi_max - 1
                    wi_alt = random.randint(wi_min, wi_max)
                else:
                    wi_alt = wi + 1
                    #
                    if type==3 and wi_alt>2:
                        print("error in increase", wi, ">", wi_alt)
                    #
                #
            else: # decrease
                if wi==wi_min:
                    wi_min = 1
                    wi_alt = random.randint(wi_min, wi_max)
                else:
                    wi_alt = wi - 1
                #
                if type==3 and wi_alt>2:
                    print("error in decrease", wi_alt)
                #
            #
        #
        return wi_alt
        
    def undo_attack_2(self, attack_list):
        r = self._r
       
        llist = []
        for i in attack_list:
            w = self.w_list[i]
            layer = r.get_layer_at(w.li)
            layer.set_weight_index(w.ni, w.ii, w.wi)
            #w.wi_alt = w.wi
            self.w_list[i].wi_alt = w.wi
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
        
    def make_attack_list_2(self, attack_num, mode):
        r = self._r
        w_num = len(self.w_list)
        
        attack_list = []
        while len(attack_list)<=attack_num:
            attack_i = random.randrange(w_num)
            if attack_list.count(attack_i)>0:
                continue;
            #
            w = self.w_list[attack_i]
            type = w.type
            # mode : 0=random, 1=neighbor
            alt = self.shift_weight_2(type, w.wi, mode)
            self.w_list[attack_i].wi_alt = alt
            attack_list.append(attack_i)
        #
        return attack_list
    
    def multi_attack(self, ce, attack_num, temperature, asw=1):
        r = self._r
        
        attack_list = []
        mode = 1 # 0:random, 1:neighbor
        attack_list = self.make_attack_list_2(attack_num, mode)
        #
        # mpi
        #
        llist = []
        w_num = len(attack_list)
        for i in attack_list:
            w = self.w_list[i]
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
        #
        # mpi
        #
        ret = self.acceptance(ce, ce_alt, temperature, asw)
        if ret<=0:
            self.undo_attack_2(attack_list)
        else:
            ce = ce_alt
            for i in attack_list:
                self.w_list[i].wi = self.w_list[i].wi_alt
            #
        #
        return ce, ret
        
    def main_loop_logathic(self, idx, wtype, asw=1):
        r = self._r
        
        w_num = len(self.w_list)
        lv_min = 0
        lv_max = int(math.log(w_num/100, 2)) + 1 # 1 % max
        ce = self.evaluate()
        
        min = 200
        for temperature in range(lv_max, 0, -1):
            #self.delta_avg = ce * 0.1
            attack_num = 2 ** temperature
            hist = []
            atk_flag = True
            part = 0 # hit count
            num = 0  # loop count
            while atk_flag:
                self.delta_avg = ce * 0.1
                ce, ret = self.multi_attack(ce, attack_num, temperature, asw)
                if ret<0:
                    print("exit from while :", ce, ret)
                    break
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
                #
                # mpi
                #
                print("[%d:%d] %d [%s:%d] (%d, %d) %06f, ce:"
                        % (idx, temperature, num, wtype, attack_num, part, s, rate), ce)
                #
                if num>min:
                    if num>10000 or rate<0.01:
                        atk_flag = False
                    #
                #
            # while
            
            if ce<=0:
                print("exit from for")
                break
            #
        #
        r.save()
        
    def main_loop(self, idx, ce, wtype, temperature, total, minimum=0.1, debug=0, asw=1):
        r = self._r
        
        w_num = len(self.w_list)
        attack_num = 1
        ce = r.evaluate()
        print("ce:", ce)
        self.delta_avg = ce * 0.1
        t_min = minimum # 1.0, 0.1, 0.01
        lp = 0
        
        while temperature>t_min:
            num = 0
            self.delta_avg = ce*0.1
            while num<total:
                ce, ret = self.multi_attack(ce, attack_num, temperature, asw)
                if ce<0:
                    print(ret, "ce => zero")
                    return # -1
                #
                print(idx, "[%d/%d]"%(num, total), "T=%f"%(temperature), "(%d)"%(lp), "\t", ret, "\t", ce)
                #
                num += 1
                self.delta_avg = ce*0.1
            #
            lp = lp +1
            temperature = temperature*0.95 #0.90, 0.95
        #
        return ce
        #if self.mpi==False or self.rank==0:
        #    r.save()
        #
        
#
#
#
