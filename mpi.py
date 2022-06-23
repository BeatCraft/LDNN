#! /usr/bin/python
# -*- coding: utf-8 -*-
#
import os
import sys
import time
import math
#
from mpi4py import MPI
#import cupy as cp
#import cupyx

import util
import core
import dgx
import train

sys.setrecursionlimit(10000)

class worker(object):
    def __init__(self, com, rank, size, r):
        self._com = com
        self._rank = rank
        self._size = size
        self.r = r
        self.mode = 0
        self.mode_w = 0
        self.mode_e = 0
        self.mse_idx = 2
        
        self.train = train.Train(self.r)

    def evaluate(self):
        r = self.r

        ce = 0.0
        if self.mode_e==0 or self.mode_e==1: # classification
            ce = r.evaluate()
        elif self.mode_e==2 or self.mode_e==3: # CNN single
            r.propagate()
            layer = r.get_layer_at(mse_idex)
            ent = layer.mse(0)
        elif self.mode_e==4: # regression
            r.propagate()
            r._gpu.mse(r.output._gpu_output, r.input._gpu_output, r._gpu_entropy, self._data_size, self._bacth_size)
            r._gpu.copy(r._batch_cross_entropy, r._gpu_entropy)
            ce = np.sum(r._batch_cross_entropy)/np.float32(batch_size)
        else:
            print("evaluate() N/A")
        #

        ce_list = self._com.gather(ce, root=0)
        sum = 0.0
        if self._rank==0:
            for i in ce_list:
                sum = sum + i
            #
            avg = sum/float(self._size)
        #
        if self._rank==0:
            ce_avg_list = [avg]*self._size
        else:
            ce_avg_list = None
        #
        ce_avg = self._com.scatter(ce_avg_list, root=0)
        return ce_avg

    def multi_attack(self, ce, w_list, mode, div):
        r = self.r
        #
        if self._rank==0:
            attack_list = self.train.make_attack_list(div, mode, w_list)
            alt_list = []
            for i in attack_list:
                w = w_list[i]
                alt_list.append(w.wi_alt)
                #print("w.wi_alt=", w.wi_alt)
            #
            #print(alt_list)
        else:
            attack_list = []
            alt_list = []
        #
        attack_list = self._com.bcast(attack_list, root=0)
        alt_list = self._com.bcast(alt_list, root=0)

        #print(alt_list)

        idx = 0
        for i in attack_list:
            w = w_list[i]
            w.wi_alt = alt_list[idx]
            #try:
            #    w.wi_alt = alt_list[idx]
            #except:
            #    print("debug: ", self._rank, i)
            #    exit()
            #
            idx += 1
        #

        llist = []
        for i in attack_list:
            w = w_list[i]
            #print(w.li, w.ni, w.ii, w.wi, w.wi_alt)
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
        if ce_alt<ce:
            ce = ce_alt
            ret = 1
            for i in attack_list:
                w = w_list[i]
                w.wi = w.wi_alt
            #
        else:
            self.train.undo_attack(w_list, attack_list)
        #

        return ce, ret
        
    #def loop_k(self, w_list, wtype, m, n=1):
    def loop_k(self, w_list, wtype, m, n=1, atk=50, atk_r = 0.05):
        r = self.r
        
        ce = self.evaluate()
        print(self._rank, ce)
        ret = 0
        w_num = len(w_list)
        d = 100
        lv_min = 0
        lv_max = int(math.log(w_num/d, 2)) + 1
        #atk = 50
        
        
        for j in range(n):
            total = 0
            #for lv in range(lv_min, lv_max+1):
            #    div = 2**lv
            #    total = 0
            #    for i in range(atk - int(atk*atk_r*lv)):
            #        ce, ret = self.multi_attack(ce, w_list, 1, div)
            #        total += ret
            #        if self._rank==0:
            #            print(m, wtype, "[", j, "] lv", lv, div, "i", i, "ce", ce, total)
            #        #
            #    #
            #
           
            for lv in range(lv_max, -1, -1):
                div = 2**lv
                total = 0
                #for i in range(atk - int(atk*atk_r*lv)):
                i = 0
                while 1:
                    ce, ret = self.multi_attack(ce, w_list, 0, div)
                    total += ret
                    if self._rank==0:
                        print(m, wtype, "[", j, "] lv", lv, div, "i", i, "ce", ce, total)
                    #
                    
                    if i>atk:
                        if float(total)/float(i)<0.1:
                            break;
                        #
                    #
                    i += 1
                    if i>=1000:
                        break;
                    #
                # while
            #
            if self._rank==0:
                r.save()
            #
        #
        
        return 0
        
    def loop_sa(self, w_list, wtype, m, n=1, atk=50, atk_r = 0.05):
        r = self.r
        
        ce = self.evaluate()
        print(self._rank, ce)
        ret = 0
        w_num = len(w_list)
        d = 100
        lv_min = 0
        lv_max = int(math.log(w_num/d, 2)) + 1
        
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
                    if self._rank==0:
                        print(m, wtype, "[", j, "] lv", lv, div, "(", i, ")", "ce", ce, total)
                    #
                    if k>atk:
                        tr = float(total)/float(i)
                        pr = float(part)/float(k)
                        if tr<0.05 and pr<0.05:
                            flag = 0
                            #
                            # refreash here if needed
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
            #
            if self._rank==0:
                r.save()
            #
        #
        return 0
        
    #def multi_attack_sa4(self, ce, w_list, mode, div):
    def multi_attack_sa4(self, ce, w_list, mode=1, div=0, pbty=0):
        r = self.r
        #
        if self._rank==0:
            attack_list = self.train.make_attack_list(div, mode, w_list)
            alt_list = []
            for i in attack_list:
                w = w_list[i]
                alt_list.append(w.wi_alt)
            #
        else:
            attack_list = []
            alt_list = []
        #
        attack_list = self._com.bcast(attack_list, root=0)
        alt_list = self._com.bcast(alt_list, root=0)

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
        
        if ce_alt<ce:
            ce = ce_alt
            ret = 1
            for i in attack_list:
                w = w_list[i]
                w.wi = w.wi_alt
            #
        else:
            delta = abs(ce_alt - ce)
            if pbty>3: # 2, 3?
                diff = math.log10(delta) - math.log10(ce)
                limit = -1.0 # - 1.0/(1.0+math.log(div, 2))
                print(diff, limit)
                if (diff < limit):
                    #
                    print(self._rank, "RESET", diff, limit)
                    #
                    ce = ce_alt
                    ret = 1
                    pbty = 0
                #
            #
        #

        if ret==0:
            self.train.undo_attack(w_list, attack_list)
        #
        return ce, ret, pbty
        
    def loop_sa4(self, w_list, wtype, n=1, atk=1000, atk_r = 0.01):
        r = self.r
        
        ce = self.evaluate()
        print(self._rank, ce)
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
                #i = 0
                #k = 0
                #flag = 1
                #while flag:
                for i in range(atk):
                    #ce, ret = self.multi_attack(ce, w_list, 0, div)
                    ce, ret, pbty = self.multi_attack_sa4(ce, w_list, 0, div, pbty)
                    total += ret
                    part += ret
                    if self._rank==0:
                        #print(m, wtype, "[", j, "] lv", lv, div, "(", i, ")", "ce", ce, total)
                        print(wtype, "[", j, "]", wtype, lv,"/", lv_max, "|", div, "(", i, ")", "ce", ce, part, total, pbty)
                    #
                #
                rate = float(part)/float(atk)
                if rate<atk_r:
                    lv_max -= 1
                    pbty += 1
                    if lv_max<2:
                        lv_max = 2
                        #pbty += 1
                    #
                #
            #
            if self._rank==0:
                r.save()
            #
            #if j>0 and j%100==0:
            #    lv_max = int(math.log(w_num/d, 2)) + 1
            #
        #
        return 0
        
    def loop_sa5(self, idx, w_list, wtype, loop=1, min=200):
        r = self.r
        
        ce = self.evaluate()
        print(self._rank, ce)
        ret = 0
        w_num = len(w_list)
        d = 100
        lv_min = 0
        lv_max = int(math.log(w_num/d, 2)) + 1
        
        pbty = 0
                
        for j in range(loop):
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
                    rate = float(s)/float(num)
                    if self._rank==0:
                        print(wtype, idx, "[", j, "]", wtype, lv,"/", lv_max, "|", div, "(", num, ")", "ce", ce, part, total, s, rate)
                    #
                    if num>min:
                        if num>10000 or rate<0.01:
                            atk_flag = False
                        #
                    #
                #
            #
            if self._rank==0:
                r.save()
            #
        #
        return 0
        
def main():
    argvs = sys.argv
    argc = len(argvs)
    #
    return 0
#
#
#
if __name__=='__main__':
    print(">> start")
    sts = main()
    print(">> end")
    print("\007")
    sys.exit(sts)
#
#
#
