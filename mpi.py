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
        if self.mode_e==0: # classification
            ce = r.evaluate()
        elif self.mode_e==1: # regression
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
        total = 0
        
        for j in range(n):
            for lv in range(lv_min, lv_max+1):
                div = 2**lv
                total = 0
                for i in range(atk - int(atk*atk_r*lv)):
                    ce, ret = self.multi_attack(ce, w_list, 1, div)
                    total += ret
                    if self._rank==0:
                        print(m, wtype, "[", j, "] lv", lv, div, "i", i, "ce", ce, total)
                    #
                #
            #
            
            for lv in range(lv_max, -1, -1):
                div = 2**lv
                total = 0
                for i in range(atk - int(atk*atk_r*lv)):
                    ce, ret = self.multi_attack(ce, w_list, 0, div)
                    total += ret
                    if self._rank==0:
                        print(m, wtype, "[", j, "] lv", lv, div, "i", i, "ce", ce, total)
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
