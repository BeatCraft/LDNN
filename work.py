#! /usr/bin/python
# -*- coding: utf-8 -*-
#
import os
import sys
import time
import math
import random
#
from mpi4py import MPI
#import cupy as cp
#import cupyx

import util
import core
#import dgx
import train

sys.setrecursionlimit(10000)

def output(path, msg):
    with open(path, 'a') as f:
        print(msg, file=f)
    #

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

    def log(self, path, msg):
        with open(path, 'a') as f:
            print(msg, file=f)
        #
    
    def multi_attack_sa5(self, ce, w_list, mode=1, div=0):
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
            delta = ce_alt - ce
            if self._rank==0:
                hit = self.train.random_hit(delta, div)
            else:
                hit = 0
            #
            hit = self._com.bcast(hit, root=0)
            if hit==1:
                ret = 1
                ce = ce_alt
            else:
                ret = 0
                #self.undo_attack(w_list, attack_list)
            #
        #

        if ret==0:
            self.train.undo_attack(w_list, attack_list)
        #
        return ce, ret
        
    def loop_sa5(self, idx, w_list, wtype, loop=1, min=200, base=2.0, debug=0):
        r = self.r
        
        ce = self.evaluate()
        print(self._rank, ce)
        ret = 0
        w_num = len(w_list)
        d = 100
        lv_max = int(math.log(w_num/d, base)) + 1
        lv_min = int(lv_max/10) + 1
        
        for j in range(loop):
            total = 0
            rec = [0] * (lv_max+1)
            for lv in range(lv_max, -1, -1):
                div = int(base**lv)
                part = 0
                atk_flag = True
                num = 0
                hist = []
                
                while atk_flag:
                    ce, ret = self.multi_attack_sa5(ce, w_list, 0, div) # lv
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
                        print(wtype, idx, "[", j, "]", lv,"/", lv_max, "|", div, "(", num, ")", "ce", ce, part, total, s, rate)
                    else:
                        pass
                    #
                    if num>min:
                        if num>10000 or rate<0.01:
                            atk_flag = False
                        #
                    #
                # while
                rec[lv] = part
                print("rank : ", self._rank)
            # for lv
            print(self._rank, lv_max, lv_min)
            if self._rank==0:
                r.save()
                if debug==1:
                    log = "%d, %s" % (j+1, '{:.10g}'.format(ce))
                    output("./log.csv", log)
                    spath = "./wi/wi-%04d.csv" % (j+1)
                    r.save_as(spath)
                #
            else:
                print("rank : ", self._rank)
            #
            if lv_max>lv_min and rec[lv_max]==0:
                lv_max = lv_max -1
            #
        #
        return ce


    def multi_attack_sa_cnn(self, ce, fc_w_list, cnn_w_list, fc_atk_num, cnn_atk_num):
        r = self.r
        mode = 0
        #
        if self._rank==0:
            #attack_list = self.train.make_attack_list(div, mode, w_list)
            fc_attack_list = self.train.make_attack_list(fc_atk_num, mode, fc_w_list)
            cnn_attack_list = self.train.make_attack_list(cnn_atk_num, mode, cnn_w_list)
            #attack_list = fc_attack_list + cnn_attack_list
            
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
        
    def loop_sa_cnn(self, loop, fc_w_list, cnn_w_list, debug=0):
        r = self.r
        ce = self.evaluate()
        print(self._rank, ce)
        min=200
        
        fc_num = len(fc_w_list)
        cnn_num = len(cnn_w_list)
        
        ret = 0
        t_max = 10
        for t in range(t_max, -1, -1):
            total = 0
            rec = [0] * (t_max+1)
            hist = []
            #fc_atk_num = random.randint(0, 100)#int(fc_num*0.01*(float(t)/100.0))
            #cnn_atk_num = random.randint(0, 10)#int(cnn_num*0.01*(float(t)/100.0))
            #if fc_atk_num<1:
            #    fc_atk_num = 1
            #
            #if cnn_atk_num<1:
            #    cnn_atk_num = 1
            #
            atk_flag = True
            part = 0
            num = 0
            while atk_flag:
                if self._rank==0:
                    fc_atk_num = t#random.randint(1, 10)
                    cnn_atk_num = 1#random.randint(1, 10)
                    fc_attack_list = self.train.make_attack_list(fc_atk_num, 0, fc_w_list)
                    cnn_attack_list = self.train.make_attack_list(cnn_atk_num, 0, cnn_w_list)
                    fc_alt_list = []
                    for i in fc_attack_list:
                        w = fc_w_list[i]
                        fc_alt_list.append(w.wi_alt)
                    #
                    cnn_alt_list = []
                    for i in cnn_attack_list:
                        w = cnn_w_list[i]
                        cnn_alt_list.append(w.wi_alt)
                    #
                else:
                    fc_attack_list = []
                    cnn_attack_list = []
                    fc_alt_list = []
                    cnn_alt_list = []
                #
                fc_attack_list = self._com.bcast(fc_attack_list, root=0)
                cnn_attack_list = self._com.bcast(cnn_attack_list, root=0)
                fc_alt_list = self._com.bcast(fc_alt_list, root=0)
                cnn_alt_list = self._com.bcast(cnn_alt_list, root=0)
                
                idx = 0
                for i in fc_attack_list:
                    w = fc_w_list[i]
                    w.wi_alt = fc_alt_list[idx]
                    idx += 1
                #
                
                idx = 0
                for i in cnn_attack_list:
                    w = cnn_w_list[i]
                    w.wi_alt = cnn_alt_list[idx]
                    idx += 1
                #
                
                llist = []
                for i in fc_attack_list:
                    w = fc_w_list[i]
                    layer = r.get_layer_at(w.li)
                    layer.set_weight_index(w.ni, w.ii, w.wi_alt)
                    if w.li in llist:
                        pass
                    else:
                        llist.append(w.li)
                    #
                #
                
                for i in cnn_attack_list:
                    w = cnn_w_list[i]
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
                    for i in fc_attack_list:
                        w = fc_w_list[i]
                        w.wi = w.wi_alt
                    #
                    for i in cnn_attack_list:
                        w = cnn_w_list[i]
                        w.wi = w.wi_alt
                    #
                else:
                    self.train.undo_attack(fc_w_list, fc_attack_list)
                    self.train.undo_attack(cnn_w_list, cnn_attack_list)
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
                if self._rank==0:
                    #print("[", idx,"]", t, "/", t_max, "|", "(", num, ")", "[fc:", fc_atk_num,"]", "[cnn:", cnn_atk_num,"]", "ce", ce, part, total, s)
                    print("[%d] %d/%d (%d) [fc:%03d][cnn:%03d] ce %09f (%d, %d, %d)" %(loop, t, t_max, num, fc_atk_num, cnn_atk_num, ce, part, total, s))
                #
                if num>min:
                    if num>10000 or rate<0.01:
                        atk_flag = False
                    #
                #
            # while
            rec[t] = part
        # for t
        if self._rank==0:
            r.save()
            if debug==1:
                log = "%d, %s" % (loop+1, '{:.10g}'.format(ce))
                output("./log.csv", log)
                spath = "./wi/wi-%04d.csv" % (loop+1)
                r.save_as(spath)
            #
        #
        return ce
        
    def loop_sa_20(self, loop, w_list, wtype, debug=0):
        r = self.r
        ce = self.evaluate()
        print(ce)
        
        min=200
        w_num = len(w_list)
        ret = 0
        t_max = 100#1000
        for t in range(t_max, 0, -1): #-10
            total = 0
            rec = [0] * (t_max+1)
            hist = []

            atk_flag = True
            part = 0
            num = 0
            while atk_flag:
                if self._rank==0:
                    atk_num = random.randint(1, t)
                    attack_list = self.train.make_attack_list(atk_num, 0, w_list)
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
                else:
                    attack_list = []
                    alt_list = []
                #
                attack_list = self._com.bcast(attack_list, root=0)
                alt_list = self._com.bcast(alt_list, root=0)
                #
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
                    if self._rank==0:
                        hit = self.train.random_hit(delta, t)
                    else:
                        hit = 0
                    #
                    hit = self._com.bcast(hit, root=0)
                    if hit==1:
                        ret = 1
                        ce = ce_alt
                    else:
                        ret = 0
                        self.train.undo_attack(w_list, attack_list)
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
                if self._rank==0:
                    print("[%d] %d/%d (%d) [%s:%03d] ce %09f (%d, %d, %d) %f" %(loop, t, t_max, num, wtype, atk_num, ce, part, total, s, rate))
                #
                if num>min:
                    if num>10000 or rate<0.01:
                        atk_flag = False
                    #
                #
            # while
            rec[t] = part
        # for t
        if self._rank==0:
            r.save()
            if debug==1:
                log = "%d, %s" % (loop+1, '{:.10g}'.format(ce))
                output("./log.csv", log)
                spath = "./wi/wi-%04d.csv" % (loop+1)
                r.save_as(spath)
            #
        #
        return ce
    
    #
    # basic loop
    #
    def loop_sa_t(self, ce, temperature, attack_num, w_list, wtype, debug=0):
        r = self.r
        w_num = len(w_list)
        ret = 0
        part = 0 # hit count
        num = 0 # loop count
        #total = 0
        min = 200
        hist = []
        
        atk_flag = True
        
        while atk_flag:
            if self._rank==0:
                attack_list = self.train.make_attack_list(attack_num, 0, w_list)
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
            else:
                attack_list = []
                alt_list = []
            #
            attack_list = self._com.bcast(attack_list, root=0)
            alt_list = self._com.bcast(alt_list, root=0)
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
                if self._rank==0:
                    hit = self.train.random_hit(delta, temperature)
                else:
                    hit = 0
                #
                hit = self._com.bcast(hit, root=0)
                if hit==1:
                    ret = 1
                    ce = ce_alt
                else:
                    ret = 0
                    self.train.undo_attack(w_list, attack_list)
                #
            #
                
            num += 1
            #total += ret
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
            if self._rank==0:
                print("[%d] %d [%s:%d] ce %09f (%d, %d) %06f"
                    % (temperature, num, wtype, attack_num, ce, part, s, rate))
            #
            if num>min:
                if num>10000 or rate<0.01:
                    atk_flag = False
                #
            #
        # while
        
        #if self._rank==0:
        #    r.save()
        #    if debug==1:
        #        log = "%d, %s" % (loop+1, '{:.10g}'.format(ce))
        #        output("./log.csv", log)
        #        spath = "./wi/wi-%04d.csv" % (loop+1)
        #        r.save_as(spath)
        #    #
        #
        return ce


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
