#! /usr/bin/python
# -*- coding: utf-8 -*-
#
import os
import sys
import time
import math
#
from mpi4py import MPI
import cupy as cp
import cupyx

#
# LDNN Modules
#

import core
#import util
import package
import gdx
#import gpu
import train
#import test

sys.setrecursionlimit(10000)

#MINI_BATCH_SIZE = [7500, 7500, 7500, 7500, 7500, 7500, 7500, 7500, 7500, 7500]
#MINI_BATCH_START = [0, 7500, 15000, 22500, 30000, 37500, 45000, 52500]
MINI_BATCH_SIZE = [6250, 6250, 6250, 6250, 6250, 6250, 6250, 6250, 6250, 6250]
MINI_BATCH_START = [0, 6250, 12500, 18750, 25000, 31250, 37500, 43750]
#MINI_BATCH_SIZE = [2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500, 2500]
#MINI_BATCH_START = [0, 2500, 5000, 7500, 10000, 12500, 15000, 17500]

class worker(object):
    def __init__(self, com, rank, size, package_id, config_id, path):
        self._com = com
        self._rank = rank
        self._size = size
        self._processor_name = MPI.Get_processor_name()

        cp.cuda.Device(self._rank).use()
        #x = cp.arange(1024, dtype=cp.float32)
        #y = cp.zeros(1024, dtype=cp.float32)
        #y = x*5
        #print(y)

        my_gpu = gdx.Gdx(self._rank)

        pack = package.Package(package_id)
        pack.load_batch()

        self.r = pack.setup_dnn(my_gpu, config_id)
        self.r.set_scale_input(1)
        self.r.set_path(path)
        self.r.load()
        self.r.update_weight()

        train_data_batch = pack._train_image_batch
        print(len(train_data_batch))
        train_label_batch = pack._train_label_batch
        data_size = pack._image_size
        num_class = pack._num_class
        batch_offset = MINI_BATCH_START[self._rank]
        batch_size = MINI_BATCH_SIZE[self._rank]
        self.r.prepare(batch_size, data_size, num_class)
        self.r.set_batch(data_size, num_class, train_data_batch, train_label_batch, batch_size, batch_offset)
        self.train = train.Train(self.r)

    def evaluate(self):
        ce = self.r.evaluate()
        
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

    def multi_attack(self, ce, w_list, mode=1, div=0):
        r = self.r
        #
        if self._rank==0:
            attack_list = self.train.make_attack_list(div, mode, w_list)
        else:
            attack_list = []
        #
        attack_list = self._com.bcast(attack_list, root=0)
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
        ce_alt = self.evaluate()
        ret = 0
        if ce_alt<ce:
            ce = ce_alt
            ret = 1
        else:
            self.train.undo_attack(attack_list)
        #
        return ce, ret

    def w_loop(self, c, n, d, ce, w_list, lv_min, lv_max, label):
        level = lv_min
        mode = 1
        r = self.r

        for j in range(n):
            #
            div = float(d)*float(2**(level))
            #
            cnt = 0
            for i in range(50):
                ce, ret = self.multi_attack(ce, w_list, 1, div)
                cnt = cnt + ret
                #if self._rank==0:
                #    print("[%d|%s] %d : H : %d : %f, %d (%d, %d) %d (%d, %d)" %
                #            (c, label, j, i, ce, level, lv_min, lv_max, cnt, 2**(level), int(len(w_list)/div)))
                #
            #

            if self._rank==0:
                 print("[%d|%s] %d : H : %f, %d (%d, %d) %d (%d, %d)" % 
                         (c, label, j, ce, level, lv_min, lv_max, cnt, 2**(level), int(len(w_list)/div)))
            #

            for i in range(50):
                ce, ret = self.multi_attack(ce, w_list, 0, div)
                cnt = cnt + ret
                #if self._rank==0:
                #    print("[%d|%s] %d : C : %d : %f, %d (%d, %d) %d (%d, %d)" %
                #            (c, label, j, i, ce, level, lv_min, lv_max, cnt, 2**(level), int(len(w_list)/div)))
                #
            #
            if self._rank==0:
                print("[%d|%s] %d : C : %f, %d (%d, %d) %d (%d, %d)" % 
                        (c, label, j, ce, level, lv_min, lv_max, cnt, 2**(level), int(len(w_list)/div)))
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
        #
        return ce, lv_min, lv_max

    def loop(self, n=1):
        w_list = None
        ce = 0.0
        ret = 0

        ce = self.evaluate()
        #r = self.r

        #for i in range(n):
        #    ce = self.evaluate()
        #    print("[%d][%d]CE=%f" % (i, self._rank, ce))
        #
        if self._rank==0:
            w_list = self.train.make_w_list([core.LAYER_TYPE_CONV_4, core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT])
        else:
             w_list = []
        #
        w_list = self._com.bcast(w_list, root=0)
        w_num = len(w_list)
        print("rank=%d, w=%d" % (self._rank, w_num))
        d = 100
        lv_min = 0
        lv_max = int(math.log(w_num/d, 2)) + 1

        for i in range(n):
            ce, lv_min, lv_min = self.w_loop(i, 500, d, ce, w_list, lv_min, lv_max, "all")
            if self._rank==0:
                self.r.save()
            #
        #
        return 0

def main():
    argvs = sys.argv
    argc = len(argvs)
    #
    com = MPI.COMM_WORLD
    rank = com.Get_rank()
    size = com.Get_size()
    print("(%d, %d)" % (rank, size))

    package_id = 1  # 0 : MNIST, 1 : Cifar-10
    config_id = 0   # 0 : FC, 1 : CNN
    #
    wk = worker(com, rank, size, package_id, config_id, "./wi.csv")
    #ce = wk.evaluate()
    #print("[%d] %f" %(rank, ce))
    wk.loop(50)

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
