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
        
    def set_batch(self, data_size, num_class, train_data_batch, train_label_batch,  batch_size, batch_offset):
        self._bacth_size = batch_size
        self._data_size = data_size
        #
        r = self._r
        r.prepare(batch_size, data_size, num_class)
        r.set_batch(data_size, num_class, train_data_batch, train_label_batch,  batch_size, batch_offset)
    
    def set_path(path):
        self._path = path
    
    def evaluate(self, mode=0):
        r = self._r
        ce = r.evaluate()
        return ce
        
        r.propagate()
        #
        if mode==0:
            ce = r.get_cross_entropy()
        elif mode ==1:
            r._gpu.mse(r.output._gpu_output, r.input._gpu_output, r._gpu_entropy, self._data_size, self._bacth_size)
            r._gpu.copy(r._batch_cross_entropy, r._gpu_entropy)
            ce = np.sum(r._batch_cross_entropy)/np.float32(batch_size)
        #
        return ce
    
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
                        w_list.append((li, ni, ii))
                    #
                #
            #
        #
        return w_list

    def weight_ops(self, attack_i, mode, w_list):
        r = self._r
        w = w_list[attack_i]
        li = w[0]
        ni = w[1]
        ii = w[2]
        layer = r.get_layer_at(li)
        wi = layer.get_weight_index(ni, ii)
        wi_alt = wi
        maximum = core.WEIGHT_INDEX_MAX
        minimum = core.WEIGHT_INDEX_MIN
        #
        if mode>0: # heat
            if wi<maximum:
                wi_alt = wi + 1
            #
        else:
            if wi>minimum:
                wi_alt = wi - 1
            #
        #
        return wi, wi_alt
        
    def make_attack_list(self, div, mode, w_list):
        r = self._r
        w_num = len(w_list)
        attack_num = int(w_num/div)
        if attack_num<1:
            attack_num = 1
        #
        attack_list = []
        for i in range(attack_num*10):
            if i>=attack_num:
                break
            #
            attack_i = random.randrange(w_num)
            w = w_list[attack_i]
            li = w[0]
            ni = w[1]
            ii = w[2]
            wi, wi_alt = self.weight_ops(attack_i, mode, w_list)
            if wi!=wi_alt:
                attack_list.append((li, ni, ii, wi, wi_alt))
            #
        #
        return attack_list

    def undo_attack(self, attack_list):
        r = self._r
        for wt in attack_list:
            li = wt[0]
            ni = wt[1]
            ii = wt[2]
            wi = wt[3]
            wi_alt = wt[4]
            #
            layer = r.get_layer_at(li)
            layer.set_weight_index(ni, ii, wi)
        #
        c = r.count_layers()
        for li in range(c):
            layer = r.get_layer_at(li)
            layer.update_weight()
        #
        
    def multi_attack(self, ce, w_list, mode=1, div=0):
        r = self._r
        attack_list = self.make_attack_list(div, mode, w_list)
        w_num = len(attack_list)
        #print(len(attack_list))
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
            self.undo_attack(attack_list)
        #
        return ce, ret

    def w_loop(self, c, n, d, ce, w_list, lv_min, lv_max, label):
        r = self._r
        level = lv_min
        mode = 1
        #
        for j in range(n):
            div = float(d)*float(2**(level))
            #
            cnt = 0
            for i in range(100):
                ce, ret = self.multi_attack(ce, w_list, 1, div)
                cnt = cnt + ret
                print("[%d|%s] %d : H : %d : %f, %d (%d, %d) %d (%d, %d)" % (c, label, j, i, ce, level, lv_min, lv_max, cnt, 2**(level), int(len(w_list)/div)))
            #
            for i in range(100):
                ce, ret = self.multi_attack(ce, w_list, 0, div)
                cnt = cnt + ret
                print("[%d|%s] %d : C : %d : %f, %d (%d, %d) %d (%d, %d)" % (c, label, j, i, ce, level, lv_min, lv_max, cnt, 2**(level), int(len(w_list)/div)))
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
            #r.export_weight("./wi.csv")
            r.save()
            #path = "./test-%04d.png" % (j)
            #save_img(r, path)
        #
        return ce, lv_min, lv_max

    def loop(self, n=1):
        r = self._r
        w_list = None
        ce = 0.0
        ret = 0
        #
        ce = self.evaluate()
        print("CE starts with %f" % ce)
        #
        w_list = self.make_w_list([core.LAYER_TYPE_CONV_4, core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT])
        w_num = len(w_list)
        print(len(w_list))
        #
        d = 100
        lv_min = 0
        lv_max = int(math.log(w_num/d, 2)) + 1
        #
        for i in range(n):
            ce, lv_min, lv_min = self.w_loop(i, 500, d, ce, w_list, lv_min, lv_max, "all")
            r.save()
        #
        return 0
        
    def stochastic_loop(self, pack, dsize, bsize, n=1):
        r = self._r
        w_list = None
        ce = 0.0
        ret = 0
        #
        #ce = self.evaluate()
        #print("CE starts with %f" % ce)
        #
        w_list = self.make_w_list([core.LAYER_TYPE_CONV_4, core.LAYER_TYPE_HIDDEN, core.LAYER_TYPE_OUTPUT])
        w_num = len(w_list)
        print(len(w_list))
        #
        d = 100
        lv_min = 0
        lv_max = int(math.log(w_num/d, 2)) + 1
        
        train_data_batch = pack._train_image_batch
        train_label_batch = pack._train_label_batch
        num_class = pack._num_class
        tbs = len(train_data_batch)
        num = int(tbs/dsize)
        for i in range(num):
            batch_offset = bsize*i
            print("%d, %d, %d" % (dsize, bsize, i))
            self.set_batch(dsize, num_class, train_data_batch, train_label_batch, bsize, batch_offset)
            #r.set_batch(dsize, num_class, train_data_batch, train_label_batch, bsize, batch_offset)
            ce = self.evaluate()
            print("CE starts with %f" % ce)
            ce, lv_min, lv_min = self.w_loop(i, 1, d, ce, w_list, lv_min, lv_max, "all")
            r.save()
        #
        return 0
    
    

#def save_img(r, path):
#    r._gpu.copy(r.output._output_array, r.output._gpu_output)
#    print(r.output._output_array)
#    imgArray = np.reshape(r.output._output_array, (480, 640))
#    imgArray = imgArray*255
#    imgArray = np.array(imgArray, dtype=np.uint8)
#    print(imgArray)
#    pilImg = Image.fromarray(np.uint8(imgArray))
#    pilImg.save(path)
