#! /usr/bin/python
# -*- coding: utf-8 -*-
#

#
# LDNN : lesser's Deep Neural Network
#

import os, sys, time, math
from stat import *
import random
import copy
import math
import multiprocessing as mp
import numpy as np
import struct
import pickle
import pyopencl as cl
#
from PIL import Image
#
# LDNN Modules
#
import util
import package
import core
import gpu
import train
import test
#
sys.setrecursionlimit(10000)
#
#
#
def setup_autoencoder(r, size):
    mode = 0
    # 0 : input
    c = r.count_layers()
    input = core.InputLayer(c, size, size, None, r._gpu, mode)
    r.layers.append(input)
    r.input = input
    # 1 : enc
    c = r.count_layers()
    enc_1 = core.HiddenLayer(c, size, 128, input, r._gpu, mode)
    r.layers.append(enc_1)
    # 2 : enc
    c = r.count_layers()
    enc_2 = core.HiddenLayer(c, 128, 64, enc_1, r._gpu, mode)
    r.layers.append(enc_2)
    # 3 : intermediate
    c = r.count_layers()
    inter = core.HiddenLayer(c, 64, 32, enc_2, r._gpu, mode)
    r.layers.append(inter)
    # 4 : dec
    c = r.count_layers()
    dec_1 = core.HiddenLayer(c, 32, 64, inter, r._gpu, mode)
    r.layers.append(dec_1)
    # 5 : dec
    c = r.count_layers()
    dec_2 = core.HiddenLayer(c, 64, 128, dec_1, r._gpu, mode)
    r.layers.append(dec_2)
    # 3 : output
    c = r.count_layers()
    output = core.RegressionOutputLayer(c, 128, size, dec_2, r._gpu, mode)
    r.layers.append(output)
    r.output = output
    #
    r.set_evaluate_mode(1)

def setup_cnn(r):
    # 0 : input 28 x 28 x 1 = 784
    c = r.count_layers()
    input = core.InputLayer(c, self._image_size, self._image_size, None, my_gpu)
    r.layers.append(input)
    # 1 : CNN 28 x 28 x 1 > 28 x 28 x 3
    c = r.count_layers()
    cnn_1 = core.Conv_4_Layer(c, 28, 28, 1, 3, input, my_gpu)
    r.layers.append(cnn_1)
    # 2 : max 28 x28 x 3 > 14 x 14 x 3
    c = r.count_layers()
    max_1 = core.MaxLayer(c, 3, 28, 28, cnn_1, my_gpu)
    r.layers.append(max_1)
    # 3 : CNN 14 x 14 x 3 > 14 x 14 x 6
    c = r.count_layers()
    cnn_2 = core.Conv_4_Layer(c, 14, 14, 3, 6, max_1, my_gpu)
    r.layers.append(cnn_2)
    # 4 : max 14 x 14 x 6 > 7 x 7 x 6
    c = r.count_layers()
    max_2 = core.MaxLayer(c, 6, 14, 14, cnn_2, my_gpu)
    r.layers.append(max_2)
    # 5 : hidden : (7 x 7 x 160 X 64 = 784 x 64
    c = r.count_layers()
    hidden_1 = core.HiddenLayer(c, 294, 64, max_2, my_gpu)
    r.layers.append(hidden_1)
    # 6 : hidden : 64 x 64
    c = r.count_layers()
    hidden_2 = core.HiddenLayer(c, 64, 64, hidden_1, my_gpu)
    r.layers.append(hidden_2)
    # 7 : output : 64 x 10
    c = r.count_layers()
    output = core.OutputLayer(c, 64, 10, hidden_2, my_gpu)
    r.layers.append(output)

def setup_fc(r, size):
    print("setup_fc(%d)" % (size))
    mode = 0
    c = r.count_layers()
    input = core.InputLayer(c, size, size, None, r._gpu, mode)
    r.layers.append(input)
    # 1 : hidden : 28 x 28 x 1 = 784
    c = r.count_layers()
    hidden_1 = core.HiddenLayer(c, size, 64, input, r._gpu, mode)
    r.layers.append(hidden_1)
    # 2 : hidden : 64
    c = r.count_layers()
    hidden_2 = core.HiddenLayer(c, 64, 64, hidden_1, r._gpu, mode)
    r.layers.append(hidden_2)
    # 3 : hidden : 64
    c = r.count_layers()
    hidden_3 = core.HiddenLayer(c, 64, 64, hidden_2, r._gpu, mode)
    r.layers.append(hidden_3)
    # 3 : output
    c = r.count_layers()
    output = core.OutputLayer(c, 64, 10, hidden_3, r._gpu, mode)
    r.layers.append(output)

def setup_dnn(my_gpu, path, config=0, mode=0):
    r = core.Roster(mode)
    r.set_gpu(my_gpu)
    r.set_path(path)
    #
    if config==0: # fc
        setup_fc(r, 28*28)
    elif config==1: # cnn
        setup_cnn(r)
    elif config==2: # auto-encoder
        setup_autoencoder(r, 28*28)
    #
    if os.path.isfile(path):
        r.import_weight(path)
    else:
        r.init_weight()
        r.export_weight(path)
    #
    if my_gpu:
        r.update_weight()
    #
    return r
    
    
def save_array_to_png(array, w, h, path): # uint8, 1d, RGB
    data_in = np.reshape(array, (3, w*h))
    r = data_in[0]
    g = data_in[1]
    b = data_in[2]
    
    data = np.zeros((w*h, 3), dtype=np.uint8)
    
    for i in range(w*h):
        data[i][0] = r[i]
        data[i][1] = g[i]
        data[i][2] = b[i]
    #
    data = np.reshape(data, (h, w, 3)) # (2048, 1536, 4)
    pimg = Image.fromarray(data)
    pimg.save(path)

def save_array_gray_to_png(array, w, h, path): # uint8, 1d
    data = np.reshape(array, (h, w))
    pimg = Image.fromarray(data)
    pimg.save(path)

def main():
    argvs = sys.argv
    argc = len(argvs)
    print(argc)
    #
    if argc==7:
        pass
    else:
        print("error in sh")
        return 0
    #
    platform_id = int(argvs[1])
    device_id = int(argvs[2])
    package_id = int(argvs[3])
    config = int(argvs[4])
    mode = int(argvs[5])
    batch_size = int(argvs[6])
    batch_offset = 0
    #
    print("platform_id=%d" % (platform_id))
    print("device_id=%d" % (device_id))
    print("package_id=%d" % (package_id))
    print("config=%d" % (config))
    print("mode=%d" % (mode))
    print("batch_size=%d" % (batch_size))
    #
    my_gpu = gpu.Gpu(platform_id, device_id)
    my_gpu.set_kernel_code()
    #
    pack = package.Package(package_id)
    pack.load_batch()
    #
    r = pack.setup_dnn(my_gpu, config, mode)
    if package_id==0 or package_id==1: # MNIST, cifar-10
        r.set_scale_input(1)
    #
    r.set_path("../ldnn_config/cifar-10-batches-py/wi.csv")
    r.load()
    r.update_weight()

    if mode==0: # train
        print("batch_offset=%d" % (batch_offset))
        t = train.Train(r)
        data_size = pack._image_size
        num_class = pack._num_class
        train_data_batch = pack._train_image_batch
        train_label_batch = pack._train_label_batch
        t.set_batch(data_size, num_class, train_data_batch, train_label_batch, batch_size, batch_offset)
        #
        t.loop()
    if mode==4: # train
        t = train.Train(r)
        data_size = pack._image_size
        t.stochastic_loop(pack, data_size, batch_size, 1)
    elif mode==1: # test
        test.test_n(r, pack, 100)
    elif mode==2: # denomi
        test.test_n(r, pack, 500)
        
        r.denominate()
        r.update_weight()
        path = pack.save_path()
        r.export_weight(path)
        
        test.test_n(r, pack, 500)
        
    elif mode==3: # test for auto-encoder
        t = train.Train(r)
        data_size = pack._image_size
        num_class = pack._num_class
        train_data_batch = pack._train_image_batch
        train_label_batch = pack._train_label_batch
        t.set_batch(data_size, num_class, train_data_batch, train_label_batch, 10, 0)
        r.propagate()
        r._gpu.copy(r.output._output_array, r.output._gpu_output)
        #print(r.output._output_array)
        #print(r.output._output_array.shape)
        
        array_in = train_data_batch[0]
        array_in = np.array(array_in, dtype=np.uint8)
        #save_array_to_png(array_in, 32, 32, "./in.png")
        save_array_gray_to_png(array_in, 28, 28, "./in.png")

        array_out = r.output._output_array[0]*255
        array_out = np.array(array_out, dtype=np.uint8)
        #save_array_to_png(array_out, 32, 32, "./test.png")
        save_array_gray_to_png(array_out, 28, 28, "./test.png")
    else:
        print("mode error : %d" % (mode))
        return 0
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
