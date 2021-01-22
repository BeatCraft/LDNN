#! c:/Python26/python.exe
# -*- coding: utf-8 -*-

import os, sys, time, math
from stat import *
import random
import copy
import multiprocessing as mp
import pickle
import numpy as np
import csv

from PIL import Image

# LDNN Modules
import gpu
import util

#
#
#
sys.setrecursionlimit(10000)
#
# constant values
#
WEIGHT_SET_0 = [-1.0, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625, -0.0078125,
                0.0,
                0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]
WEIGHT_SET_1 = [-1.0, -0.5, -0.25, -0.125, 0, 0.125, 0.25, 0.5, 1.0]
WEIGHT_SET_2 = [-1.0, -0.5, -0.25, 0, 0.25, 0.5, 1.0]
#
WEIGHT_SET = WEIGHT_SET_1
WEIGHT_INDEX_SIZE = len(WEIGHT_SET)
WEIGHT_INDEX_ZERO = WEIGHT_INDEX_SIZE/2
WEIGHT_INDEX_MAX = WEIGHT_INDEX_SIZE-1
WEIGHT_INDEX_MIN = 0
#
WEIGHT_SET_CNN = [0.0, 0.125, 0.25, 0.5, 1.0]
#
#
#
class Weight:
    def __init__(self, li, ni, ii, wi):
        self._li = li
        self._ni = ni
        self._ii = ii
        self._wi = wi
        self._wi_alt = wi
    
    def set_all(self, li, ni, ii, wi):
        self._li = li
        self._ni = ni
        self._ii = ii
        self._wi = wi
    
    def set_index(self, li, ni, ii):
        self._li = li
        self._ni = ni
        self._ii = ii
    
    def set_wi(self, wi):
        self._wi_alt = self._wi
        self._wi = wi
    
    def alternate_wi(self):
        temp = self._wi
        self._wi = self._wi_alt
        self._wi_alt = temp
        return self._wi

    def get_all(self):
        return self._li, self._ni, self._ii, self._wi
    
    def get_index(self):
        return self._li, self._ni, self._ii

    def get_wi(self):
        return self._wi
#
#
#
class Node:
    def __init__(self):
        self._w_id_list = []

    def get_weight(self, i):
        return self._w_id_list[i]

    def add_weight(self, w_id):
        self._w_id_list.append(w_id)
#
#
#
LAYER_TYPE_INPUT   = 0
LAYER_TYPE_HIDDEN  = 1
LAYER_TYPE_OUTPUT  = 2
LAYER_TYPE_CONV_4  = 3
LAYER_TYPE_MAX    = 4 # MAX
#
class Layer(object):
    # i         : index of layers
    # type      : 0 = input, 1 = hidden, 2 = output
    # input : stimulus from a previous layer
    # num_input : number of inputs / outputs from a previous layer
    # node : neurons
    # num_node  : numbers of neurons in a layer
    def __init__(self, i, type, num_input, num_node, pre, gpu=None):
        self._pre = pre
        self._index = i
        self._type = type
        if gpu is not None:
            #print "GPU"
            self._gpu = gpu
        else:
            self._gpu = None
        #
        self._id = -1
        self._num_input = num_input
        self._num_node = num_node
        #
        self._node_marker = np.zeros( (self._num_node), dtype=np.int32)
        #
        self._num_update = 0
    
    def count_locked_weight(self):
        cnt = 0
        for ni in range(self._num_node):
            for ii in range(self._num_input):
                if self._weight_lock[ni][ii]==1:
                    cnt = cnt + 1
                #
            #
        #
        return cnt
        
    def count_weight(self):
        return self._num_node*self._num_input
        
    def get_pre_layer(self):
        return self._pre
        
    def set_marker_pre(self, ni, v):
        if self._pre:
            self._pre.set_marker(ni, v)
        #
            
    def set_marker(self, ni, v):
        self._node_marker[ni] = v

    def get_marker(self, ni):
        return self._node_marker[ni]

    def set_num_update(self, n):
        self._num_update = n
    
    def get_num_update(self):
        return self._num_update
        
    def prepare(self, batch_size):
        pass
    
    def get_num_node(self):
        return self._num_node
        
    def get_num_input(self):
        return self._num_input
    
    # gpu must be checked before this method is called
    def update_weight(self):
        pass

    def propagate(self, array_in, ni=-1, ii=-1, wi=-1, debug=0):
        pass
    
    def get_weight_index(self, ni, ii):
        return self._weight_index_matrix[ni][ii]
    
    def get_weight_property(self, ni, ii):
        return self._weight_property[ni][ii]
    
    def set_weight_property(self, ni, ii, p):
        self._weight_property[ni][ii] = p

    def reset_weight_property_all(self):
        for ni in range(self._num_node):
            for ii in range(self._num_input):
                self.set_weight_property(ni, ii, 0)
            #
        #

    def get_weight_lock(self, ni, ii):
        return self._weight_lock[ni][ii]
    
    def set_weight_lock(self, ni, ii, l):
        self._weight_lock[ni][ii] = l
  
    def get_weight_mbid(self, ni, ii):
        return self._weight_mbid[ni][ii]
    
    def set_weight_mbid(self, ni, ii, mbid):
        self._weight_mbid[ni][ii] = mbid
    
    def reset_weight_mbid(self):
        for ni in range(self._num_node):
            for ii in range(self._num_input):
                self.set_weight_mbid(ni, ii, 0)
            #
        #
    
    def assign_weight_mbid_random(self, max):
        for ni in range(self._num_node):
            for ii in range(self._num_input):
                mbid = random.randrange(max)
                self.set_weight_mbid(ni, ii, mbid)
            #
        #
    def unlock_weight_all(self):
        for ni in range(self._num_node):
            for ii in range(self._num_input):
                self.set_weight_lock(ni, ii, 0)
            #
        #
    
    def set_weight_index(self, ni, ii, wi): # wi : weight index
        self._weight_index_matrix[ni][ii] = wi
        self._weight_matrix[ni][ii] = WEIGHT_SET[wi]
    
    def init_weight_with_random_index(self):
        for ni in range(self._num_node):
            for ii in range(self._num_input):
                #wi = WEIGHT_INDEX_SIZE-1
                wi = random.randrange(WEIGHT_INDEX_SIZE)
                self.set_weight_index(ni, ii, wi)

    def export_weight_index(self):
        return self._weight_index_matrix.tolist()
    
    def import_weight_index(self, wi_list):
        self._weight_index_matrix = np.array(wi_list, dtype=np.int32).copy()
        for ni in range(self._num_node):
            for ii in range(self._num_input):
                self.set_weight_index(ni, ii, self._weight_index_matrix[ni][ii])
            #
        #

    def set_id(self, id):
        if id>=0:
            self._id = id

    def get_id(self):
        return self._id
    
    def get_type(self):
        return self._type
#
#
#
class InputLayer(Layer):
    def __init__(self, i, num_input, num_node, pre, gpu=None):
        print("InputLayer::__init__()")
        super(InputLayer, self).__init__(i, LAYER_TYPE_INPUT, num_input, num_node, pre, gpu)
        #
        
    def count_locked_weight(self):
        return 0
        
    def prepare(self, batch_size):
        self._batch_size = batch_size
        #
        self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.float32)
        if self._gpu:
            self._gpu_output = self._gpu.dev_malloc(self._output_array)
        #
        
    def propagate(self, array_in, ni=-1, ii=-1, wi=-1, debug=0):
        pass
    
    def get_num_update(self):
        return 0

    def get_weight_property(self, ni, ii):
        return 0
        
    def set_weight_property(self, ni, ii, wi):
        pass
    
    def reset_weight_property_all(self):
        pass
            
    def get_weight_lock(self, ni, ii):
        return 0

    def set_weight_lock(self, ni, ii, lock):
        pass
        
    def unlock_weight_all(self):
        pass
    
    def set_weight_index(self, ni, ii, wi):
        pass
        
    def get_weight_index(self, ni, ii):
        return 0
    
    def export_weight_index(self):
        return None

    def set_weight_mbid(self, ni, ii, mbid):
        pass
        
    def get_weight_mbid(self, ni, ii):
        pass
        
    def count_weight(self):
        return 0
#
#
#
class HiddenLayer(Layer):
    def __init__(self, i, num_input, num_node, pre, gpu=None):
        print("HiddenLayer::__init__()")
        super(HiddenLayer, self).__init__(i, LAYER_TYPE_HIDDEN, num_input, num_node, pre, gpu)
        #
        self._weight_index_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.int32)
        self._weight_lock = np.zeros( (self._num_node, self._num_input), dtype=np.int32)
        self._weight_mbid = np.zeros( (self._num_node, self._num_input), dtype=np.int32)
        self._weight_property = np.zeros( (self._num_node, self._num_input), dtype=np.int32)
        self._weight_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.float32)
        #
        if gpu:
            self._gpu_weight = self._gpu.dev_malloc(self._weight_matrix)
        #
    
    def prepare(self, batch_size):
        self._batch_size = batch_size
        #
        self._product_matrix = np.zeros( (self._batch_size, self._num_node, self._num_input), dtype=np.float32)
        
        if self._gpu:
            self._gpu_product = self._gpu.dev_malloc(self._product_matrix)
        #
        self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.float32)
        if self._gpu:
            self._gpu_output = self._gpu.dev_malloc(self._output_array)
    
    def update_weight(self):
        self._gpu.copy(self._gpu_weight, self._weight_matrix)
        
    def propagate(self, array_in, ni=-1, ii=-1, wi=-1, debug=0):
        stride_1 = self._num_node * self._num_input
        stride_2 = self._num_input
        if ni>=0: # alt
            self._gpu.multiple_x_by_w_batch_alt(array_in, self._gpu_weight, self._gpu_product,
                                                self._batch_size, stride_1, stride_2,
                                                self._num_input, self._num_node, ni, ii, WEIGHT_SET[wi])
        else: # propagation
            self._gpu.multiple_x_by_w_batch(array_in, self._gpu_weight, self._gpu_product,
                                            self._batch_size, stride_1, stride_2,
                                            self._num_input, self._num_node)
        #
        # sum
        #
        activation = 0 # 0 : relu, 1 : skip
        self._gpu.sum(self._gpu_product, self._gpu_output,
                        self._num_input, self._num_node, activation, self._batch_size)
        self._gpu.scale_layer(self._gpu_output, self._num_node, self._batch_size)
#
#
#
class OutputLayer(Layer):
    def __init__(self, i, num_input, num_node, pre, gpu=None):
        print("OutputLayer::__init__()")
        super(OutputLayer, self).__init__(i, LAYER_TYPE_OUTPUT, num_input, num_node, pre, gpu)
        #
        self._weight_index_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.int32)
        self._weight_lock = np.zeros( (self._num_node, self._num_input), dtype=np.int32)
        self._weight_mbid = np.zeros( (self._num_node, self._num_input), dtype=np.int32)
        self._weight_property = np.zeros( (self._num_node, self._num_input), dtype=np.int32)
        self._weight_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.float32)
        #
        if gpu:
            self._gpu_weight = self._gpu.dev_malloc(self._weight_matrix)

    def prepare(self, batch_size):
        self._batch_size = batch_size
        #
        self._product_matrix = np.zeros( (self._batch_size, self._num_node, self._num_input), dtype=np.float32)
        if self._gpu:
            self._gpu_product = self._gpu.dev_malloc(self._product_matrix)
        #
        self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.float32)
        if self._gpu:
            self._gpu_output = self._gpu.dev_malloc(self._output_array)
        
    def update_weight(self):
        self._gpu.copy(self._gpu_weight, self._weight_matrix)
        #
    def propagate(self, array_in, ni=-1, ii=-1, wi=-1, debug=0):
        stride_1 = self._num_node * self._num_input
        stride_2 = self._num_input
        
        if ni>=0: # alt
            self._gpu.multiple_x_by_w_batch_alt(array_in, self._gpu_weight, self._gpu_product,
                                                self._batch_size, stride_1, stride_2,
                                                self._num_input, self._num_node, ni, ii, WEIGHT_SET[wi])
        else: # propagation
            self._gpu.multiple_x_by_w_batch(array_in, self._gpu_weight, self._gpu_product,
                                            self._batch_size, stride_1, stride_2,
                                            self._num_input, self._num_node)
        #
        activation = 1
        self._gpu.sum(self._gpu_product, self._gpu_output,
                        self._num_input, self._num_node, activation, self._batch_size)
        #
        self._gpu.softmax(self._gpu_output, self._num_node, self._batch_size)
        #
        # debug
#        self._gpu.copy(self._output_array, self._gpu_output)
#        print self._output_array

#
# 2 x 2 simple max filter for 2D image data
# w : image width, i : index, h : image height
class MaxLayer(Layer):
    def __init__(self, i, ch, w, h, pre, gpu=None):
        print("MaxLayer::__init__()")
        self._ch = ch
        self._batch_stride = w * h * ch
        num_input = w*h
        self._x = int(w/2)
        self._y = int(h/2)
        num_node = self._x*self._y
        super(MaxLayer, self).__init__(i, LAYER_TYPE_MAX, num_input, num_node, pre, gpu)
        #

    def get_num_update(self):
        return 0

    def get_weight_property(self, ni, ii):
        return 0
        
    def set_weight_property(self, ni, ii, wi):
        pass
    
    def reset_weight_property_all(self):
        pass
    
    def get_weight_lock(self, ni, ii):
        return 0
        
    def set_weight_lock(self, ni, ii, lock):
        pass
    
    def unlock_weight_all(self):
        pass
    
    def set_weight_index(self, ni, ii, wi):
        pass
        
    def get_weight_index(self, ni, ii):
        return 0
    
    def export_weight_index(self):
        return None

    def import_weight_index(self, wi_list):
        pass
    
    def count_weight(self):
        return 0
    
    def count_locked_weight(self):
        return 0
        
    def prepare(self, batch_size):
        print("MaxLayer::prepare()")
        self._batch_size = batch_size
        self._output_array = np.zeros((self._batch_size, self._ch, self._num_node), dtype=np.float32)
        #
        if self._gpu:
            self._gpu_output = self._gpu.dev_malloc(self._output_array)
        #
        
    def propagate(self, array_in, ni=-1, ii=-1, wi=-1, debug=0):
#        print("MaxLayer::propagate()")
        self._gpu.max_batch(array_in, self._gpu_output,
                            self._ch, self._x, self._y,
                            self._batch_size, self._num_input)
        #
#        self._gpu.copy(self._output_array, self._gpu_output)
#        print self._output_array
    def set_weight_mbid(self, ni, ii, mbid):
        pass
        
    def get_weight_mbid(self, ni, ii):
        pass

class Conv_4_Layer(Layer):
    def __init__(self, i, w, h, ch, filter, pre, gpu=None):
        print("Convolution Layer ver.4 ::__init__()")
        self._cache = 0 # on
        self._ch = ch
        self._w = w
        self._h = h
        self._filter = filter
        self._filter_size = 3 * 3 * ch # width and height of filter are fixed to 3
        num_input = self._filter_size
        num_node = self._filter
        super(Conv_4_Layer, self).__init__(i, LAYER_TYPE_CONV_4, num_input, num_node, pre, gpu)

        # mems for weights
        self._weight_index_matrix = np.zeros( (self._filter, self._filter_size), dtype=np.int32)
        self._weight_lock = np.zeros( (self._filter, self._filter_size), dtype=np.int32)
        self._weight_mbid = np.zeros( (self._filter, self._filter_size), dtype=np.int32)
        self._weight_property = np.zeros( (self._filter, self._filter_size), dtype=np.int32)
        self._weight_matrix = np.zeros( (self._filter, self._filter_size), dtype=np.float32)
        #
        if self._gpu:
            self._gpu_weight = self._gpu.dev_malloc(self._weight_matrix)
        #
        
    def prepare(self, batch_size):
        print("Conv_4_Layer::prepare(%d)" %(batch_size))
        self._batch_size = batch_size
        # intermidiate
        self._padded_array = np.zeros((self._batch_size, (self._w+2)*(self._h+2)*self._ch), dtype=np.float32)

        # output
        self._output_array = np.zeros((self._batch_size, self._filter, self._w*self._h), dtype=np.float32)
        #
        if self._gpu:
            self._gpu_padded = self._gpu.dev_malloc(self._padded_array)
            self._gpu_output = self._gpu.dev_malloc(self._output_array)
        #
        
    def set_weight_index(self, ni, ii, wi):
        self._weight_index_matrix[ni][ii] = wi
        self._weight_matrix[ni][ii] = WEIGHT_SET[wi]

    def init_weight_with_random_index(self):
        for ni in range(self._num_node):
            for ii in range(self._num_input):
                wi = random.randrange(len(WEIGHT_SET))
                #wi = 7 # 5:0.0, 7:0.5
                self.set_weight_index(ni, ii, wi)
            #
        #
        
    def update_weight(self):
        self._gpu.copy(self._gpu_weight, self._weight_matrix)
        
    def propagate(self, array_in, ni=-1, ii=-1, wi=-1, debug=0):
#        print("Conv_4_Layer::propagate()")
#        if self._cache:
#            pass
#        else:
#            self._gpu.conv_4_pad_batch(array_in, self._gpu_padded, self._w, self._h, self._ch, self._batch_size)
#            self._cache = 1
        #
        self._gpu.conv_4_pad_batch(array_in, self._gpu_padded, self._w, self._h, self._ch, self._batch_size)
        
        # ni : filetr index, 0 to num of filter -1
        # ii : index of matrix, 0 to 3*3*ch-1
        if ni>=0:
            wi_undo = self._weight_index_matrix[ni][ii]
            self.set_weight_index(ni, ii, wi)
            self.update_weight()
            self._gpu.conv_4_roll_batch(self._gpu_padded, self._gpu_weight, self._gpu_output, self._w, self._h, self._ch, self._filter, self._batch_size)
            self.set_weight_index(ni, ii, wi_undo)
            self.update_weight()
        else:
            self._gpu.conv_4_roll_batch(self._gpu_padded, self._gpu_weight, self._gpu_output, self._w, self._h, self._ch, self._filter, self._batch_size)
        #
        # scale
        #
        size = self._filter * self._w * self._h
        self._gpu.scale_layer(self._gpu_output, size, self._batch_size)
        #
        # debug
#        self._gpu.copy(self._output_array, self._gpu_output)
#        print(self._output_array)
    #
    def save_output(self):
        #pass
        self._gpu.copy(self._output_array, self._gpu_output)
        #
        for bi in range(self._batch_size):
            for fi in range(self._filter):
                data_array = self._output_array[bi][fi]
                size = self._w * self._h
                max = np.max(data_array)
                min = np.min(data_array)
                print("max=%f, min=%f" % (max, min))
                #img_array = np.zeros( (self._h, self._w), dtype=np.int32)
                img = Image.new("L", (self._w, self._h), 0)
                pix = img.load()
                for y in range(self._h):
                    for x in range(self._w):
                        v = data_array[self._w*y + x]
                        #print v
                        v1 = int(v*255/max)
                        pix[x,y] = v1
                        
                        #img_array[y, x] = int(v*255/max)
                        #print img_array[y, x]
#                        v1 / 255 = v / max
                    #
                #
                #print img_array
                #img_gray = Image.fromarray(img_array)
                #print(img_gray.mode)
                img.save("./debug/cnn/%d_%d.png" %(bi, fi))
            #
        #
#
#
#
class Roster:
    def __init__(self):
        self._weight_list = []
        self._gpu = None
        self._remote = None
        self.layers = []
        self._batch_size = 1
        
    def set_gpu(self, gpu):
        self._gpu = gpu
        self._remote = None
        
    def set_remote(self, remote):
        self._gpu = None
        self._remote = remote

    def prepare(self, batch_size, data_size, num_class):
        self.num_class = num_class
        self._batch_size = batch_size
        #
        if self._gpu:
            self._batch_data = np.zeros((batch_size, data_size), dtype=np.float32)
            self._gpu_input = self._gpu.dev_malloc(self._batch_data)
            self._batch_class = np.zeros(batch_size, dtype=np.int32)
            self._gpu_labels = self._gpu.dev_malloc(self._batch_class)
            self._batch_cross_entropy = np.zeros(batch_size, dtype=np.float32)
            self._gpu_entropy = self._gpu.dev_malloc(self._batch_cross_entropy)
        #
        for layer in self.layers:
            layer.prepare(batch_size)
        #
        
    def set_data(self, data, data_size, label, batch_size):
        self._gpu.copy(self._gpu_input, data)
        self._gpu.copy(self._gpu_labels, label)
        #
        layer = self.getLayerAt(0) # input layer
        layer._gpu.scale(self._gpu_input, layer._gpu_output, data_size, float(255.0), layer._num_node, batch_size, 0)
        #
        #layer = self.getLayerAt(1) # CNN
        #layer._cache = 0
#        self._gpu.copy(layer._output_array, layer._gpu_output)
#        print layer._output_array[0]
            
    def init_weight(self):
        c = self.countLayers()
        for i in range(1, c):
            layer = self.getLayerAt(i)
            if layer.get_type()==LAYER_TYPE_MAX:
                pass
            else:
                layer.init_weight_with_random_index()
            #
        #

    def reset_weight_property(self, p=0):
        c = self.countLayers()
        for i in range(1, c):
            layer = self.getLayerAt(i)
            layer.reset_weight_property_all()
#            nc = layer._num_node
#            ic = layer._num_input
#            for ni in range(nc):
#                for ii in range(ic):
#                    if layer.get_type()==LAYER_TYPE_MAX:
#                        pass
#                    else:
#                        layer.set_weight_property(ni, ii, p)
#                    #
#                #
#            #
        #

    def count_locked_weight(self):
        cnt = 0
        c = self.countLayers()
        for i in range(1, c):
            layer = self.getLayerAt(i)
            cnt = cnt + layer.count_locked_weight()
#            nc = layer._num_node
#            ic = layer._num_input
#            for ni in range(nc):
#                for ii in range(ic):
#                    lock = layer.get_weight_lock(ni, ii)
#                    if lock>0:
#                        cnt = cnt + 1
#                    #
#                #
#            #
        #
        return cnt
    
    def count_weight(self):
        cnt = 0
        c = self.countLayers()
        for i in range(1, c):
            layer = self.getLayerAt(i)
            cnt = cnt + layer.count_weight()
        #
        return cnt


    def reset_weight_mbid(self):
        c = self.countLayers()
        for i in range(1, c):
            layer = self.getLayerAt(i)
            layer.reset_weight_mbid()
        #
    #
    def assign_weight_mbid(self, mbsize):
        c = self.countLayers()
        for i in range(1, c):
            layer = self.getLayerAt(i)
            layer.assign_weight_mbid_random(mbsize)
        #
    #

    def unlock_weight_all(self):
        c = self.countLayers()
        for i in range(1, c):
            layer = self.getLayerAt(i)
            layer.unlock_weight_all()
#            nc = layer._num_node
#            ic = layer._num_input
#            for ni in range(nc):
#                for ii in range(ic):
#                    layer.set_weight_lock(ni, ii, 0)
#                #
#            #
        #

    def update_weight(self):
        for layer in self.layers:
            layer.update_weight()
        #

    def count_layers(self):
        return len(self.layers)

    # this should be obsolute
    def countLayers(self):
        return len(self.layers)

    def get_layers(self):
        if self.countLayers() == 0:
            return 0
        #
        return self.layers

    def getLayerAt(self, i):
        return self.get_layer_at(i)
    
    def get_layer_at(self, i):
        c = self.countLayers()
        if i>=c:
            print("error : Roster : getLayerAt")
            return None
        #
        return self.layers[i]

    def add_layer(self, type, num_input, num_node):
        c = self.countLayers()
        if type==LAYER_TYPE_INPUT:
            layer = InputLayer(c, num_input, num_node, self._gpu)
            self.layers.append(layer)
            return layer
        elif type==LAYER_TYPE_HIDDEN:
            layer = HiddenLayer(c, num_input, num_node, self._gpu)
            self.layers.append(layer)
            return layer
        elif type==LAYER_TYPE_OUTPUT:
            layer = OutputLayer(c, num_input, num_node, self._gpu)
            self.layers.append(layer)
            return layer
        elif type==LAYER_TYPE_CONV_4:
            print("not yet")
            return
        elif type==LAYER_TYPE_MAX:
            print("not yet")
            return
 
    def get_inference(self):
        #ret = []
        c = self.countLayers()
        output = self.getLayerAt(c-1)
        output._gpu.copy(output._output_array, output._gpu_output)
        #print output._output_array[0]
        return output._output_array

    def get_answer(self):
        ret = []
        c = self.countLayers()
        output = self.getLayerAt(c-1)
        output._gpu.copy(output._output_array, output._gpu_output)
        #
        for i in range(self._batch_size):
            inf = output._output_array[i]
            max_index = -1
            max = -1.0
            for j in range(self.num_class):
                if inf[j]>max:
                    max = inf[j]
                    max_index = j
                #
            #
            ret.append(max_index)
        #
        return ret
    
    def get_cross_entropy(self):
        c = self.countLayers()
        output = self.getLayerAt(c-1)
        self._gpu.k_cross_entropy(output._gpu_output, self._gpu_entropy,
                                  self._gpu_labels, self.num_class, self._batch_size)
        self._gpu.copy(self._batch_cross_entropy, self._gpu_entropy)
        # debug
        #print self._batch_cross_entropy
        #
        #ret = np.sum(self._batch_cross_entropy)/float(self._batch_size)
        #print "    CE=%f" % (ret)
        #return ret
        s = np.sum(self._batch_cross_entropy)
        s = s/float(self._batch_size)
        
        if np.isnan(s):
            for i in range(self._batch_size):
                li = c-1
                if np.isnan(self._batch_cross_entropy[i]):
                    print("NaN : %d" % (i))
                    for li in range(c):
                        output = self.getLayerAt(li)
                        self._gpu.copy(output._output_array, output._gpu_output)
                        print("layer : %d" % (li))
                        print(output._output_array[i].shape)
                        print(output._output_array[i])
                    #
                #
            #
        #
        return s
        #return np.sum(self._batch_cross_entropy)/float(self._batch_size)
    
    def export_weight_index(self, path):
        print("Roster : export_weight_index(%s)" % path)
        with open(path, "w") as f:
            writer = csv.writer(f, lineterminator='\n')
            c = self.countLayers()
            for i in range(1, c):
                layer = self.getLayerAt(i)
                #print "%d : %d" % (i, layer.get_type())
                data = layer.export_weight_index()
                if data:
                    writer.writerows(data)
                #
            #
        #

    def import_weight_index(self, path):
        print("Roster : import_weight_index(%s)" % path)
        with open(path, "r") as f:
            reader = csv.reader(f)
            lc = self.countLayers()
            for i in range(1, lc):
                layer = self.getLayerAt(i)
                #print "%d : %d" % (i, layer.get_type())
                
                #count_weight
                
                if layer.get_type()==LAYER_TYPE_MAX:
                    #print "fuck"
                    continue
                #
                nc  = layer._num_node
                block = []
                for row in reader:
                    line = []
                    for cell in row:
                        line.append(cell)
                    #
                    block.append(line)
                    if len(block)==nc:
                        break
                    #
                #
                layer.import_weight_index(block)
            # for
        # with

    def propagate(self, li=-1, ni=-1, ii=-1, wi=-1, debug=0):
        c = self.countLayers()
        pre = self.getLayerAt(0)
        for i in range(1, c):
            layer = self.getLayerAt(i)
            if i==li: # alt
                layer.propagate(pre._gpu_output, ni, ii, wi, debug)
            else: # propagation
                layer.propagate(pre._gpu_output, -1, -1, -1, debug)
            #
            pre = layer
        #
#
#
def main():
    r = Roster()
    #
    input_layer = r.add_layer(0, 196, 196)
    hidden_layer_1 = r.add_layer(1, 196, 32)
    hidden_layer_2 = r.add_layer(1, 32, 32)
    output_layer = r.add_layer(2, 32, 10)
    r.init_weight()
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
# EOF
