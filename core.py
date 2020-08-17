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
LAYER_TYPE_INPUT  = 0
LAYER_TYPE_HIDDEN = 1
LAYER_TYPE_OUTPUT = 2
LAYER_TYPE_CONV   = 3
LAYER_TYPE_POOL   = 4
LAYER_TYPE_CONV_2D = 5

class Layer(object):
    # i         : index of layers
    # type      : 0 = input, 1 = hidden, 2 = output
    # num_input : number of inputs / outputs from a previous layer
    # num_node  : numbers of neurons
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
        # mems for weights
        self._weight_index_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.int32)
        self._weight_lock = np.zeros( (self._num_node, self._num_input), dtype=np.int32)
        self._weight_property = np.zeros( (self._num_node, self._num_input), dtype=np.int32)
        self._weight_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.float32)
        #
        self._node_marker = np.zeros( (self._num_node), dtype=np.int32)
        #
        self._num_update = num_input/10
        if self._num_update<1:
            self._num_update = 1
        #
        self._learning = 1 # 0 : off, 1 : on
        #
        # allocate mems for output, also intermediate working area when it is needed
        #
        
    def get_pre_layer(self):
        return self._pre
        
    def set_marker_pre(self, ni, v):
        if self._pre:
            self._pre.set_marker(ni, v)
            
    def set_marker(self, ni, v):
        self._node_marker[ni] = v

    def get_marker(self, ni):
        return self._node_marker[ni]

    def set_learning(self, v):
        self._learning = v

    def get_learning(self):
        return self._learning

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

    def get_weight_lock(self, ni, ii):
        return self._weight_lock[ni][ii]
    
    def set_weight_lock(self, ni, ii, l):
        self._weight_lock[ni][ii] = l
    
    def set_weight_index(self, ni, ii, wi): # weight index
        #print "set_weight_index(%d, %d)" % (ni, ii)
        self._weight_index_matrix[ni][ii] = wi
        self._weight_matrix[ni][ii] = WEIGHT_SET[wi]
    
    def init_weight_with_random_index(self):
        #print "init_weight_with_random_index"
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
                #print "(%d, %d) = %d" % (ni, ii, self._weight_index_matrix[ni][ii])
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
    def __init__(self, i, num_input, num_node, gpu=None):
        print "InputLayer::__init__()"
        super(InputLayer, self).__init__(i, LAYER_TYPE_INPUT, num_input, num_node, gpu)
        self._learning = 0 # off
    
    def prepare(self, batch_size):
        self._batch_size = batch_size
        #
        self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.float32)
        if self._gpu:
            self._gpu_output = self._gpu.dev_malloc(self._output_array)
        #
        
    def propagate(self, array_in, ni=-1, ii=-1, wi=-1, debug=0):
        pass

#
#
#
class HiddenLayer(Layer):
    def __init__(self, i, num_input, num_node, gpu=None):
        print "HiddenLayer::__init__()"
        super(HiddenLayer, self).__init__(i, LAYER_TYPE_HIDDEN, num_input, num_node, gpu)
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
        # sum
        #
        activation = 0 # 0 : relu, 1 : skip
        self._gpu.sum(self._gpu_product, self._gpu_output,
                        self._num_input, self._num_node, activation, self._batch_size)
        self._gpu.scale_layer(self._gpu_output, self._num_node, self._batch_size)
        #
        # normalize
        #
        #self._gpu.normalize_layer(self._gpu_output, self._num_node, self._batch_size)
        #
        # relu
        #
        #self._gpu.relu(self._gpu_output, self._batch_size, self._num_node, 1) #  self._num_node : stride
        
#
#
#
class OutputLayer(Layer):
    def __init__(self, i, num_input, num_node, gpu=None):
        print "OutputLayer::__init__()"
        super(OutputLayer, self).__init__(i, LAYER_TYPE_OUTPUT, num_input, num_node, gpu)
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
        #self._gpu.copy(self._output_array, self._gpu_output)
        #print self._output_array
#
#
# w : image width, i : index, h : image height, stride : size of convolusion matrix
class ConvLayer(Layer):
    def __init__(self, i,  w, h, stride, gpu=None):
        print "ConvLayer::__init__()"
        #
        self._index = i
        self._type = LAYER_TYPE_CONV
        self._gpu = gpu
        self._id = -1
        #
        self._batch_stride = w * h
        self._kernel_x = w - (stride-1) # 26
        self._kernel_y = h - (stride-1) # 26
        self._kernel_num = self._kernel_x * self._kernel_y # 676
        self._kernel_size = stride * stride # 9
        #
        super(ConvLayer, self).__init__(i, LAYER_TYPE_CONV, self._kernel_size, self._kernel_num, gpu)
        #
        self._cnv_map = np.zeros( (self._kernel_num, self._kernel_size), dtype=np.int32)
        #
        cmap_index = 0
        for y in range(self._kernel_y):
            for x in range(self._kernel_x):
                cmap = self._cnv_map[cmap_index]
                cmap[0] = x + 0 + y*w
                cmap[1] = x + 1 + y*w
                cmap[2] = x + 2 + y*w
                cmap[3] = x + 0 + (y+1)*w
                cmap[4] = x + 1 + (y+1)*w
                cmap[5] = x + 2 + (y+1)*w
                cmap[6] = x + 0 + (y+2)*w
                cmap[7] = x + 1 + (y+2)*w
                cmap[8] = x + 2 + (y+2)*w
                cmap_index = cmap_index + 1
                #print cmap
            #
        #

    def prepare(self, batch_size):
        self._batch_size = batch_size
        self._output_array = np.zeros((self._batch_size, self._kernel_num), dtype=np.float32)
        #
        if self._gpu:
            self._gpu_weight = self._gpu.dev_malloc(self._weight_matrix)
            self._gpu_cnv_map = self._gpu.dev_malloc(self._cnv_map)
            self._gpu_output = self._gpu.dev_malloc(self._output_array)
            #
            #print self._cnv_map[0]
            self._gpu.copy(self._gpu_cnv_map, self._cnv_map)
        #
        
    def propagate(self, array_in, ni=-1, ii=-1, wi=-1, debug=0):
        if ni>=0: # alt
            alt_y, alt_x = divmod(ni, self._kernel_x)
            self._gpu.conv_batch_alt(array_in, self._gpu_cnv_map, self._gpu_weight, self._gpu_output,
                                     self._kernel_size, self._kernel_x, self._kernel_y,
                                     self._batch_size, self._batch_stride,
                                     alt_x, alt_y, ii, WEIGHT_SET[wi])
                                     
        else: # propagation                                                
            self._gpu.conv_batch(array_in, self._gpu_cnv_map, self._gpu_weight, self._gpu_output,
                                 self._kernel_size, self._kernel_x, self._kernel_y,
                                 self._batch_size, self._batch_stride)
                                 
            #self._gpu.copy(self._output_array, self._gpu_output)
            #print self._output_array[0]#.sum()
            #ddd = self._output_array[0]
            #for i in range(self._batch_size):
            #    print "%d : %f" % (i, self._output_array[i].sum())
            #
        #
        
#
# 2 x 2 simple max filter for 2D image data
# w : image width, i : index, h : image height
class MaxLayer(Layer):
    def __init__(self, i, ch, w, h, gpu=None):
        print "MaxLayer::__init__()"
        #
        self._index = i
        self._type = LAYER_TYPE_POOL
        self._gpu = gpu
        self._id = -1 # reserved
        #
        self._ch = ch
        self._num_input = w * h
        self._x = w/2
        self._y = h/2
        self._num_node = self._x * self._y
        self._batch_stride = w * h * ch
        #
        #super(MaxLayer, self).__init__(i, LAYER_TYPE_POOL, num_input, num_node, gpu)
        #
        self._learning = 0
    
    def set_weight_index(self, ni, ii, wi):
        pass
    
    def export_weight_index(self):
        return None
    
    def import_weight_index(self, wi_list):
        pass
    
    def prepare(self, batch_size):
        self._batch_size = batch_size
        self._output_array = np.zeros((self._batch_size, self._ch, self._num_node), dtype=np.float32)
        #
        if self._gpu:
            self._gpu_output = self._gpu.dev_malloc(self._output_array)
        #
        
    def propagate(self, array_in, ni=-1, ii=-1, wi=-1, debug=0):
        self._gpu.max_batch(array_in, self._gpu_output,
                            self._ch, self._x, self._y,
                            self._batch_size, self._num_input)
        #
        #self._gpu.copy(self._output_array, self._gpu_output)
        #print self._output_array[0]

# fixed : kernel size = 3 x 3, kernel stride = 1
# ch : number of channels, filter : number of filters
# number of output channels is the same as number of filters
class Conv2dLayer(Layer):
    def __init__(self, i, w, h, ch, filter, gpu=None):
        print "Conv2dLayer::__init__()"
        #
        self._index = i
        self._type = LAYER_TYPE_CONV_2D
        self._gpu = gpu
        self._id = -1 # reserved
        #
        self._ch = ch
        self._w = w
        self._h = h
        self._image_size = self._w * self._h
        self._filter = filter
        self._filter_size = 3*3
        self._ch_stride = self._w * self._h
        self._batch_stride = self._ch_stride * self._filter
        # mems for weights
        self._weight_index_matrix = np.zeros( (self._filter, self._filter_size), dtype=np.int32)
        self._weight_lock = np.zeros( (self._filter, self._filter_size), dtype=np.int32)
        self._weight_property = np.zeros( (self._filter, self._filter_size), dtype=np.int32)
        self._weight_matrix = np.zeros( (self._filter, self._filter_size), dtype=np.float32)
        #
        self._num_input = self._filter_size
        self._num_node = self._filter
        #
        if self._gpu:
            self._gpu_weight = self._gpu.dev_malloc(self._weight_matrix)

    # allocate mems for output, also intermediate working area when it is needed
    def prepare(self, batch_size):
        self._batch_size = batch_size
        self._output_array = np.zeros((self._batch_size, self._filter, self._image_size), dtype=np.float32)
        #
        if self._gpu:
            #self._gpu_weight = self._gpu.dev_malloc(self._weight_matrix)
            self._gpu_output = self._gpu.dev_malloc(self._output_array)
        #
    
    def set_weight_index(self, ni, ii, wi):
        self._weight_index_matrix[ni][ii] = wi
        self._weight_matrix[ni][ii] = WEIGHT_SET_CNN[wi]
    
    def init_weight_with_random_index(self):
        for ni in range(self._num_node):
            for ii in range(self._num_input):
                wi = random.randrange(len(WEIGHT_SET_CNN))
                self.set_weight_index(ni, ii, wi)
            #
        #
    
    def update_weight(self):
        self._gpu.copy(self._gpu_weight, self._weight_matrix)
        
    def propagate(self, array_in, ni=-1, ii=-1, wi=-1, debug=0):
        if ni>=0: # alt
            pass
#            self._gpu.conv2d_batch_alt(array_in, self._gpu_weight, self._gpu_output,
#                                       self._w, self._h, self._ch, self._filter, self._batch_size,
#                                       ni, ii, WEIGHT_SET_CNN[wi])#wi)
        else:
            self._gpu.conv2d_batch(array_in, self._gpu_weight, self._gpu_output,
                                   self._w, self._h, self._ch, self._filter, self._batch_size)
            self._gpu.scale_cnn(self._gpu_output, self._batch_size, self._filter,
                                self._image_size*self._filter, self._image_size)

        #
#        self._gpu.scale_cnn(self._gpu_output, self._batch_size, self._filter,
#                            self._image_size*self._filter, self._image_size)
        # normalize
        #
        #self._gpu.normalize_batch_cnn(self._gpu_output, self._batch_size, self._image_size*self._filter, self._filter, self._image_size)
        #
        # relu
        #
        #self._gpu.relu(self._gpu_output, self._batch_size, self._filter, self._image_size) #  self._num_node : stride
        
        
            #self._gpu.copy(self._output_array, self._gpu_output)
            #print self._output_array[0][0]
            #print self._weight_matrix[0]
            #print self._output_array[1]
            #print self._weight_matrix[1]
    
#    def init_weight_with_random_index(self):
#        for ni in range(self._num_node):
#            for ii in range(self._num_input):
#                wi = random.randrange(WEIGHT_INDEX_ZERO, WEIGHT_INDEX_SIZE)
#                self.set_weight_index(ni, ii, wi)
#            #
#        #
        
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
        layer = self.getLayerAt(0) # input layer
        layer._gpu.scale(self._gpu_input, layer._gpu_output, data_size, float(255.0),
                         layer._num_node, batch_size, 0)
        #
        # preprocess cnn and max
        #
        debug = 0
        pre = layer
        li = 1
        layer = self.getLayerAt(li)
        while layer.get_learning()==0:
            layer.propagate(pre._gpu_output, -1, -1, -1, debug)
            pre = layer
            li = li +1
            layer = self.getLayerAt(li)
        #
#        cnn_layer = self.getLayerAt(1)
#        cnn_layer.propagate(layer._gpu_output, -1, -1, -1, debug)
        #
#        max_layer = self.getLayerAt(2)
#        max_layer.propagate(cnn_layer._gpu_output, -1, -1, -1, debug)
        
            
    def init_weight(self):
        c = self.countLayers()
        for i in range(1, c):
            layer = self.getLayerAt(i)
            if layer.get_type()==LAYER_TYPE_POOL:
                pass
            else:
                layer.init_weight_with_random_index()
            #
        #

    def reset_weight_property(self, p=0):
        c = self.countLayers()
        for i in range(1, c):
            layer = self.getLayerAt(i)
            nc = layer._num_node
            ic = layer._num_input
            for ni in range(nc):
                for ii in range(ic):
                    if layer.get_type()==LAYER_TYPE_POOL:
                        pass
                    else:
                        layer.set_weight_property(ni, ii, p)
                    #
                #
            #
        #

    def unlock_weight_all(self):
        c = self.countLayers()
        for i in range(1, c):
            layer = self.getLayerAt(i)
            nc = layer._num_node
            ic = layer._num_input
            for ni in range(nc):
                for ii in range(ic):
                    layer.set_weight_lock(ni, ii, 0)
        #

    def update_weight(self):
        for layer in self.layers:
            layer.update_weight()

    def countLayers(self):
        return len(self.layers)

    def get_layers(self):
        if self.countLayers() == 0:
            return 0
        return self.layers

    def getLayerAt(self, i):
        c = self.countLayers()
        if i>=c:
            print "error : Roster : getLayerAt"
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
        elif type==LAYER_TYPE_CONV:
            print "not yet"
            return
        elif type==LAYER_TYPE_POOL:
            print "not yet"
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
                    print "NaN : %d" % (i)
                    
                    output = self.getLayerAt(li)
                    self._gpu.copy(output._output_array, output._gpu_output)
                    print output._output_array[i]
                    
                    output = self.getLayerAt(li-1)
                    self._gpu.copy(output._output_array, output._gpu_output)
                    print output._output_array[i]
                #
            #
        #
        return s
        #return np.sum(self._batch_cross_entropy)/float(self._batch_size)
    
    def export_weight_index(self, path):
        print "Roster : export_weight_index(%s)" % path
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
        print "Roster : import_weight_index(%s)" % path
        with open(path, "r") as f:
            reader = csv.reader(f)
            lc = self.countLayers()
            for i in range(1, lc):
                layer = self.getLayerAt(i)
                #print "%d : %d" % (i, layer.get_type())
                if layer.get_type()==LAYER_TYPE_POOL:
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
            if layer.get_learning()>0:
                pass
            else:
                pre = layer
                continue
            #
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
    print ">> start"
    sts = main()
    print ">> end"
    print("\007")
    sys.exit(sts)
#
#
# EOF
