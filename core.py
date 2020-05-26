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
WEIGHT_SET = WEIGHT_SET_2
WEIGHT_INDEX_SIZE = len(WEIGHT_SET)
WEIGHT_INDEX_ZERO = WEIGHT_INDEX_SIZE/2
WEIGHT_INDEX_MAX = WEIGHT_INDEX_SIZE-1
WEIGHT_INDEX_MIN = 0
#
#
#
def sigmoid(x):
    a = 0.0
    try:
        a = np.exp(-x)
    except OverflowError:
        a = float('inf')
        print "sigmoid(fuck)"
    #
    ret = 1.0 / (1.0 + a)
    return ret
#
#
#
def relu(x):
    if x<=0.0:
        return 0.0
    #
    return x
#
#
#
def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x-c)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y

def softmax_no_exp(x):
    sum_x = np.sum(x)
    if sum_x==0.0:
        return np.zeros_like((x))
    #
    y = x / sum_x
    return y
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

class Layer(object):
    # i         : index of layers
    # type      : 0 = input, 1 = hidden, 2 = output
    # num_input : number of inputs / outputs from a previous layer
    # num_node  : numbers of neurons
    def __init__(self, i, type, num_input, num_node, gpu=None):
        self._index = i
        self._type = type
        self._gpu = gpu
        self._id = -1
        self._num_input = num_input
        self._num_node = num_node
        #
        self._weight_index_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.int32)
        self._weight_lock = np.zeros( (self._num_node, self._num_input), dtype=np.int32)
        self._weight_property = np.zeros( (self._num_node, self._num_input), dtype=np.int32)
        self._weight_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.float32)
    
    def set_batch(self, batch_size):
        self._batch_size = batch_size
        if self._type>0:
            self._product_matrix = np.zeros( (self._batch_size, self._num_node, self._num_input), dtype=np.float32)
            self._gpu_product = self._gpu.dev_malloc(self._product_matrix)

        self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.float32)
        self._gpu_output = self._gpu.dev_malloc(self._output_array)
            
    def update_weight(self):
        # gpu must be checked before this method is called
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
    
    def getType(self):
        return self._type
#
#
#
class InputLayer(Layer):
    def __init__(self, i, num_input, num_node, gpu=None):
        print "InputLayer::__init__()"
        super(InputLayer, self).__init__(i, LAYER_TYPE_INPUT, num_input, num_node, gpu)
    
    def prepare(self, batch_size):
        self._batch_size = batch_size
        #
        self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.float32)
        self._gpu_output = self._gpu.dev_malloc(self._output_array)
    
    def propagate(self, array_in, ni=-1, ii=-1, wi=-1, debug=0):
        pass

#
#
#
class HiddenLayer(Layer):
    def __init__(self, i, num_input, num_node, gpu=None):
        print "HiddenLayer::__init__()"
        super(HiddenLayer, self).__init__(i, LAYER_TYPE_HIDDEN, num_input, num_node, gpu)
        self._gpu_weight = self._gpu.dev_malloc(self._weight_matrix)
    
    def prepare(self, batch_size):
        self._batch_size = batch_size
        #
        self._product_matrix = np.zeros( (self._batch_size, self._num_node, self._num_input), dtype=np.float32)
        self._gpu_product = self._gpu.dev_malloc(self._product_matrix)
        #
        self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.float32)
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
        activation = 0
        self._gpu.k_sum(self._gpu_product, self._gpu_output,
                        self._num_input, self._num_node, activation, self._batch_size)
        #
        # normalize
        self._gpu.normalize(self._gpu_output, self._num_node, self._batch_size)
#
#
#
class OutputLayer(Layer):
    def __init__(self, i, num_input, num_node, gpu=None):
        print "OutputLayer::__init__()"
        super(OutputLayer, self).__init__(i, LAYER_TYPE_OUTPUT, num_input, num_node, gpu)
        self._gpu_weight = self._gpu.dev_malloc(self._weight_matrix)

    def prepare(self, batch_size):
        self._batch_size = batch_size
        #
        self._product_matrix = np.zeros( (self._batch_size, self._num_node, self._num_input), dtype=np.float32)
        self._gpu_product = self._gpu.dev_malloc(self._product_matrix)
        #
        self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.float32)
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
        self._gpu.k_sum(self._gpu_product, self._gpu_output,
                        self._num_input, self._num_node, activation, self._batch_size)
        #
        self._gpu.k_softmax(self._gpu_output, self._num_node, self._batch_size)
#
#
# w : image width, i : index, h : image height, stride : size of convolusion matrix
class ConvLayer(Layer):
    def __init__(self, i,  w, h, stride, num_input, gpu=None):
        print "ConvLayer::__init__()"
        #
        kx = w - (stride-1)
        ky = h - (stride-1)
        knum = kx * ky
        self._num_node = knum
        #
        super(HiddenLayer, self).__init__(i, LAYER_TYPE_CONV, num_input, num_node, gpu)
        #
        self._cnv_map = np.zeros( (self._num_node, stride), dtype=np.int32)
        self._weight_index = np.zeros( (self._num_node, stride), dtype=np.int32)
        self._gpu_weight = self._gpu.dev_malloc(self._weight_matrix)
        
        self._weight_index_matrix = np.zeros( (self._num_node, stride), dtype=np.int32)
        self._weight_lock = np.zeros( (self._num_node, self._num_input), dtype=np.int32)
        self._weight_property = np.zeros( (self._num_node, self._num_input), dtype=np.int32)
        
        #self._cnv_map = np.zeros( (self._num_node, stride), dtype=np.int32)
        #self._cnv_weight_index = np.zeros( (self._num_node, stride), dtype=np.int32)
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
            pass
        else:
            return
        #
        self._batch_data = np.zeros((batch_size, data_size), dtype=np.float32)
        self._gpu_input = self._gpu.dev_malloc(self._batch_data)
        self._batch_class = np.zeros(batch_size, dtype=np.int32)
        self._gpu_labels = self._gpu.dev_malloc(self._batch_class)
        self._batch_cross_entropy = np.zeros(batch_size, dtype=np.float32)
        self._gpu_entropy = self._gpu.dev_malloc(self._batch_cross_entropy)
        #
        for layer in self.layers:
            layer.set_batch(batch_size)
        #
        
    def set_data(self, data, data_size, label, batch_size):
        self._gpu.copy(self._gpu_input, data)
        self._gpu.copy(self._gpu_labels, label)
        layer = self.getLayerAt(0) # input layer
        layer._gpu.scale(self._gpu_input, layer._gpu_output, data_size, float(255.0),
                         layer._num_node, batch_size, 0)
    
    def init_weight(self):
        c = self.countLayers()
        for i in range(1, c):
            layer = self.getLayerAt(i)
            layer.init_weight_with_random_index()
        #

    def reset_weight_property(self, p=0):
        c = self.countLayers()
        for i in range(1, c):
            layer = self.getLayerAt(i)
            nc = layer._num_node
            ic = layer._num_input
            for ni in range(nc):
                for ii in range(ic):
                    layer.set_weight_property(ni, ii, p)
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
        elif type==LAYER_TYPE_HIDDEN:
            layer = HiddenLayer(c, num_input, num_node, self._gpu)
            self.layers.append(layer)
        elif type==LAYER_TYPE_OUTPUT:
            layer = OutputLayer(c, num_input, num_node, self._gpu)
            self.layers.append(layer)
        elif type==LAYER_TYPE_CONV:
            print "not yet"
            return
        elif type==LAYER_TYPE_POOL:
            print "not yet"
            return
 
    def get_inference(self):
        ret = []
        c = self.countLayers()
        output = self.getLayerAt(c-1)
        output._gpu.copy(output._output_array, output._gpu_output)
        return output._output_array
        
    def get_cross_entropy(self):
        c = self.countLayers()
        output = self.getLayerAt(c-1)
        self._gpu.k_cross_entropy(output._gpu_output, self._gpu_entropy,
                                  self._gpu_labels, self.num_class, self._batch_size)
        self._gpu.copy(self._batch_cross_entropy, self._gpu_entropy)
        #ret = np.sum(self._batch_cross_entropy)/float(self._batch_size)
        #print "    CE=%f" % (ret)
        #return ret
        return np.sum(self._batch_cross_entropy)
        #return np.sum(self._batch_cross_entropy)/float(self._batch_size)
    
    def export_weight_index(self, path):
        print "Roster : export_weight_index(%s)" % path
        with open(path, "w") as f:
            writer = csv.writer(f, lineterminator='\n')
            c = self.countLayers()
            for i in range(1, c):
                layer = self.getLayerAt(i)
                writer.writerows(layer.export_weight_index())

    def import_weight_index(self, path):
        #print "Roster : import_weight_index(%s)" % path
        with open(path, "r") as f:
            reader = csv.reader(f)
            lc = self.countLayers()
            for i in range(1, lc):
                layer = self.getLayerAt(i)
                nc  = layer._num_node
                block = []
                for row in reader:
                    line = []
                    for cell in row:
                        line.append(cell)
            
                    block.append(line)
                    if len(block)==nc:
                        break
                #
                layer.import_weight_index(block)
            # for
        # with

    def propagate(self, li=-1, ni=-1, ii=-1, wi=-1, debug=0):
        c = self.countLayers()
        pre = self.getLayerAt(0)
        # this line can be deleted later
        pre.propagate(pre._gpu_output, -1, -1, -1, debug)
        #
        # input layer is pre-prosessed
        #
        for i in range(1, c):
            layer = self.getLayerAt(i)
            #layer.propagate(pre._gpu_output, ni, ii, wi, debug)
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
