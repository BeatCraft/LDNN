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

# won't work at all
WEIGHT_SET_1 = [0.0, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]

WEIGHT_SET_2 = [-1, -0.72363462, -0.52364706, -0.37892914, -0.27420624, -0.19842513, -0.14358729, -0.10390474,
                -0.07518906, -0.05440941, -0.03937253, -0.02849133, -0.02061731, -0.0149194, -0.01079619, -0.0078125,
                0,
                0.0078125, 0.01079619, 0.0149194, 0.02061731, 0.02849133, 0.03937253, 0.05440941, 0.07518906,
                0.10390474, 0.14358729, 0.19842513, 0.27420624, 0.37892914, 0.52364706, 0.72363462, 1]

# upto 59 %
WEIGHT_SET_3 = [-1.0, -0.5, -0.25, -0.125, 0.125, 0.25, 0.5, 1.0]

# upto 50 %
WEIGHT_SET_4 = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

WEIGHT_SET = WEIGHT_SET_0
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
    
    ret = 1.0 / (1.0 + a)
    return ret

#def sigmoid(a):
#    s = 1 / (1 + e**-a)
#    return s
#
#def sigmoid(x):
#    return 1.0 / (1.0 + np.exp(-x))
#
#def sigmoid(x):
#    sigmoid_range = 34.538776394910684
#
#    if x <= -sigmoid_range:
#        return 1e-15
#    if x >= sigmoid_range:
#        return 1.0 - 1e-15
#
#    return 1.0 / (1.0 + np.exp(-x))

#
#
#
def relu(x):
    if x<=0.0:
        return 0.0
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
class Layer:
    # i         : index of layers
    # type      : 0 = input, 1 = hidden, 2 = output
    # num_input : number of inputs / outputs from a previous layer
    # num_node  : numbers of neurons
    def __init__(self, i, type, num_input, num_node, gpu=None):
        self._gpu = gpu
        self._id = -1
        self._index = i
        self._type = type
        self._num_input = num_input
        self._num_node = num_node
        self._weight_index_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.int32)
        self._weight_lock = np.zeros( (self._num_node, self._num_input), dtype=np.int32)
        self._weight_property = np.zeros( (self._num_node, self._num_input), dtype=np.int32)
        #self._weight_property = np.ones( (self._num_node, self._num_input), dtype=np.int32)
        self._weight_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.float32)
        self._sum = np.zeros(self._num_node, dtype=np.float32)
        #self._node_list = []
    
    def set_batch(self, batch_size):
        self._batch_size = batch_size
        if self._type>0:
            self._product_matrix = np.zeros( (self._batch_size, self._num_node, self._num_input), dtype=np.float32)
            self._gpu_product = self._gpu.dev_malloc(self._product_matrix)

        self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.float32)
        self._gpu_output = self._gpu.dev_malloc(self._output_array)

    def init_gpu(self):
        if self._gpu:
            pass
        else:
            print "no gpu"
            return
        
        if self._type>0:
            self._gpu_weight = self._gpu.dev_malloc(self._weight_matrix)

    def update_weight_gpu(self):
        if self._type>0:
            self._gpu.copy(self._gpu_weight, self._weight_matrix)

    def propagate(self, array_in, ni=-1, ii=-1, wi=-1, debug=0):
        if self._type==0: # input
            pass
        else:
            stride_1 = self._num_node * self._num_input
            stride_2 = self._num_input
            #print self._weight_matrix
            if ni>=0:
                for bi in range(self._batch_size):
                    self._gpu.multiple_x_by_w_alt(array_in, self._gpu_weight, self._gpu_product,
                                                  bi, stride_1, stride_2,
                                                  self._num_input, self._num_node,
                                                  ni, ii, WEIGHT_SET[wi])
            else:
                for bi in range(self._batch_size):
                    self._gpu.multiple_x_by_w(array_in, self._gpu_weight, self._gpu_product,
                                              bi, stride_1, stride_2,
                                              self._num_input, self._num_node)
            #
            self._gpu.copy(self._product_matrix, self._gpu_product)
            #
            for bi in range(self._batch_size):
                if self._type==1:
                    for i in range(len(self._product_matrix[bi])):
                        self._output_array[bi][i] = relu( np.sum(self._product_matrix[bi][i]) )
            
                    self._gpu.copy(self._gpu_output, self._output_array)
                    
                else:
                    sum = np.zeros(len(self._product_matrix[bi]), dtype=np.float32)
                    for i in range(len(self._product_matrix[bi])):
                        sum[i] = np.sum(self._product_matrix[bi][i])

                    self._output_array[bi] = softmax(sum)

                #print self._output_array[bi]
        #
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
        #print self._weight_index_matrix

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
class Roster:
    def __init__(self):
        self._weight_list = []
        self._gpu = None
        self.layers = []
        self._batch_size = 1
    
    def set_batch(self, batch, batch_size, data_size, debug=0):
        self._batch_data = np.zeros((batch_size, data_size), dtype=np.float32)
        self._gpu_input = self._gpu.dev_malloc(self._batch_data)
        self._batch_class = np.zeros(batch_size, dtype=int)
        self._batch_size = batch_size
        #
        for layer in self.layers:
            layer.set_batch(batch_size)
        
        layer = self.getLayerAt(0) # input layer
        for i in range(batch_size):
            entry = batch[i]
            self._batch_data[i] = entry[0].copy()
            self._batch_class[i] = entry[1]
        #
        self._gpu.copy(self._gpu_input, self._batch_data)
        layer._gpu.scale(self._gpu_input, layer._gpu_output, 28*28, float(255.0), layer._num_node, batch_size, 0)
        ### debug use ###
        if debug:
            self._gpu.copy(layer._output_array, layer._gpu_output)
            for i in range(batch_size):
                print layer._output_array[i]
        
    def set_gpu(self, my_gpu):
        self._gpu = my_gpu
    
    def init_weight(self):
        c = self.countLayers()
        for i in range(1, c):
            layer = self.getLayerAt(i)
            layer.init_weight_with_random_index()


    def reset_weight_property(self, p=0):
        c = self.countLayers()
        for i in range(1, c):
            layer = self.getLayerAt(i)
            nc = layer._num_node
            ic = layer._num_input
            for ni in range(nc):
                for ii in range(ic):
                    layer.set_weight_property(ni, ii, p)

    def unlock_weight_all(self):
        c = self.countLayers()
        for i in range(1, c):
            layer = self.getLayerAt(i)
            nc = layer._num_node
            ic = layer._num_input
            for ni in range(nc):
                for ii in range(ic):
                    layer.set_weight_lock(ni, ii, 0)
#        for w in self._weight_list:
#            w._lock = 0

#    def restore_weighgt(self, w_list):
#        c = 0
#        for w in self._weight_list:
#            wi = int(w_list[c])
#            w.set_index(wi)
#            w.set_id(c)
#            c += 1

    def update_weight(self):
        for layer in self.layers:
            layer.update_weight_gpu()

#    def get_weight_list(self):
#        return self._weight_list

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
        return self.layers[i]

    def add_layer(self, type, num_input, num_node):
        c = self.countLayers()
        layer = Layer(c, type, num_input, num_node, self._gpu)
        #if c>0: # skip input layer
        #    for i in range(num_node):
        #        node = Node()
        #        layer.add_node(node)
                #for j in range(num_input):
                #    w = Weight(layer, i, j)
                #    self._weight_list.append(w)
                #    k = 0
                #    k = len(self._weight_list) - 1
                #    node.add_weight(k)
        #
        self.layers.append(layer)
        if self._gpu:
            layer.init_gpu()

    def get_inference(self):
        ret = []
        c = self.countLayers()
        output = self.getLayerAt(c-1)
        return output._output_array
    
    def export_weight_index(self, path):
        print "Roster : export_weight_index(%s)" % path
        with open(path, "w") as f:
            writer = csv.writer(f, lineterminator='\n')
            c = self.countLayers()
            for i in range(1, c):
                layer = self.getLayerAt(i)
                writer.writerows(layer.export_weight_index())

    def import_weight_index(self, path):
        print "Roster : import_weight_index(%s)" % path
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
    
#    def propagate(self, li=-1, ni=-1, ii=-1, wi=-1, debug=0):
#        c = self.countLayers()
#        if li>0: # propagation with alt value
#            pre = self.getLayerAt(li-1)
#            for i in range(li, c):
#                layer = self.getLayerAt(i)
#                layer.propagate(pre._gpu_output, ni, ii, wi, debug)
#                pre = layer
#
#        else:    # simple propagatione
#            pre = self.getLayerAt(0)
            # this line can be deleted later since input layer is pre-prosessed
            #pre.propagate(pre._gpu_output, -1, -1, -1, debug)
#            for i in range(1, c):
#                layer = self.getLayerAt(i)
#                layer.propagate(pre._gpu_output, -1, -1, -1, debug)
#                pre = layer

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
            if i==li:
                layer.propagate(pre._gpu_output, ni, ii, wi, debug)
            else:
                layer.propagate(pre._gpu_output, -1, -1, -1, debug)
            
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
