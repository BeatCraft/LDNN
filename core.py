#! c:/Python26/python.exe
# -*- coding: utf-8 -*-

import os, sys, time, math
from stat import *
import random
import copy
import multiprocessing as mp
import pickle
import numpy as np

# LDNN Modules
import gpu

#from PIL import Image
#from PIL import ImageFile
#from PIL import JpegImagePlugin
#from PIL import ImageFile
#from PIL import PngImagePlugin
#import zlib

sys.setrecursionlimit(10000)
#
# constant values
#
WEIGHT_SET_0 = [-0.90, -0.85, -0.80, -0.75, -0.70, -0.65, -0.60, -0.55, -0.50,
                -0.45, -0.40, -0.35, -0.30, -0.25, -0.20, -0.15, -0.10, -0.05,
                0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

WEIGHT_SET_1 = [-1.0, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625, -0.0078125,
                0.0,
                0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]

WEIGHT_SET_2 = [0.0, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]

WEIGHT_SET_3 = [-1, -0.72363462, -0.52364706, -0.37892914, -0.27420624, -0.19842513, -0.14358729, -0.10390474,
                -0.07518906, -0.05440941, -0.03937253, -0.02849133, -0.02061731, -0.0149194, -0.01079619, -0.0078125,
                0,
                0.0078125, 0.01079619, 0.0149194, 0.02061731, 0.02849133, 0.03937253, 0.05440941, 0.07518906,
                0.10390474, 0.14358729, 0.19842513, 0.27420624, 0.37892914, 0.52364706, 0.72363462, 1]

WEIGHT_SET_4 = [0.0, 0.0078125, 0.01079619, 0.0149194, 0.02061731, 0.02849133, 0.03937253, 0.05440941, 0.07518906,
                0.10390474, 0.14358729, 0.19842513, 0.27420624, 0.37892914, 0.52364706, 0.72363462, 1.0]


#WEIGHT_SET_5 = [-1.00, -0.95, -0.90, -0.85, -0.80, -0.75, -0.70, -0.65, -0.60, -0.55,
#                -0.50, -0.45, -0.40, -0.35, -0.30, -0.25, -0.20, -0.15, -0.10, -0.05,
#                0.00,
#                0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
#                0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
#
#WEIGHT_SET_6 = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
#                0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
#WEIGHT_SET_3 = [-1.0, -0.5, -0.25, -0.125, 0, 0.125, 0.25, 0.5, 1.0]
#WEIGHT_SET_9 = [0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]

WEIGHT_SET = WEIGHT_SET_1
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
#
#
#
class Weight:
    # layer : Layer class object
    # node  : index of neurons
    # i     : index of neurons in previous layer
    # index : index of WEIGHT_SET / weight index
    # id    : seaquencial number assigned by Roster
    def __init__(self, layer, node, i):
        self._layer = layer
        self._node = node
        self._i = i
        self._index = 0
        self._id = -1
        self._lock = 0
        self._step = 0
    
    def set_id(self, id):
        self._id = id
    
    def get_id(self):
        return self._id
    
    def set(self, w):
        return self._layer.set_weight(self._node, self._i, w)

    def get(self):
        return self._layer.get_weight(self._node, self._i)

    def set_index(self, i):
        if i>=0 and i<WEIGHT_INDEX_SIZE:
            self._index = i
            self.set( WEIGHT_SET[i] )

        return self._index

    def get_index(self):
        return self._index
#
#
#
class Node:
    # layer : Layer class object
    # index : index in the parent layer
    # start : index of the 1st Weghit
    # num   : number of Weight attached
    def __init__(self): #, layer, index, start, num):
#        self._layer = layer
#        self._index = index
#        self._start = start
#        self._num = num
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
        #self._batch_size = batch_size
        self._id = -1
        self._index = i
        self._type = type
        self._num_input = num_input
        self._num_node = num_node
        self._weight_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.float32)
        #self._product_matrix = np.zeros( (self._batch_size, self._num_node, self._num_input), dtype=np.float32)
        self._sum = np.zeros(self._num_node, dtype=np.float32)
        self._y_array = np.zeros(self._num_node*self._batch_size, dtype=np.float32)
        self._node_list = []
    
    def set_batch(self, batch_size):
        self._batch_size = batch_size
        #
        if self._type==0: # input
            self._input_array = np.zeros((self._batch_size, self._num_node), dtype=int)
        else:
            self._input_array = np.zeros((self._batch_size, self._num_node), dtype=np.float32)
            self._product_matrix = np.zeros( (self._batch_size, self._num_node, self._num_input), dtype=np.float32)
        
        self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.float32)
    

    def init_gpu(self):
        if self._gpu:
            pass
        else:
            print "no gpu"
            return
        
        self._gpu_input = self._gpu.dev_malloc(self._input_array)
        if self._type>0:
            self._gpu_weight = self._gpu.dev_malloc(self._weight_matrix)
            self._gpu_product = self._gpu.dev_malloc(self._product_matrix)

        self._gpu_output = self._gpu.dev_malloc(self._output_array)


#    def makeBatchBuffer(self, size, num):
#        self._batch_size = size # ie. 28*28
#        self._batch_num = num   # ie. 100
#        self._batch_array = np.zeros( (num, size), dtype=np.float32)
#        self._gpu_batch_mem = self._gpu.dev_malloc(self._batch_array)
    
    def update_weight(self):
        if self._type>0:
            self._gpu.copy(self._gpu_weight, self._weight_matrix)

    def add_node(self, node):
        self._node_list.append(node)
    
    def get_node(self, i):
        return self._node_list[i]

    def batch_propagate(self, array_in, index, w, wi_alt, debug):
        if self._type==0: # input
#            self._gpu.batch_scale(array_in, self._gpu_output, index, 28*28, float(255.0), self._num_node, 0)
#            if debug==1:
#                self._gpu.copy(self._y_array, self._gpu_output)
#               print "input"
#               print array_in
#                print self._y_array
            pass
        elif self._type==1: # hidden
            if w!=None and w._layer._index==self._index:
                self._gpu.multiple_x_by_w_alt(array_in, self._gpu_weight, self._gpu_product,
                                              self._num_input, self._num_node,
                                              w._i, w._node, WEIGHT_SET[wi_alt])
            else:
                self._gpu.multiple_x_by_w(array_in, self._gpu_weight, self._gpu_product,
                                          self._num_input, self._num_node)
            
            self._gpu.copy(self._product_matrix, self._gpu_product)
            i = 0
            for row in self._product_matrix:
                self._sum[i] = relu( np.sum(row) )
                i += 1
                
            self._gpu.copy(self._gpu_output, self._sum)
            if debug==1:
                print "hidden"
                print self._sum
        else: # output
            if w!=None and w._layer._index==self._index:
                self._gpu.multiple_x_by_w_alt(array_in, self._gpu_weight, self._gpu_product,
                                              self._num_input, self._num_node,
                                              w._i, w._node, WEIGHT_SET[wi_alt])
            else:
                self._gpu.multiple_x_by_w(array_in, self._gpu_weight, self._gpu_product,
                                          self._num_input, self._num_node)

            self._gpu.copy(self._product_matrix, self._gpu_product)
            i = 0
            for row in self._product_matrix:
                self._sum[i] = np.sum(row)
                i += 1
        
            self._y_array = softmax(self._sum)
            if debug==1:
                print self._y_array
    
    def propagate(self, array_in, debug=0):
        if self._type==0:   # input
            self._gpu.copy(self._gpu_input, array_in)
            self._gpu.scale(self._gpu_input, self._gpu_output, float(255.0), self._num_node, 0)
            if debug==1:
                self._gpu.copy(self._y_array, self._gpu_output)
                print "input"
                print array_in
                print self._y_array
            
        elif self._type==1:   # hidden
            self._gpu.multiple_x_by_w(array_in, self._gpu_weight, self._gpu_product,
                                      self._num_input, self._num_node)
            self._gpu.copy(self._product_matrix, self._gpu_product)
            i = 0
            for row in self._product_matrix:
                self._sum[i] = relu( np.sum(row) )
                #self._sum[i] = sigmoid(np.sum(row))
                #self._sum[i] = np.sum(row)
                i += 1
            
            self._gpu.copy(self._gpu_output, self._sum)
            if debug==1:
                print "hidden"
                print self._sum
    
        else: # output
            self._gpu.multiple_x_by_w(array_in, self._gpu_weight, self._gpu_product,
                                      self._num_input, self._num_node)
            self._gpu.copy(self._product_matrix, self._gpu_product)
            i = 0
            for row in self._product_matrix:
                self._sum[i] = np.sum(row)
                i += 1
                    
            self._y_array = softmax(self._sum)
            if debug==1:
                print self._y_array

    def propagate_alt(self, array_in, w, wi_alt):
        if self._type==0: # input
            self._gpu.copy(self._gpu_input, array_in)
            self._gpu.scale(self._gpu_input, self._gpu_output, float(255.0), self._num_node, 0)
        elif self._type==1: # hidden
            if w!=None and w._layer._index==self._index:
                self._gpu.multiple_x_by_w_alt(array_in, self._gpu_weight, self._gpu_product,
                                              self._num_input, self._num_node,
                                              w._i, w._node, WEIGHT_SET[wi_alt])
            else:
                self._gpu.multiple_x_by_w(array_in, self._gpu_weight, self._gpu_product,
                                         self._num_input, self._num_node)
            
            self._gpu.copy(self._product_matrix, self._gpu_product)
            i = 0
            for row in self._product_matrix:
                self._sum[i] = relu( np.sum(row) )
                #self._sum[i] = sigmoid(np.sum(row))
                #self._sum[i] = np.sum(row)
                i += 1

            self._gpu.copy(self._gpu_output, self._sum)
        else: # output
            self._gpu.multiple_x_by_w_alt(array_in, self._gpu_weight, self._gpu_product,
                                          self._num_input, self._num_node,
                                          w._i, w._node, WEIGHT_SET[wi_alt])
            self._gpu.copy(self._product_matrix, self._gpu_product)
            #
            # softmax must be applied below this
            #
            i = 0
            for row in self._product_matrix:
                self._sum[i] = np.sum(row)
                i += 1
            
            self._y_array = softmax(self._sum)
                    
    def propagate_2(self, array_in, index, w, wi_alt, debug):
        if self._type==0: # input
            self._gpu.copy(self._gpu_input, array_in)
            self._gpu.scale(self._gpu_input, self._gpu_output, float(255.0), self._num_node, 0)
        elif self._type==1: # hidden
            self._gpu.multiple_x_by_w_alt(array_in, self._gpu_weight, self._gpu_product,
                                          self._num_input, self._num_node,
                                          w._i, w._node, WEIGHT_SET[wi_alt])
            self._gpu.copy(self._product_matrix, self._gpu_product)
            i = 0
            for row in self._product_matrix:
                self._sum[i] = relu( np.sum(row) )
                #self._sum[i] = sigmoid(np.sum(row))
                #self._sum[i] = np.sum(row)
                i += 1

            self._gpu.copy(self._gpu_output, self._sum)
        else: # output
            if w!=None and w._layer._index==self._index:
                self._gpu.multiple_x_by_w_alt(array_in, self._gpu_weight, self._gpu_product,
                                              self._num_input, self._num_node,
                                              w._i, w._node, WEIGHT_SET[wi_alt])
            else:
                self._gpu.multiple_x_by_w(array_in, self._gpu_weight, self._gpu_product,
                                          self._num_input, self._num_node)
            
            self._gpu.copy(self._product_matrix, self._gpu_product)
   
            i = 0
            for row in self._product_matrix:
                self._sum[i] = np.sum(row)
                i += 1

            self._y_array = softmax(self._sum)
            if debug==1:
                print self._y_array

    def get_weight(self, node, i):
        return self._weight_matrix[node][i]
    
    def set_weight(self, node, i, w):
        self._weight_matrix[node][i] = w

    def set_id(self, id):
        if id>=0:
            self._id = id

    def get_id(self):
        return self._id
    
    def getType(self):
        return self._type

    def get_y_array(self):
        #if self._gpu:
        #    return self._gpu_output
        return self._y_array
#
#
#
class Roster:
    def __init__(self):
        self._weight_list = []
        self._gpu = None
        self.layers = []
        self._batch_size = 1
#        self._batch_in = None
#        self._batch_in_size = 0
#        self._batch_in_num = 0
#        self._gpu_batch_in = None
#        self._batch_out = None
#        self._batch_out_size = 0
#        self._batch_out_num = 0
#        self._gpu_batch_out = None
    
    def set_batch(self, batch, num, size):
        self._batch_data = np.zeros( (num, size), dtype=np.float32)
        self._batch_class = np.zeros(num, dtype=int)
        #
        for layer in self.layers:
            layer.set_batch(num)
        
        for i in range(num):
            entry = batch[i]
            r._batch_data[i] = entry[0].copy()
            r._batch_class[i] = entry[1].copy()
    
        layer = r.getLayerAt(0)
        r._gpu.copy(layer._gpu_input, r._batch_data)
        for i in range(batch_size):
            layer._gpu.batch_scale(layer._gpu_input, layer._gpu_output, i, 28*28, float(255.0), layer._num_node, 0)
    
    def set_gpu(self, my_gpu):
        self._gpu = my_gpu
    
#    def makeBatchBufferIn(self, size, num):
#        self._batch_in_size = size
#        self._batch_in_num = num
#        self._batch_in = np.zeros( (num, size), dtype=np.float32)
#        self._gpu_batch_in = self._gpu.dev_malloc(self._batch_in)
    
#    def makeBatchBufferOut(self, size, num):
#        self._batch_out_size = size
#        self._batch_out_num = num
#        self._batch_out = np.zeros( (num, size), dtype=np.float32)
        #self._gpu_batch_out = self._gpu.dev_malloc(self._batch_out)
    
    def init_weight(self):
        c = 0
        for w in self._weight_list:
            i = random.randrange(WEIGHT_INDEX_SIZE)
            w.set_index(i)
            #w.set_index(WEIGHT_INDEX_ZERO)
            #
            w.set_id(c)
            c += 1

    def unlock_weight_all(self):
        for w in self._weight_list:
            w._lock = 0

    def restore_weighgt(self, w_list):
        c = 0
        for w in self._weight_list:
            wi = int(w_list[c])
            #print wi
            #w.set( WEIGHT_SET[int(wi)] )
            w.set_index(wi)
            w.set_id(c)
            c += 1
            #print "restore : %d, %d" % (wi, w.get_index())

    def update_weight(self):
        for layer in self.layers:
            layer.update_weight()

    def get_weight_list(self):
        return self._weight_list

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
        if c>0: # skip input layer
            for i in range(num_node):
                node = Node()
                layer.add_node(node)
                for j in range(num_input):
                    w = Weight(layer, i, j)
                    self._weight_list.append(w)
                    k = 0
                    k = len(self._weight_list) - 1
                    node.add_weight(k)

        self.layers.append(layer)
        if self._gpu:
            layer.init_gpu()

    def get_inference(self):
        ret = []
        c = self.countLayers()
        output = self.getLayerAt(c-1)
        return output.get_y_array()
    
    def batch_propagate_all(self, w, wi_alt, debug):
        for i in range(self._batch_in_num):
            self.batch_propagate(i, w, wi_alt, debug)
    
    def batch_propagate(self, index, w, wi_alt, debug):
        c = self.countLayers()
        pre = self.getLayerAt(0)
        
        pre._gpu_output = pre._gpu_batch_mem[28*28*index]
        
        #pre.batch_propagate(self._gpu_batch_in, index, w, wi_alt, debug)
        for i in range(1, c):
            layer = self.getLayerAt(i)
            #layer.batch_propagate(pre._gpu_output, index, w, wi_alt, debug)
            layer.batch_propagate(pre._gpu_output, index, w, wi_alt, debug)
            pre = layer
        #
        self._batch_out[index] = pre._y_array
        #
    
    def propagate(self, data, debug=0):
        c = self.countLayers()
        pre = self.getLayerAt(0)
        pre.propagate(data, debug)
        for i in range(1, c):
            layer = self.getLayerAt(i)
            layer.propagate(pre._gpu_output, debug)
            pre = layer

    def propagate_alt(self, data, w, wi_alt):
        c = self.countLayers()
        pre = self.getLayerAt(0)
        pre.propagate_alt(data, w, wi_alt)
        #if w._layer._index==0:
        #    pre.propagate_gpu_alt(data, w, wi_alt)
        #else:
        #    pre.propagate(data)
        for i in range(1, c):
            layer = self.getLayerAt(i)
            if w._layer._index==i:
                layer.propagate_alt(pre._gpu_output, w, wi_alt)
            else:
                layer.propagate(pre._gpu_output)
            pre = layer
#
#
#
def main():
    r = Roster()
    
    input_layer = r.add_layer(0, 196, 196)
    hidden_layer_1 = r.add_layer(1, 196, 32)
    hidden_layer_2 = r.add_layer(1, 32, 32)
    output_layer = r.add_layer(2, 32, 10)
    r.init_weight()
    
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
