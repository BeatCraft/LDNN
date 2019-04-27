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

WEIGHT_SET = WEIGHT_SET_2
WEIGHT_INDEX_SIZE = len(WEIGHT_SET)
WEIGHT_INDEX_ZERO = 0#WEIGHT_INDEX_SIZE/2
WEIGHT_INDEX_MAX = WEIGHT_INDEX_SIZE-1
WEIGHT_INDEX_MIN = 0
#WEIGHT_RANDOM_RANGE = 6
#
#
#
def sigmoid(x):
    a = 0.0
    try:
        a = np.exp(-x)#math.exp(-x)
    except OverflowError:
        a = float('inf')
        print "sigmoid(fuck)"
    
    ret = 1.0 / (1.0 + a)
    #print "sigmoid(%f)=%f" % (x, ret)
    return ret#1.0 / (1.0 + a)

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
    return np.exp(x)
#
#
#
class Weight:
    # layer : Layer class object
    # node  : index of neurons
    # i     : index of neurons in previous layer
    # index : index of lookup table / weight index
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
        self._weight_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.float32)
        self._product_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.float32)
        self._sum = np.zeros(self._num_node, dtype=np.float32)
        self._y_array = np.zeros(self._num_node, dtype=np.float32)
        #self._weight_list = []

        self.nodes = []
            
        if self._type==0:
            self._input_array = np.zeros(num_node, dtype=int)
        else:
            self._input_array = np.zeros(num_node, dtype=np.float32)
        self._output_array = np.zeros(num_node, dtype=np.float32)
    
    def init_gpu(self):
        #print "init_gpu()"
        if self._gpu:
            pass
        else:
            print "no gpu"
            return
        
        self._gpu_input = self._gpu.dev_malloc(self._input_array)
        if self._type>0:
            self._gpu_weight = self._gpu.dev_malloc(self._weight_matrix)
            self._gpu_product = self._gpu.dev_malloc(self._weight_matrix)
        #self._gpu_sum = self._gpu.dev_malloc(self._output_array)
        self._gpu_output = self._gpu.dev_malloc(self._output_array)
        #print self._output_array

    def update_weight(self):
        if self._type>0:
            self._gpu.write(self._gpu_weight, self._weight_matrix)
            #print self._weight_matrix
    
#    def propagate(self, data):
#        if self._gpu:
#            self.propagate_gpu(data)
#        else:
#            self.propagate_cpu(data)
#
#    def propagate_cpu(self, array_in):
#        if self._type==0:   # input
#            self._y_array = array_in/255.0
#        elif self._type==1: # hidden
#            for i in range(self._num_node):
#                sum = np.sum(self._weight_matrix[i]*array_in)
#                self._y_array[i] = relu(sum)
#        elif self._type==2: # output
#            for i in range(self._num_node):
#                sum = np.sum(self._weight_matrix[i]*array_in)
#                self._y_array[i] = np.exp(sum)
#
    def propagate(self, array_in):
        if self._type==0:   # input
            self._gpu.copy(self._gpu_input, array_in)
            self._gpu.scale(self._gpu_input, self._gpu_output, float(255.0), self._num_node)
        elif self._type==1:   # hidden
            self._gpu.multiple_x_by_w(array_in, self._gpu_weight, self._gpu_product,
                                      self._num_input, self._num_node)
            self._gpu.copy(self._product_matrix, self._gpu_product)
            i = 0
            for row in self._product_matrix:
                #self._sum[i] = relu( np.sum(row) )
                #self._sum[i] = sigmoid(np.sum(row))
                self._sum[i] = np.sum(row)
                i += 1
            
            self._gpu.copy(self._gpu_output, self._sum)
        else: # output
            self._gpu.multiple_x_by_w(array_in, self._gpu_weight, self._gpu_product,
                                      self._num_input, self._num_node)
            self._gpu.copy(self._product_matrix, self._gpu_product)
            i = 0
            for row in self._product_matrix:
                self._sum[i] = np.sum(row)
                i += 1

            sum = np.sum(self._sum)
            i = 0
            for row in self._sum:
                if sum>0.0:
                    self._y_array[i] = row/sum
                else:
                    self._y_array[i] = 0.0
                    print "FUCK"
                i += 1

    def propagate_alt(self, array_in, w, wi_alt):
        if self._type==0: # input
            self._gpu.copy(self._gpu_input, array_in)
            self._gpu.scale(self._gpu_input, self._gpu_output, float(255.0), self._num_node)
        elif self._type==1: # hidden            
            self._gpu.multiple_x_by_w_alt(array_in, self._gpu_weight, self._gpu_product,
                                          self._num_input, self._num_node,
                                          w._i, w._node, WEIGHT_SET[wi_alt])
            self._gpu.copy(self._product_matrix, self._gpu_product)
            i = 0
            for row in self._product_matrix:
                #self._sum[i] = relu( np.sum(row) )
                #self._sum[i] = sigmoid(np.sum(row))
                self._sum[i] = np.sum(row)
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
                                      
            sum = np.sum(self._sum)
            i = 0
            for row in self._sum:
                if sum>0.0:
                    self._y_array[i] = row/sum
                else:
                    self._y_array[i] = 0.0
                    print "FUCK"
        
                i += 1

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

    def set_gpu(self, my_gpu):
        self._gpu = my_gpu
    
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
        if c>0:
            for i in range(num_node):
                for j in range(num_input):
                    self._weight_list.append( Weight(layer, i, j) )

        self.layers.append(layer)

        if self._gpu:
            layer.init_gpu()

#    def get_inference(self, softmax=0):
#        ret = []
#        c = self.countLayers()
#        output = self.getLayerAt(c-1)
#        y_array = output.get_y_array()
#        sum = np.sum(y_array)
#        for a in y_array:
#            ret.append(a/sum)
#
#        return ret

    def get_inference(self):
        ret = []
        c = self.countLayers()
        output = self.getLayerAt(c-1)
        return output.get_y_array()
    
#    def propagate(self, data):
#        if self._gpu:
#            self.propagate_gpu(data)
#        else:
#            self.propagate_cpu(data)
#
#    def propagate_cpu(self, data):
#        c = self.countLayers()
#        pre = self.getLayerAt(0)
#        pre.propagate(data)
#        for i in range(1, c):
#            array_y = pre.get_y_array()
#            layer = self.getLayerAt(i)
#            layer.propagate(array_y)
#            pre = layer
#
    def propagate(self, data):
        c = self.countLayers()
        pre = self.getLayerAt(0)
        pre.propagate(data)
        for i in range(1, c):
            layer = self.getLayerAt(i)
            layer.propagate(pre._gpu_output)
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
