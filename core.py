#! c:/Python26/python.exe
# -*- coding: utf-8 -*-

import os, sys, time, math
from stat import *
import random
import copy
#import multiprocessing as mp
import pickle
import numpy as np

if sys.platform.startswith('darwin'):
    pass
else:
    import plat
    if plat.ID==2:
        import cupy as cp
        import cupyx
    #
#
import csv
from PIL import Image

# LDNN Modules
import gpu
import util
#
#sys.setrecursionlimit(10000)
#
# constant values
#
WEIGHT_SET_0 = [-1.0, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625, -0.0078125,
                0.0,
                0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0] # 17
WEIGHT_SET_1 = [-1.0, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625,
                0.0,
                0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0] # 15
WEIGHT_SET_2 = [-1.0, -0.5, -0.25, -0.125, -0.0625, -0.03125,
                0.0,
                0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0] # 13
WEIGHT_SET_3 = [-1.0, -0.5, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 0.5, 1.0] # 11
WEIGHT_SET_4 = [-1.0, -0.5, -0.25, -0.125, 0, 0.125, 0.25, 0.5, 1.0] # 9
WEIGHT_SET_5 = [-1.0, -0.5, -0.25, -0.125, -0.0625, 0.0625, 0.125, 0.25, 0.5, 1.0] # 10
WEIGHT_SET_6 = [-1.0, -0.5, -0.25, -0.125, -0.0625, -0.03125, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0] # 12
WEIGHT_SET_7 = [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0] # 6
WEIGHT_SET_8 = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0] # 9
WEIGHT_SET_9 = [-1.0, -0.5, 0.0, 0.5, 1.0] # 5
#
WEIGHT_SET = WEIGHT_SET_9
WEIGHT_INDEX_SIZE = len(WEIGHT_SET)
WEIGHT_INDEX_ZERO = WEIGHT_INDEX_SIZE/2
WEIGHT_INDEX_MAX = WEIGHT_INDEX_SIZE-1
WEIGHT_INDEX_MIN = 0

CNN_WEIGHT_SET_1 = [-2.0, -1.0, 0.0, 1.0, 2.0]
CNN_WEIGHT_SET_2 = [-1.0, -0.5, 0.0, 0.5, 1.0]
CNN_WEIGHT_SET_3 = [0.0, 1.0]
CNN_WEIGHT_SET_4 = [0.0, 0.5, 1.0]
CNN_WEIGHT_SET = CNN_WEIGHT_SET_4
CNN_WEIGHT_INDEX_SIZE = len(CNN_WEIGHT_SET)
CNN_WEIGHT_INDEX_ZERO = int(CNN_WEIGHT_INDEX_SIZE/2)
CNN_WEIGHT_INDEX_MAX = CNN_WEIGHT_INDEX_SIZE - 1
CNN_WEIGHT_INDEX_MIN = 0
#
#
#
class Weight:
    def __init__(self, li, ni, ii, wi, type=-1):
        self.li = li
        self.ni = ni
        self.ii = ii
        self.wi = wi
        self.wi_alt = wi
        self.mark = 0
        self.type = type
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
LAYER_TYPE_MAX     = 4 # MAX
#
class Layer(object):
    # i         : index of layers
    # type      : 0 = input, 1 = hidden, 2 = output
    # input : stimulus from a previous layer
    # num_input : number of inputs / outputs from a previous layer
    # node : neurons
    # num_node  : numbers of neurons in a layer
    def __init__(self, i, type, num_input, num_node, pre, gpu=None):
        self._pre = None
        self._next = None
        self._pre = pre
        if self._pre:
            self._pre._next = self
        #
        self._index = i
        self._type = type
        if gpu is not None:
            #print("GPU")
            self._gpu = gpu
        else:
            self._gpu = None
        #
        self._id = -1
        self._num_input = num_input
        self._num_node = num_node
        
    def count_weight(self):
        return self._num_node*self._num_input
        
    def get_pre_layer(self):
        return self._pre
        
    def prepare(self, batch_size):
        pass
    
    def get_num_node(self):
        return self._num_node
        
    def get_num_input(self):
        return self._num_input
    
    # gpu must be checked before this method is called
    def update_weight(self):
        pass

    def propagate(self, array_in, debug=0):
        pass
    
    def getWeight(self, ni, ii):
        wi = self._weight_index_matrix[ni][ii]
        w = Weight(self._index, ni, ii, wi, self._type)
        return w
    
    def get_weight_index(self, ni, ii):
        return self._weight_index_matrix[ni][ii]
    
    def set_weight_index(self, ni, ii, wi): # wi : weight index
        self._weight_index_matrix[ni][ii] = wi
        if self._type==LAYER_TYPE_HIDDEN or self._type==LAYER_TYPE_OUTPUT:
            self._weight_matrix[ni][ii] = WEIGHT_SET[wi]
        elif self._type==LAYER_TYPE_CONV_4:
            self._weight_matrix[ni][ii] = CNN_WEIGHT_SET[wi]
        #
    
    def init_weight_mode(self, ni, ii, mode, wi=0):
        if mode==0: # random
            if self._type==LAYER_TYPE_HIDDEN or self._type==LAYER_TYPE_OUTPUT:
                wmax = WEIGHT_INDEX_SIZE
            elif self._type==LAYER_TYPE_CONV_4:
                wmax = CNN_WEIGHT_INDEX_SIZE
            #
            wi = random.randrange(wmax)
            self.set_weight_index(ni, ii, wi)
        elif mode==1: # random in rallow range
            if self._type==LAYER_TYPE_HIDDEN or self._type==LAYER_TYPE_OUTPUT:
                wmin = 1
                wmax = WEIGHT_INDEX_SIZE - 1
            elif self._type==LAYER_TYPE_CONV_4:
                wmin = 0
                wmax = CNN_WEIGHT_INDEX_SIZE -1
            #
            wi = random.randrange(wmin, wmax, 1)
            self.set_weight_index(ni, ii, wi)
        elif mode==2: # fixed value
            self.set_weight_index(ni, ii, wi)
        #
        
    def init_weight_with_mode(self, mode=0, value=0):
        for ni in range(self._num_node):
            for ii in range(self._num_input):
                self.init_weight_mode(ni, ii, mode, value)
            #
        #

    def init_weight(self, ni, ii, wi=-1):
        if wi<0:
            if self._type==LAYER_TYPE_HIDDEN or self._type==LAYER_TYPE_OUTPUT:
                wi = random.randrange(WEIGHT_INDEX_SIZE)
            elif self._type==LAYER_TYPE_CONV_4:
                wi = random.randrange(CNN_WEIGHT_INDEX_SIZE)
            #
        #
        self.set_weight_index(ni, ii, wi)
        
    def init_weight_with_random_index(self):
        for ni in range(self._num_node):
            for ii in range(self._num_input):
                self.init_weight(ni, ii)
            #
        #
    
    def init_weight_with_value(self, value):
        for ni in range(self._num_node):
            for ii in range(self._num_input):
                self.init_weight(ni, ii, value)
            #
        #
    
    def export_weight_index(self):
        return self._weight_index_matrix.tolist()
    
    def import_weight_index(self, wi_list):
        self._weight_index_matrix = np.array(wi_list, dtype=np.int32).copy()
        for ni in range(self._num_node):
            for ii in range(self._num_input):
                wi = self._weight_index_matrix[ni][ii]
                self.set_weight_index(ni, ii, wi)
            #
        #

    def set_id(self, id):
        if id>=0:
            self._id = id

    def get_id(self):
        return self._id
    
    def get_type(self):
        return self._type
        
    def reset(self):
        pass
#
#
#
class InputLayer(Layer):
    def __init__(self, i, num_input, num_node, pre, gpu=None):
        print("InputLayer::__init__()")
        super(InputLayer, self).__init__(i, LAYER_TYPE_INPUT, num_input, num_node, pre, gpu)
        
    def prepare(self, batch_size):
        self._batch_size = batch_size
        self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.float32)
        if self._gpu:
            if self._gpu.type==0:
                self._gpu_output = self._gpu.dev_malloc(self._output_array)
            elif self._gpu.type==1:
                self._gpu_output = self._gpu.allocateArray(self._output_array)
            #
        #

    def propagate(self, array_in, debug=0):
        if debug:
            print("input")
            if self._gpu:
                if self._gpu.type==0: # OpenCL
                    self._gpu.copy(self._output_array, self._gpu_output)
                    print((self._output_array[0]))
                elif self._gpu.type==1: # DGX
                    pass
                #
            #
        #
    
    def set_weight_index(self, ni, ii, wi):
        pass
        
    def get_weight_index(self, ni, ii):
        return 0
        
    def export_weight_index(self):
        return None
        
    def count_weight(self):
        return 0

class HiddenLayer(Layer):
    def __init__(self, i, num_input, num_node, pre, gpu=None):
        print("HiddenLayer::__init__()")
        super(HiddenLayer, self).__init__(i, LAYER_TYPE_HIDDEN, num_input, num_node, pre, gpu)
        
        self._weight_index_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.int32)
        self._weight_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.float32)
        
        if self._gpu:
            if self._gpu.type==0:
                self._gpu_weight = self._gpu.dev_malloc(self._weight_matrix)
            elif self._gpu.type==1:
                self._gpu_weight = self._gpu.allocateArray(self._weight_matrix)
            #
        else:
            print("error")
        #
        self._scale = 3 # scale only
        
    def set_scale(self, mode):
        self._scale = mode
        
    def get_scale(self, mode):
        return self._scale
        # 0 : none
        # 1 : layer normalize
        # 2 : batch normalize
        # 3 : max scale
    
    def prepare(self, batch_size):
        print("HiddenLayer::prepare(%d)" % (batch_size))
        self._batch_size = batch_size
        #self._product_matrix = np.zeros( (self._batch_size, self._num_node, self._num_input), dtype=np.float32)
        self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.float32)

        if self._gpu:
            if self._gpu.type==0:
                #self._product_matrix = np.zeros( (self._batch_size, self._num_node, self._num_input), dtype=np.float32)
                #self._gpu_product = self._gpu.dev_malloc(self._product_matrix)
                self._gpu_output = self._gpu.dev_malloc(self._output_array)
            elif self._gpu.type==1:
                #self._gpu_product = self._gpu.allocateArray(self._product_matrix)
                self._gpu_output = self._gpu.allocateArray(self._output_array)
            #
        else:
            pass
        #

    def update_weight(self):
        if self._gpu:
            if self._gpu.type==0:
                self._gpu.copy(self._gpu_weight, self._weight_matrix)
            elif self._gpu.type==1:
                self._gpu_weight = self._gpu.allocateArray(self._weight_matrix)
            #
        else:
            pass
        #

    def propagate(self, array_in, debug=0):
        if self._gpu:
            pass
        else:
            return
        #
        
        stride_1 = self._num_node * self._num_input
        stride_2 = self._num_input
        # activation mode
        #   0 : none
        #   1 : normal
        #   2 : 0.000001
        #   3 : y/20
        a_mode = 1
        
        if self._gpu.type==0: # OpenCL
            self._gpu.macRelu(array_in, self._gpu_weight, self._gpu_output,
                              self._batch_size, self._num_node, self._num_input, a_mode)
            #self._gpu.copy(self._output_array, self._gpu_output)
            #print((self._output_array.shape))
            #print((self._output_array[0]))
                    
            #self._gpu.multiple_x_by_w_batch(array_in, self._gpu_weight, self._gpu_product,
            #                                self._batch_size, stride_1, stride_2,
            #                                self._num_input, self._num_node)
            #self._gpu.sum(self._gpu_product, self._gpu_output, self._num_input, self._num_node, activation, self._batch_size)
            if self._scale==0:
                pass
            elif self._scale==1:
                self._gpu.normalize_layer(self._gpu_output, self._batch_size, self._num_node)
            elif self._scale==2:
                self._gpu.normalize_batch(self._gpu_output, self._batch_size, self._num_input, self._num_node)
            elif self._scale==3:
                self._gpu.scale_layer(self._gpu_output, self._num_node, self._batch_size)
            elif self._scale==4:
                self._gpu.normalize_layer(self._gpu_output, self._batch_size, self._num_node)
                self._gpu.scale_layer(self._gpu_output, self._num_node, self._batch_size)
            #
            if debug:
                print("scale", self._scale)
                print(self._index, "hidden, input")
                tarray = np.zeros((self._batch_size, self._num_input), dtype=np.float32)
                self._gpu.copy(tarray, array_in)
                print((tarray[0]))
                    
                print(self._index, "hidden")
                self._gpu.copy(self._output_array, self._gpu_output)
                print((self._output_array[0]))
            #
        elif self._gpu.type==1: # DGX
            self._gpu.macRelu3(array_in, self._gpu_weight, self._gpu_output, self._batch_size, self._num_node, self._num_input, a_mode)
            #self._gpu.layerNormalize(self._gpu_output, self._batch_size, self._num_node)
            self._gpu.layerScale(self._gpu_output, self._batch_size, self._num_node)
            #mx = cp.max(self._gpu_output)
            #self._gpu_output = self._gpu_output/mx
            if debug:
                print(self._index, "hidden, input")
                darray = cp.asnumpy(array_in)
                print(darray[0])
                    
                print(self._index, "hidden")
                darray = cp.asnumpy(self._gpu_output)
                print(darray[0])
            #
        #

class OutputLayer(Layer):
    def __init__(self, i, num_input, num_node, pre, gpu=None):#, mode=0):
        print("OutputLayer::__init__()")
        super(OutputLayer, self).__init__(i, LAYER_TYPE_OUTPUT, num_input, num_node, pre, gpu)#, mode)
        #
        self._weight_index_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.int32)
        self._weight_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.float32)

        if self._gpu:
            if self._gpu.type==0:
                self._gpu_weight = self._gpu.dev_malloc(self._weight_matrix)
            elif self._gpu.type==1:
                self._gpu_weight = self._gpu.allocateArray(self._weight_matrix)
            #
        #

    def prepare(self, batch_size):
        print("OutputLayer::prepare(%d)" % (batch_size))
            
        self._batch_size = batch_size
        self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.float32)
        self._softmax_array = np.zeros((self._batch_size, self._num_node), dtype=np.float64)
        
        if self._gpu:
            if self._gpu.type==0:
                self._gpu_output = self._gpu.dev_malloc(self._output_array)
                self._gpu_softmax = self._gpu.dev_malloc(self._softmax_array)
            elif self._gpu.type==1:
                print("output:dgx")
                self._gpu_output = self._gpu.allocateArray(self._output_array)
                self._gpu_softmax = self._gpu.allocateArray(self._softmax_array)
            #
        else:
            pass
        #
        
    def update_weight(self):
        if self._gpu:
            if self._gpu.type==0:
                self._gpu.copy(self._gpu_weight, self._weight_matrix)
            elif self._gpu.type==1:
                self._gpu_weight = self._gpu.allocateArray(self._weight_matrix)
            #
        else:
            pass
        #
        
    def propagate(self, array_in, debug=0):
        stride_1 = self._num_node * self._num_input
        stride_2 = self._num_input
        activation = 1

        if self._gpu:
            if self._gpu.type==0: # OpenCL
                self._gpu.macRelu(array_in, self._gpu_weight, self._gpu_output,
                                  self._batch_size, self._num_node, self._num_input, 0)
                # softmax
                if debug:
                    print("output")
                    self._gpu.copy(self._output_array, self._gpu_output)
                    print((self._output_array[0]))
                #
                self._gpu.softmax(self._gpu_output, self._num_node, self._batch_size)
                if debug:
                    print("softmax")
                    self._gpu.copy(self._output_array, self._gpu_output)
                    print((self._output_array[0]))
                #
            elif self._gpu.type==1: # DGX
                #self._gpu.mac(array_in, self._gpu_weight, self._gpu_output, self._batch_size, self._num_node, self._num_input)
                self._gpu.macRelu3(array_in, self._gpu_weight, self._gpu_output, self._batch_size, self._num_node, self._num_input, 0)
                self._gpu.softmax(self._gpu_output, self._gpu_softmax, self._batch_size, self._num_node)
                if debug:
                    print("softmax", self._index)
                    darray = cp.asnumpy(self._gpu_softmax)
                    print(darray[0])
                #
            #
        else:
            pass
        #

class RegressionOutputLayer(Layer):
    def __init__(self, i, num_input, num_node, pre, gpu=None):
        print("RegressionOutputLayer::__init__()")
        super(RegressionOutputLayer, self).__init__(i, LAYER_TYPE_OUTPUT, num_input, num_node, pre, gpu)#, mode)
        #
        self._weight_index_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.int32)
        self._weight_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.float32)
        #
        if gpu:
            if self._gpu.type==0:
                self._gpu_weight = self._gpu.dev_malloc(self._weight_matrix)
            elif self._gpu.type==1:
                print("error")
            #
        else:
            print("error")
        #
        
    def prepare(self, batch_size):
        self._batch_size = batch_size
        self._product_matrix = np.zeros( (self._batch_size, self._num_node, self._num_input), dtype=np.float32)
        self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.float32)
        #
        if self._gpu:
            if self._gpu.type==0:
                self._gpu_product = self._gpu.dev_malloc(self._product_matrix)
                self._gpu_output = self._gpu.dev_malloc(self._output_array)
            elif self._gpu.type==1:
                pass
            #
        else:
            print("error")
        #
    
    def update_weight(self):
        if self._gpu:
            if self._gpu.type==0:
                self._gpu.copy(self._gpu_weight, self._weight_matrix)
            elif self._gpu.type==1:
                pass
            #
        else:
            print("error")
        #

    def propagate(self, array_in, debug=0):
        stride_1 = self._num_node * self._num_input
        stride_2 = self._num_input
        # multiple
        if self._gpu:
            if self._gpu.type==0:
                self._gpu.multiple_x_by_w_batch(array_in, self._gpu_weight, self._gpu_product,
                                                self._batch_size, stride_1, stride_2,
                                                self._num_input, self._num_node)
                # sum
                activation = 1 # relu=0, skip=1
                self._gpu.sum(self._gpu_product, self._gpu_output,
                            self._num_input, self._num_node, activation, self._batch_size)
                #
                if debug:
                    print("output", self._index)
                    self._gpu.copy(self._output_array, self._gpu_output)
                    print((self._output_array[0]))
                #
            elif self._gpu.type==1:
                pass
            #
        else:
            pass
        #
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
        self.lock = False
    
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
        
    def prepare(self, batch_size):
        print("MaxLayer::prepare(%d)" % (batch_size))
        self._batch_size = batch_size
        self._output_array = np.zeros((self._batch_size, self._ch, self._num_node), dtype=np.float32)
        #
        if self._gpu:
            if self._gpu.type==0:
                self._gpu_output = self._gpu.dev_malloc(self._output_array)
            elif self._gpu.type==1:
                self._gpu_output = self._gpu.allocateArray(self._output_array)
            #
        else:
            print("error")
        #
        
    def propagate(self, array_in, debug=0):
        if self.lock:
            return
        #
        if self._gpu:
            pass
        else:
            return
        #
        
        if self._gpu.type==0: # opencl
            self._gpu.max_batch(array_in, self._gpu_output,
                                self._ch, self._x, self._y, self._batch_size)
            if debug:
                print(self._index, "max")
                self._gpu.copy(self._output_array, self._gpu_output)
                print(self._output_array[0].shape)
                print(self._output_array[0][0])
            #
        elif self._gpu.type==1: # GDX
            self._gpu.max(array_in, self._gpu_output, self._ch, self._x, self._y, self._batch_size)
        #

class Conv_4_Layer(Layer):
    def __init__(self, i, w, h, ch, filter, pre, gpu=None):
        print("Convolution Layer ver.4 ::__init__()")
        
        self._w = w
        self._h = h
        self._ch = ch # number of inputs
        self._filter = filter # node / # number of outputs
        self._filter_size = 3 * 3 * ch # width and height of filter are fixed to 3
        self._num_of_w = 3 * 3 * ch# * filter
        num_input = self._num_of_w # self._filter_size
        num_node = self._filter
        #
        super(Conv_4_Layer, self).__init__(i, LAYER_TYPE_CONV_4, num_input, num_node, pre, gpu)
        #
        # mems for weights
        self._weight_index_matrix = np.zeros( (self._filter, self._num_of_w), dtype=np.int32)
        self._weight_matrix = np.zeros( (self._filter, self._num_of_w), dtype=np.float32)
        #
        if self._gpu:
            if self._gpu.type==0:
                self._gpu_weight = self._gpu.dev_malloc(self._weight_matrix)
            elif self._gpu.type==1:
                self._gpu_weight = self._gpu.allocateArray(self._weight_matrix)
            #
        else:
            print("error")
        #
        self._cache = 0 # cache for padding
        self.lock = False
    
    #def set_weight_index(self, ni, ii, wi):
    #    self.reset_weight_index(ni, ii):
    
    def reset_weight_index(self, ni, ii, wi=-1):
        #print("Conv_4_Layer::reset_weight_index(%d, %d)" % (ni, ii))
        ch = int(ii/9)
        #print(ch)
        wi_list = []
        for i in range(9):
            wi = self.get_weight_index(ni, i)
            wi_list.append(wi)
            self.init_weight(ni, i)
        #
        return wi_list
        
    def prepare(self, batch_size):
        print(("Conv_4_Layer::prepare(%d)" %(batch_size)))
        self._batch_size = batch_size
        # intermidiate
        self._padded_array = np.zeros((self._batch_size, (self._w+2)*(self._h+2)*self._ch), dtype=np.float32)

        # output
        self._output_array = np.zeros((self._batch_size, self._filter, self._w*self._h), dtype=np.float32)
        if self._gpu:
            if self._gpu.type==0:
                self._gpu_padded = self._gpu.dev_malloc(self._padded_array)
                self._gpu_output = self._gpu.dev_malloc(self._output_array)
            elif self._gpu.type==1:
                self._gpu_padded = self._gpu.allocateArray(self._padded_array)
                self._gpu_output = self._gpu.allocateArray(self._output_array)
            #
        else:
            print("error")
        #

    def update_weight(self):
        if self._gpu:
            pass
        else:
            return
        #
        
        if self._gpu.type==0:
            self._gpu.copy(self._gpu_weight, self._weight_matrix)
        elif self._gpu.type==1:
            self._gpu_weight = self._gpu.allocateArray(self._weight_matrix)
        #
    
    def reset(self):
        self._cache = 0
        
    def propagate(self, array_in, debug=0):
        if self.lock:
            return
        #
        if self._gpu:
            pass
        else:
            return
        #
        
        # activation mode
        # 0 : none
        # 1 : normal
        # 2 : 0.000001
        # 3 : y/20
        a_mode = 1
        if self._gpu.type==0: # OpenCL
            self._gpu.conv_4_pad_batch(array_in, self._gpu_padded, self._w, self._h, self._ch, self._batch_size)
            if debug:
                print(self._index, "conv pad")
                self._gpu.copy(self._padded_array, self._gpu_padded)
                print(self._padded_array[0])
                self.save_padded(0, 0, self._padded_array[0])
                self.save_padded(0, 1, self._padded_array[0])
                self.save_padded(0, 2, self._padded_array[0])
            #
            self._gpu.conv_4_roll_batch(self._gpu_padded, self._gpu_weight, self._gpu_output,
                                        self._w, self._h, self._ch, self._filter,
                                        self._batch_size, 0)
            if debug:
                print(self._index, "conv ret")
                self._gpu.copy(self._output_array, self._gpu_output)
                print((self._output_array[0][0]))
            #
            #self._gpu.normalize_batch(self._gpu_output, self._batch_size, self._w * self._h, self._filter)
            size = self._w * self._h * self._filter
            #self._gpu.relu(self._gpu_output, self._batch_size, self._filter, size, a_mode)
            #self._gpu.scale_layer(self._gpu_output, size, self._batch_size)
            self._gpu.scale_filetr(self._batch_size, self._filter, self._gpu_output, size, self._w * self._h)
            if debug:
                print(self._index, "conv, scale")
                self._gpu.copy(self._output_array, self._gpu_output)
                print((self._output_array[0][0]))
                #
                #self.save_output()
                self.save_filter_out(0, 0, self._output_array[0][0])
            #
        elif self._gpu.type==1: # GDX
            self._gpu.padding(array_in, self._gpu_padded, self._w, self._h, self._ch, self._batch_size)
            if debug:
                print(self._index, "conv", "padded", )
                darray = cp.asnumpy(self._gpu_padded)
                print((darray.shape, type(darray[0])))
            #
            self._gpu.convolusion(self._gpu_padded, self._gpu_weight, self._gpu_output, self._w, self._h, self._ch, self._filter, self._batch_size)
            if debug:
                print(self._index, "conv ret")
                darray = cp.asnumpy(self._gpu_output)
                print((darray.shape, type(darray[0])))
                print(darray[0][0])
            #
            size = self._w * self._h * self._filter
            #self._gpu.relu(self._gpu_output, size, a_mode, self._batch_size, self._filter)
            if debug:
                print(self._index, "conv, scale")
                darray = cp.asnumpy(self._gpu_output)
                print(darray[0][0])
            #
            self._gpu.filterScale(self._batch_size, self._filter, self._gpu_output, size, self._w * self._h)
        #
    
    def save_padded(self, bi, ci, data_array):
        w = self._w + 2
        h = self._h + 2
        size = w * h
        max = np.max(data_array)
        min = np.min(data_array)
        print(("max=%f, min=%f" % (max, min)))
        
        img = Image.new("L", (w, h), 0)
        pix = img.load()
        for y in range(h):
            for x in range(w):
                v = data_array[size*ci + w*y + x]
                if max>0.0:
                    v1 = int(v*255/max)
                    pix[x,y] = v1
                else:
                    pix[x,y] = 0
                #
            #
        spath = "./debug/%d-%d-%d-p.png" % (self._index, bi, ci)
        print(spath)
        img.save(spath)
    
    def save_filter_out(self, bi, fi, data_array):
        size = self._w * self._h
        max = np.max(data_array)
        min = np.min(data_array)
        print(("max=%f, min=%f" % (max, min)))
        
        img = Image.new("L", (self._w, self._h), 0)
        pix = img.load()
        for y in range(self._h):
            for x in range(self._w):
                v = data_array[self._w*y + x]
                if max>0.0:
                    v1 = int(v*255/max)
                    pix[x,y] = v1
                else:
                    pix[x,y] = 0
                #
            #
        spath = "./debug/%d-%d_%d.png" % (self._index, bi, fi)
        print(spath)
        img.save(spath)
    
    def save_output(self):
        self._gpu.copy(self._output_array, self._gpu_output)
        #
        for bi in range(self._batch_size):
            for fi in range(self._filter):
                data_array = self._output_array[bi][fi]
                self.save_filter_out(bi, fi, data_array)
            #
        #
#
#
#
class Roster:
    def __init__(self):
        self._weight_list = []
        self._gpu = None
        self.layers = []
        self.input = None
        self.output = None
        self._batch_size = 1
        self._data_size = 1
        self._eval_mode = 0
        self._path = ""
        self._scale_input = 0
        #self.mem_save = 1
        
    def set_path(self, path):
        self._path = path
        
    def save(self):
        print("Roster::save(%s)" % (self._path))
        self.export_weight(self._path)
        
    def save_as(self, path):
        print("Roster::save(%s)" % (path))
        self.export_weight(path)
    
    def load(self):
        print("Roster::load(%s)" % (self._path))
        if os.path.isfile(self._path):
            self.import_weight(self._path)
        else:
            mode = 1
            value = 0
            self.init_weight(mode, value)
            self.export_weight(self._path)
        #

    def set_evaluate_mode(self, mode):
        self._eval_mode = mode
        # 0 : CE for classification
        # 1 : MSE for autoencoder
        # 2 : MSE for regression
    
    def set_gpu(self, gpu):
        self._gpu = gpu
        self._remote = None

    def prepare(self, batch_size, data_size, num_class):
        self.num_class = num_class
        self._batch_size = batch_size
        self._data_size = data_size
        #
        self._batch_data = np.zeros((self._batch_size, data_size), dtype=np.float32)
        self._labels = np.zeros((batch_size, num_class), dtype=np.float32)
        #self._batch_cross_entropy = np.zeros(batch_size, dtype=np.float64)
        if self._gpu:
            if self._gpu.type==0: # OpenCL
                self._batch_cross_entropy = np.zeros(batch_size, dtype=np.float32)
                self._gpu_input = self._gpu.dev_malloc(self._batch_data)
                self._gpu_labels = self._gpu.dev_malloc(self._labels)
                self._gpu_entropy = self._gpu.dev_malloc(self._batch_cross_entropy)
            elif self._gpu.type==1: # GDX
                self._batch_cross_entropy = np.zeros(batch_size, dtype=np.float64)
                #if self.mem_save==0:
                #    self._gpu_input = self._gpu.allocateArray(self._batch_data)
                #
                self._gpu_labels = self._gpu.allocateArray(self._labels)
                self._gpu_entropy = self._gpu.allocateArray(self._batch_cross_entropy)
            #
        #
        self.input = self.get_layer_at(0)
        for layer in self.layers:
            layer.prepare(batch_size)
        #
        self.output = layer
    
    # batch for classification
    def set_batch(self, data_size, num_class, train_data_batch, train_label_batch, size, offset):
        print("Roster::set_batch(%d, %d, %d, %d)" % (data_size, num_class, size, offset))
        data_array = np.zeros((size, data_size), dtype=np.float32)
        labels = np.zeros((size, num_class), dtype=np.float32)
        for j in range(size):
            data_array[j] = train_data_batch[offset+j]
            #print(offset+j, type(train_label_batch[offset+j]))
            k = int(train_label_batch[offset+j])
            #print(k)
            labels[j][k] = 1.0
        #
        self.set_data(data_array, data_size, labels, size)
                
    def set_data(self, data, data_size, label, batch_size):
        print("Roster::set_data(%d, %d)" % (data_size, batch_size))
        if self._gpu:
            pass
        else:
            return
        #
        self.reset()
        
        if self._gpu.type==0: # OpenCL
            self._gpu.copy(self._gpu_input, data)
            self._gpu.copy(self._gpu_labels, label)
            if self._scale_input==0: # none
                self._gpu.copy(self.input._gpu_output, data)
            elif self._scale_input==1: # scale
                self._gpu.scale(self._gpu_input, self.input._gpu_output, data_size, float(255.0), self.input._num_node, batch_size, 0)
            elif self._scale_input==2: # batch normalize
                data = data.transpose()
                k = data.shape[0]
                for i in range(k):
                    xmean = data[i].mean()
                    xstd  = np.std(data[i])
                    zscore = (data[i]-xmean)/xstd
                    max = zscore.max()
                    min = zscore.min()
                    if max < abs(min):
                        max = abs(min)
                    #
                    zscore = zscore / max
                    data[i] = zscore / 2 + 0.5
                #
                data = data.transpose()
                print("3", data.shape)
                k = data.shape[0]
                self._gpu.copy(self.input._gpu_output, data)
            else:
                return
            #
        elif self._gpu.type==1: # GDX
            if self._scale_input==0: # none
                self.input._gpu_output = self._gpu_input
            elif self._scale_input==1: # scale 0.0 - 1.0
                #if self.mem_save==0:
                #    self._gpu_input = self._gpu.allocateArray(data)
                #    self._gpu_labels = self._gpu.allocateArray(label)
                #    self._gpu_input = self._gpu_input / 255.0
                #    self.input._gpu_output = self._gpu_input
                #else:
                #    self._gpu_labels = self._gpu.allocateArray(label)
                #    #temp = self._gpu.allocateArray(data)
                #    temp = data / 255.0
                #    self.input._gpu_output = self._gpu.allocateArray(temp)
                self._gpu_labels = self._gpu.allocateArray(label)
                temp = data / 255.0
                self.input._gpu_output = self._gpu.allocateArray(temp)
            else:
                return
            #
        else:
            return
        #
    
    def set_batch_data(self, data_size, train_data_batch, size, offset, scale=0):
        print("Roster::set_batch_data(%d, %d, %d, %d)" % (data_size, size, offset, scale))
        
        data_array = np.zeros((size, data_size), dtype=np.float32)
        for j in range(size):
            data_array[j] = train_data_batch[offset+j]
        #
        self.reset()
        
        self._gpu.copy(self._gpu_input, data_array)

        if self._scale_input==0:
            self._gpu.copy(self.input._gpu_output, self._gpu_input)
        elif self._scale_input==1:
            x_gpu = self._gpu.allocateArray(self._gpu_input)
            y_gpu = x_gpu / 255.0
            self.output._gpu_output = self._gpu.allocateArray(y_gpu)
        elif self._scale_input==2:
            self._gpu.scale_exp(self._gpu_input, self.input._gpu_output, data_size, self.input._num_node, size, 0)
        else:
            pass
        #
        
    def set_batch_label(self, data_size, train_label_batch, size, offset, scale=0):
        print("Roster::set_batch_label(%d, %d, %d, %d)" % (data_size, size, offset, scale))
        labels = np.zeros((size, data_size), dtype=np.float32)
        for j in range(size):
            labels[j] = train_label_batch[offset+j]
        #
        self.reset()
        self._gpu.copy(self._gpu_labels, labels)
    
    
    def direct_set_data(self, data_array):
        self._gpu.copy(self._gpu_input, data_array)
        self._gpu.copy(self.input._gpu_output, self._gpu_input)
        
    def direct_set_label(self, label_array):
        self._gpu.copy(self._gpu_labels, label_array)
    
    def set_scale_input(self, scale):
        self._scale_input = scale
    
    def denominate(self, all=False):
        print("Roster : denominate()")
        c = self.count_layers()
        for i in range(c):
            layer = self.get_layer_at(i)
            type = layer.get_type()
            if type==LAYER_TYPE_MAX or type==LAYER_TYPE_INPUT:
                pass
            else:
                layer.denominate()
            #
        #

    def init_weight(self, mode=0, value=0):
        c = self.count_layers()
        for i in range(c):
            layer = self.get_layer_at(i)
            type = layer.get_type()
            if type==LAYER_TYPE_MAX or type==LAYER_TYPE_INPUT:
                pass
            else:
                #layer.init_weight_with_random_index()
                layer.init_weight_with_mode(mode, value)
            #
        #
        
    def init_weight_by_layer(self, idx, mode, value=0):
        layer = self.get_layer_at(idx)
        if layer.count_weight()>0:
            if mode==0: # random
                layer.init_weight_with_random_index()
            elif mode==1: # a value
                layer.init_weight_with_value(value)
            #
        #
        
    def reset(self):
    # flush a batch depending cache when switching batches
        c = self.count_layers()
        for i in range(c):
            layer = self.get_layer_at(i)
            layer.reset()
        #
        
    def reset_weight_property(self, p=0):
        c = self.count_layers()
        for i in range(1, c):
            layer = self.get_layer_at(i)
            layer.reset_weight_property_all()
        #
    
    def count_weight(self):
        cnt = 0
        c = self.count_layers()
        for i in range(1, c):
            layer = self.get_layer_at(i)
            cnt = cnt + layer.count_weight()
        #
        return cnt

    def update_weight(self):
        for layer in self.layers:
            layer.update_weight()
        #

    def count_layers(self):
        return len(self.layers)

    def get_layers(self):
        if self.count_layers() == 0:
            return 0
        #
        return self.layers
    
    def get_layer_at(self, i):
        c = self.count_layers()
        if i>=c:
            print("error : Roster : get_layer_at")
            return None
        #
        return self.layers[i]

    def add_layer(self, type, num_input, num_node):
        c = self.count_layers()
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
        c = self.count_layers()
        output = self.get_layer_at(c-1)
        output._gpu.copy(output._output_array, output._gpu_output)
        return output._output_array

    def get_answer(self):
        print("roster::get_answer()")
        ret = []
        #c = self.count_layers()
        output = self.output #self.get_layer_at(c-1)
        if self._gpu:
            if self._gpu.type==0:
                output._gpu.copy(output._output_array, output._gpu_output)
            elif self._gpu.type==1:
                output._output_array = self._gpu.allocateArray(output._gpu_softmax)
            #
        else:
            pass
        #

        for i in range(self._batch_size):
            if self._gpu.type==0:
                inf = output._output_array[i]
            elif self._gpu.type==1: # cupy
                inf = cp.asnumpy(output._output_array)[i]
            else:
                return -1
            #
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
    
    def evaluate(self, debug=0):
        #print("Roster::evaluate()")
        self.propagate(debug)
        #
        if self._eval_mode==0: # CE for classification
            ce = self.get_cross_entropy(debug)
        elif self._eval_mode ==1: # MSE for autoencoder
            self._gpu.mse(self.output._gpu_output, self.input._gpu_output, self._gpu_entropy, self._data_size, self._batch_size)
            self._gpu.copy(self._batch_cross_entropy, self._gpu_entropy)
            ce = np.sum(self._batch_cross_entropy)/np.float32(self._batch_size)
        elif self._eval_mode ==2: # MSE for regression
            self._gpu.mse(self.output._gpu_output, self._gpu_labels, self._gpu_entropy, self._data_size, self._batch_size)
            self._gpu.copy(self._batch_cross_entropy, self._gpu_entropy)
            ce = np.sum(self._batch_cross_entropy)/np.float32(self._batch_size)
        #
        return ce
    
    def get_cross_entropy(self, debug=0):
        #print("Roster::get_cross_entropy()")
        c = self.count_layers()
        output = self.get_layer_at(c-1)

        if self._gpu:
            if self._gpu.type==0: # OenCL
                self._gpu.cross_entropy(output._gpu_output, self._gpu_labels, self._gpu_entropy, self.num_class, self._batch_size)
                self._gpu.copy(self._batch_cross_entropy, self._gpu_entropy)
                
                if debug:
                    print(self._batch_cross_entropy)
                    print("bsize", self._batch_size)
                    print("shape", self._batch_cross_entropy.shape)
                    print("sum", np.sum(self._batch_cross_entropy))
                    print("avg", np.sum(self._batch_cross_entropy)/self._batch_size)
                    k = 0.0
                    for i in range(self._batch_size):
                        k += self._batch_cross_entropy[i]
                    #
                    print(k)
                    
                    
                #
                s = np.sum(self._batch_cross_entropy)
                s = s/float(self._batch_size)
                #
                # debug
                #
                if debug and np.isnan(s):
                    for i in range(self._batch_size):
                        li = c-1
                        if np.isnan(self._batch_cross_entropy[i]):
                            print(("NaN : %d" % (i)))
                            for li in range(c):
                                output = self.get_layer_at(li)
                                self._gpu.copy(output._output_array, output._gpu_output)
                                print(("layer : %d" % (li)))
                                print((output._output_array[i].shape))
                                print((output._output_array[i]))
                            #
                        #
                    #
                #
                return s
            elif self._gpu.type==1: # DGX
                self._gpu.crossEntropy(output._gpu_softmax, self._gpu_labels, self._gpu_entropy, self._batch_size, output._num_node)
                total = self._gpu_entropy.sum()
                avg = total / float(self._batch_size)
                
                if debug:
                    print("get_cross_entropy")
                    darray = cp.asnumpy(self._gpu_entropy)
                    print(darray)
                #
                return avg
            #
        #
        return 0.0

    def export_weight(self, path):
        print(("Roster : export_weight(%s)" % path))
        self.export_weight_index(path)
        
    def export_weight_index(self, path):
        with open(path, "w") as f:
            writer = csv.writer(f, lineterminator='\n')
            c = self.count_layers()
            for i in range(1, c):
                layer = self.get_layer_at(i)
                data = layer.export_weight_index()
                if data:
                    writer.writerows(data)
                #
            #
        #
    
    def export_weight_index_by_layer(self, idx):
        layer = self.get_layer_at(idx)
        data = layer.export_weight_index()
        path = "./wi/%d.csv" % (idx)
        with open(path, "w") as f:
            writer = csv.writer(f, lineterminator='\n')
            if data:
                writer.writerows(data)
            #
        #
    
    def import_weight_index_by_layer(self, idx):
        path = "./wi/%d.csv" % (idx)
        with open(path, "r") as f:
            reader = csv.reader(f)
            layer = self.get_layer_at(idx)
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
        #
        
    def import_weight(self, path):
        print(("Roster::import_weight(%s)" % path))
        self.import_weight_index(path)
        
    def import_weight_index(self, path):
        #print("Roster : import_weight_index(%s)" % path)
        with open(path, "r") as f:
            reader = csv.reader(f)
            lc = self.count_layers()
            for i in range(1, lc):
                layer = self.get_layer_at(i)
                type = layer.get_type()
                if type==LAYER_TYPE_INPUT or type==LAYER_TYPE_MAX:
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

    def propagate(self, debug=0):
        c = self.count_layers()
        pre = self.get_layer_at(0)
        for i in range(1, c):
            #print(i)
            layer = self.get_layer_at(i)
            layer.propagate(pre._gpu_output, debug)
            #
            pre = layer
        #
        #print("end of propagate()")
#
#
#
def main():
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
