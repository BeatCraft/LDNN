#! c:/Python26/python.exe
# -*- coding: utf-8 -*-

import os, sys, time, math
from stat import *
import random
import copy
import pickle
import numpy as np

import random
from scipy.stats import norm

import pyopencl as cl

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
# constant values
#
WEIGHT_SET_0 = [-1.0, -0.5, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 0.5, 1.0] # 11
WEIGHT_SET_1 = [-1.0, -0.5, -0.25, -0.125, 0.0, 0.125, 0.25, 0.5, 1.0] # 9
WEIGHT_SET_2 = [-1.0, -0.5, -0.25, -0.125, 0.125, 0.25, 0.5, 1.0] # 8
WEIGHT_SET_3 = [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
 0,
 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # 21

#
WEIGHT_SET = WEIGHT_SET_3
WEIGHT_INDEX_SIZE = len(WEIGHT_SET)
WEIGHT_INDEX_ZERO = int(WEIGHT_INDEX_SIZE/2)
WEIGHT_INDEX_MAX = WEIGHT_INDEX_SIZE-1
WEIGHT_INDEX_MIN = 0

CNN_WEIGHT_SET_1 = [-2.0, -1.0, 0.0, 1.0, 2.0]
CNN_WEIGHT_SET_2 = [-1.0, -0.5, 0.0, 0.5, 1.0]
CNN_WEIGHT_SET_3 = [0.0, 1.0]
CNN_WEIGHT_SET_4 = [-0.25, -0.125, 0.0, 0.125, 0.25, 0.5]
CNN_WEIGHT_SET_5 = [-0.5, 0.0, 0.5]

CNN_WEIGHT_SET = WEIGHT_SET_0 # CNN_WEIGHT_SET_4
CNN_WEIGHT_INDEX_SIZE = len(CNN_WEIGHT_SET)
CNN_WEIGHT_INDEX_ZERO = int(CNN_WEIGHT_INDEX_SIZE/2)
CNN_WEIGHT_INDEX_MAX = CNN_WEIGHT_INDEX_SIZE - 1
CNN_WEIGHT_INDEX_MIN = 0

CNN_WEIGHT_SET2 = CNN_WEIGHT_SET_5
CNN_WEIGHT_INDEX_SIZE2 = len(CNN_WEIGHT_SET2)
CNN_WEIGHT_INDEX_ZERO2 = int(CNN_WEIGHT_INDEX_SIZE2/2)
CNN_WEIGHT_INDEX_MAX2 = CNN_WEIGHT_INDEX_SIZE2 - 1
CNN_WEIGHT_INDEX_MIN2 = 0

RNDWT = [norm.pdf(x, 0, 1) for x in WEIGHT_SET]
RNDWT[5] *= 0.1

def wi_std_11():
    idx = random.choices(range(WEIGHT_INDEX_SIZE), weights=RNDWT, k=1)[0]
    return idx

def wi_std_3bit():
    p = random.random()
    if p >= 0.0 and p<0.0625:
        wi = 0
    elif p > 0.0625 and p<=0.125:
        wi = 1
    elif p > 0.125 and p<=0.25:
        wi = 2
    elif p > 0.25 and p<=0.5:
        wi = 3
    elif p > 0.50 and p<=0.75:
        wi = 4
    elif p > 0.75 and p<=0.825:
        wi = 5
    elif p > 0.825 and p<=0.9375:
        wi = 6
    elif p > 0.9375 and p<=1.00:
        wi = 7
    #
    return wi

def wi_8020_3bit():
    wmax = int( (WEIGHT_INDEX_SIZE - 1) / 2 )
    i = random.randint(0, wmax-1)
    if random.random() < 0.8:
        wi = wmax + i
    else:
        wi = i
    #
    return wi

def wi_8020():
    if random.random() < 0.05:
        wi = WEIGHT_INDEX_ZERO
        return wi
    #
    
    wmax = int( (WEIGHT_INDEX_SIZE - 1) / 2 )
    i = random.randint(0, wmax-1)
    if random.random() < 0.8:
        wi = wmax + i + 1
    else:
        wi = i
    #
    return wi

# [-1.0, -0.5, -0.25, -0.125, 0, 0.125, 0.25, 0.5, 1.0]
# 1, 2, 3, 4
def wi_std():
    if random.random() < 0.05:
        wi = WEIGHT_INDEX_ZERO
        return wi
    #
    p = random.random()
    if p < 0.05:
        wi = 0
    elif p >= 0.05 and p < 0.15:
        wi = 1
    elif p >= 0.15 and p < 0.30:
        wi = 2
    elif p >= 0.30 and p < 0.50:
        wi = 3
    elif p >= 0.50 and p < 0.70:
        wi = 5
    elif p >= 0.70 and p < 0.85:
        wi = 6
    elif p >= 0.85 and p < 0.95:
        wi = 7
    else:
        wi = 8
    #
    return wi

# [-1.0, -0.5, -0.25, -0.125, 0.0, 0.125, 0.25, 0.5, 1.0]

def wi_std2():
    p = random.random()
    if p <= 0.05:
        wi = 0
    elif p > 0.05 and p<=0.15:# 0.1
        wi = 1
    elif p > 0.15 and p<=0.3:# 0.15
        wi = 2
    elif p > 0.3 and p<=0.49:# 0.25
        wi = 3
    elif p > 0.49 and p<=0.51:
        wi = 4 # 0.0
    elif p > 0.51 and p<=0.7:
        wi = 5
    elif p > 0.7 and p<=0.85:
        wi = 6
    elif p > 0.85 and p<=0.95:
        wi = 7
    elif p > 0.95 and p<=1.0:
        wi = 8 # 1.0
    #
    return wi

class Weight:
    def __init__(self, li, ni, ii, wi, type=-1):
        self.li = li
        self.ni = ni
        self.ii = ii
        self.wi = wi
        self.wi_alt = wi
        self.mark = 0
        self.type = type
        self.momentum = 0
        #print("Weight::init()")
        
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
LAYER_TYPE_CONV    = 3
LAYER_TYPE_MAX     = 4
LAYER_TYPE_FCNN    = 5
LAYER_TYPE_FCNN2   = 6
#
class Layer(object):
    # i         : index of layers
    # type      : 0 = input, 1 = hidden, 2 = output
    # input : stimulus from a previous layer
    # num_input : number of inputs / outputs from a previous layer
    # node : neurons
    # num_node  : numbers of neurons in a layer
    def __init__(self, i, type, num_input, num_node, pre, gpu=None, qmode=0):
        self._pre = None
        self._next = None
        self._pre = pre
        if self._pre:
            self._pre._next = self
        #
        self._index = i
        self._type = type
        if gpu is not None:
            self._gpu = gpu
        else:
            self._gpu = None
        #
        self._id = -1
        self._num_input = num_input
        self._num_node = num_node
        self.qmode = qmode
        #
        self.backprop = False
    
    def set_backpropagation(self, sw, lr=0.01):
        self.backprop = sw
        self.learning_rate = lr
        
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
        
    #def getWeight(self, ni, ii):
    #    wi = self._weight_index_matrix[ni][ii]
    #    w = Weight(self._index, ni, ii, wi, self._type)
    #    return w
    
    def get_weight(self, ni, ii):
        wi = self._weight_index_matrix[ni][ii]
        w = Weight(self._index, ni, ii, wi, self._type)
        return w
    
    def get_weight_index(self, ni, ii):
        return self._weight_index_matrix[ni][ii]
    
    def set_weight_index(self, ni, ii, wi):
        #pre = self._weight_index_matrix[ni][ii]
        #self._weight_index_matrix[ni][ii] = wi
        try:
            if self._type==LAYER_TYPE_HIDDEN or self._type==LAYER_TYPE_OUTPUT:
            
                if self.qmode==0 or self.qmode==1:
                    self._weight_index_matrix[ni][ii] = np.uint8(wi)
                    self._weight_matrix[ni][ii] = WEIGHT_SET[wi]
                elif self.qmode==2:
                    self._weight_index_matrix[ni][ii] = np.uint8(wi)
                #
            elif self._type==LAYER_TYPE_CONV or self._type==LAYER_TYPE_FCNN:
                self._weight_matrix[ni][ii] = CNN_WEIGHT_SET[wi]
            elif self._type==LAYER_TYPE_FCNN2:
                self._weight_matrix[ni][ii] = CNN_WEIGHT_SET2[wi]
            #
        except Exception as inst:
            print(type(inst))    # the exception type
            print(inst.args)     # arguments stored in .args
            print(inst)
            
            print("set_weight_index()")
            print(" [%d] type=%d" % (self._index, self._type))
            #print(" (%d, %d) wi=%d, pre=%d" % (ni, ii, wi, pre))
            print(" (%d, %d) wi=%d" % (ni, ii, wi))
            print(" num_node:", self._num_node)
            print(" num_input:", self._num_input)
            #print(self._weight_matrix.shape)
            exit(0)
        #
    
    def init_weight_mode(self, ni, ii, mode, wi=0):
        #print("init_weight_mode(%d, %d, %d, %d)" % (ni, ii, mode, wi))
        wmin = 0
        wmax = WEIGHT_INDEX_SIZE
        if mode==0: # random index
            if self._type==LAYER_TYPE_HIDDEN or self._type==LAYER_TYPE_OUTPUT:
                wmax = WEIGHT_INDEX_SIZE
            elif self._type==LAYER_TYPE_CONV or self._type==LAYER_TYPE_FCNN:
                wmax = CNN_WEIGHT_INDEX_SIZE
            elif self._type==LAYER_TYPE_FCNN2:
                wmax = CNN_WEIGHT_INDEX_SIZE2
            #
            wi = random.randrange(wmax)
            self.set_weight_index(ni, ii, wi)
        elif mode==1: # random index in range
            if self._type==LAYER_TYPE_HIDDEN or self._type==LAYER_TYPE_OUTPUT:
                wmin = 1
                wmax = WEIGHT_INDEX_SIZE - 1
            elif self._type==LAYER_TYPE_CONV or self._type==LAYER_TYPE_FCNN:
                wmin = 0
                wmax = CNN_WEIGHT_INDEX_SIZE - 1
            elif self._type==LAYER_TYPE_FCNN2:
                wmin = 0
                wmax = CNN_WEIGHT_INDEX_SIZE2 - 1
            #
            wi = random.randrange(wmin, wmax, 1)
            self.set_weight_index(ni, ii, wi)
        elif mode==2: # fixed value
            self.set_weight_index(ni, ii, wi)
        elif mode==3: # 8:2
            wi = wi_8020()
            self.set_weight_index(ni, ii, wi)
        elif mode==4:
            wi = wi_std()
            self.set_weight_index(ni, ii, wi)
        elif mode==5:
            wi = wi_std2()
            self.set_weight_index(ni, ii, wi)
        elif mode==6: # random value in normal distribution
            v = 0.25 * np.random.standard_normal()
            #print(mode, v)
            if v==0.0:
                v = 0.0000001
            #
            self._weight_matrix[ni][ii] = v
        elif mode==7:
            #print("mode==7")
            #wi = wi_std_3bit()
            #wi = wi_std2()
            #wi = random.randrange(WEIGHT_INDEX_SIZE)
            wi = wi_std_11()
            self.set_weight_index(ni, ii, wi)
        #
                    
    def init_weight_with_mode(self, mode=0, value=0):
        #print(self._num_node, self._num_input)
        for ni in range(self._num_node):
            for ii in range(self._num_input):
                self.init_weight_mode(ni, ii, mode, value)
            #
        #

    #def init_weight(self, ni, ii, wi=-1):
    #    if wi<0:
    #        if self._type==LAYER_TYPE_HIDDEN or self._type==LAYER_TYPE_OUTPUT:
    #            wi = random.randrange(WEIGHT_INDEX_SIZE)
    #        elif self._type==LAYER_TYPE_CONV:
    #            wi = random.randrange(CNN_WEIGHT_INDEX_SIZE)
    #        #
    #    #
    #    self.set_weight_index(ni, ii, wi)
        
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
    
    def export_weight_value(self):
        return self._weight_matrix.tolist()
    
    def import_weight_index(self, wi_list):
        self._weight_index_matrix = np.array(wi_list, dtype=np.uint8).copy()
        for ni in range(self._num_node):
            for ii in range(self._num_input):
                wi = self._weight_index_matrix[ni][ii]
                self.set_weight_index(ni, ii, wi)
            #
        #

    def import_weight_value(self, wi_list):
        self._weight_matrix = np.array(wi_list, dtype=np.float16).copy()

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
        
        if self.qmode==0:
            self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.float32)
        elif self.qmode==1:
            self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.float16)
        elif self.qmode==2:
            self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.uint8)
        #
        if self._gpu:
            if self._gpu.type==0:
                self._gpu_output = self._gpu.dev_malloc(self._output_array)
            elif self._gpu.type==1:
                self._gpu_output = self._gpu.allocateArray(self._output_array)
            elif self._gpu.type==2:
                self._gpu_output = self._gpu.alloc_buf_from_array(self._output_array)
            #
        #
    
    def debug(self):
        print("InputLayer::debug()")
        print(self._output_array[0].shape)
        print(self._output_array[0])

    def propagate(self, array_in, debug=0):
        if debug:
            print("input")
            if self._gpu:
                if self._gpu.type==0: # OpenCL
                    self._gpu.copy(self._output_array, self._gpu_output)
                    print((self._output_array[0]))
                elif self._gpu.type==1: # DGX
                    pass
                elif self._gpu.type==2: # macOS Metal
                    print(self._output_array[0])
                    self._gpu.write_np_to_mbuf(self._output_array, self._gpu_output)
                #
            #
        #
        
    def bp(self, label_array, debug=0):
        if self._gpu.type==2:
            if debug:
                print("InputLayer::bp() macOS Metal")
            #
        else:
            print("InputLayer::bp()")
        #
        
    def set_weight_index(self, ni, ii, wi):
        pass
        
    def get_weight_index(self, ni, ii):
        return 0
        
    def export_weight_index(self):
        return None
    
    def export_weight_value(self):
        return None
        
    def count_weight(self):
        return 0

class HiddenLayer(Layer):
    def __init__(self, i, num_input, num_node, pre, gpu=None):
        print("HiddenLayer::__init__(%d, %d)" % (num_input, num_node))
        super(HiddenLayer, self).__init__(i, LAYER_TYPE_HIDDEN, num_input, num_node, pre, gpu)
        
        self._scale = 3 # scale only
        self.rate = 0.001
        if self._gpu:
            pass
        else:
            print("error : no gpu")
            return
        #
                
    def set_scale(self, mode):
        self._scale = mode
        
    def get_scale(self, mode):
        # 0 : none
        # 1 : layer normalize
        # 2 : batch normalize
        # 3 : max scale
        return self._scale

    def prepare(self, batch_size):
        print("HiddenLayer::prepare(%d), %d" % (batch_size, self.qmode))
        self._batch_size = batch_size
        
        if self._gpu.type==0:
            if self.qmode==0:
                self._weight_index_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.uint8)
                self._weight_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.float32)
                self._gpu_weight = self._gpu.dev_malloc(self._weight_matrix)
                self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.float32)
                self._gpu_output = self._gpu.dev_malloc(self._output_array)
            if self.qmode==1:
                #self._momentum = np.zeros( (self._num_node, self._num_input), dtype=np.int8)
                self._weight_index_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.uint8)
                self._weight_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.float16)
                self._gpu_weight = self._gpu.dev_malloc(self._weight_matrix)
                    
                self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.float16)
                self._gpu_output = self._gpu.dev_malloc(self._output_array)
            elif self.qmode==2:
                self._weight_index_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.uint8)
                self._gpu_weight = self._gpu.dev_malloc(self._weight_index_matrix)
                self.mac_array = np.zeros( (self._batch_size, self._num_node), dtype=np.float16)
                self._gpu_mac = self._gpu.dev_malloc(self.mac_array)
                self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.uint8)
                self._gpu_output = self._gpu.dev_malloc(self._output_array)
            #
        elif self._gpu.type==1:
            self._gpu_weight = self._gpu.allocateArray(self.grad)
            self._gpu_output = self._gpu.allocateArray(self._output_array)
        elif self._gpu.type==2: # macOS Metal
            #print(" macOS Metal")
            if self.qmode==0:
                self._weight_index_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.uint8)
                self._weight_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.float32)
                self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.float32)
                self._gpu_weight = self._gpu.alloc_buf_from_array(self._weight_matrix)
                self._gpu_output = self._gpu.alloc_buf_from_array(self._output_array)
            elif self.qmode==1:
                self._weight_index_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.uint8)
                self._weight_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.float16)
                self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.float16)
            
                self._gpu_weight = self._gpu.alloc_buf_from_array(self._weight_matrix)
                self._gpu_output = self._gpu.alloc_buf_from_array(self._output_array)
            elif self.qmode==2:
                pass
            #
        #

    def update_weight(self):
        if self._gpu:
            pass
        else:
            print("HiddenLayer::update_weight() = error, no gpu")
            return
        #
        if self._gpu.type==0:
            if self.qmode==0:
                self._gpu.copy(self._gpu_weight, self._weight_matrix)
            elif self.qmode==1:
                self._gpu.copy(self._gpu_weight, self._weight_index_matrix)
            #
        elif self._gpu.type==1:
            self._gpu_weight = self._gpu.allocateArray(self._weight_matrix)
        elif self._gpu.type==2:
            self._gpu.write_np_to_mbuf(self._weight_matrix, self._gpu_weight)
        #

    def propagate(self, array_in, debug=0):
        if debug:
            print(self._index, "HiddenLayer::propagate()", debug)
            #print(self._index, "@@@@@@@@ output")
            #self._gpu.copy(self._output_array, self._gpu_output)
            #print((self._output_array[0]))
        #
        
        if self._gpu:
            pass
        else:
            print("HiddenLayer::propagate() = error, no gpu")
            return
        #
            
        # activation mode
        #   0 : none
        #   1 : normal / relu
        #   2 : 0.000001
        #   3 : y/20
        a_mode = 1
        
        if self._gpu.type==0: # OpenCL
            if self.qmode==0:
                pass
            elif self.qmode==1:
                self._gpu.macRelu(array_in, self._gpu_weight, self._gpu_output, self._batch_size, self._num_node, self._num_input, a_mode)
                if self._scale==0:
                    pass
                elif self._scale==1:
                    self._gpu.normalize_layer(self._gpu_output, self._batch_size, self._num_node)
                elif self._scale==2:
                    self._gpu.normalize_batch(self._gpu_output, self._batch_size, self._num_input, self._num_node)
                elif self._scale==3: # quantization test of Relu()
                    self._gpu.scale_layer(self._batch_size, self._gpu_output, self._num_node, 1.0)
                elif self._scale==4:
                    self._gpu.normalize_layer(self._gpu_output, self._batch_size, self._num_node)
                    self._gpu.scale_layer(self._batch_size, self._gpu_output, self._num_node)
                #
                
                if debug:
                    print("scale", self._scale)
                    print(self._index, "hidden, input")
                    tarray = np.zeros((self._batch_size, self._num_input), dtype=np.float16)
                    self._gpu.copy(tarray, array_in)
                    print((tarray[0]))
                    
                    self._gpu.copy(self._output_array, self._gpu_output)
                    print(self._index, "hidden, output")
                    print((self._output_array[0]))
                    print("hidden, weight", self._weight_matrix.shape)
                    print((self._weight_matrix[0]))
                #
            elif self.qmode==2:
                self._gpu.macReluQ(array_in, self._gpu_weight, self._gpu_mac, self._batch_size, self._num_node, self._num_input, a_mode)
                self._gpu.q_hidden_output(self._gpu_mac, self._gpu_output, self._num_node, self._batch_size)
                # quantize output
                if debug:
                    #print(self._index, "hidden, weight")
                    #print(self._weight_index_matrix)
                
                    #tarray = np.zeros((self._batch_size, self._num_input), dtype=np.uint8)
                    #self._gpu.copy(tarray, array_in)
                    #print(self._index, "hidden, input")
                    #print((tarray[0]))
        
                    self._gpu.copy(self._output_array, self._gpu_output)
                    print(self._index, "hidden, output")
                    print(self._output_array[0])
                    
                    #self._gpu.copy(self.mac_array, self._gpu_mac)
                    #print(self._index, "hidden, output")
                    #print(self.mac_array[0])
                #
            #
        elif self._gpu.type==1: # DGX
            self._gpu.macRelu3(array_in, self._gpu_weight, self._gpu_output, self._batch_size, self._num_node, self._num_input, a_mode)
            if self._scale==0:
                pass
            elif self._scale==3:
                self._gpu.layerScale(self._gpu_output, self._batch_size, self._num_node, 1.0)
            #
            if debug:
                print(self._index, "hidden, input")
                darray = cp.asnumpy(array_in)
                print(darray[0])
                    
                print(self._index, "hidden")
                darray = cp.asnumpy(self._gpu_output)
                print(darray[0])
            #
        elif self._gpu.type==2: # macOS Metal
            if debug:
                print("HiddenLayer::propagate() macOS Metal", self.qmode, debug)
            #
            
            if self.qmode==0:
                self._gpu.calc_mac_relu(self._batch_size, array_in, self._gpu_weight, self._gpu_output, self._num_node, self._num_input, a_mode)
                self._gpu.scale_layer(self._batch_size, self._num_node, 1.0, self._gpu_output)
                if self.backprop:
                    self._output_array = np.frombuffer(self._gpu_output.contents().as_buffer(self._gpu_output.length()), dtype=np.float32)
                    self._output_array = self._output_array.view(np.float32).reshape(self._batch_size, self._num_node)
                #
                if debug:
                    out = np.frombuffer(self._gpu_output.contents().as_buffer(self._gpu_output.length()), dtype=np.float32)
                    out = out.view(np.float32).reshape(self._batch_size, self._num_node)
                    print(out.shape)
                    print(out[:10])
                #
            elif self.qmode==1:
                self._gpu.calc_mac_relu(self._batch_size, array_in, self._gpu_weight, self._gpu_output, self._num_node, self._num_input, a_mode)
                self._gpu.scale_layer(self._batch_size, self._num_node, 1.0, self._gpu_output)
            
                if self.backprop:
                    self._output_array = np.frombuffer(self._gpu_output.contents().as_buffer(self._gpu_output.length()), dtype=np.float16)
                    self._output_array = self._output_array.view(np.float16).reshape(self._batch_size, self._num_node)
                #
            
                if debug:
                    out = np.frombuffer(self._gpu_output.contents().as_buffer(self._gpu_output.length()), dtype=np.float16)
                    print(out.shape)
                    print(out[:10])
                #
            elif self.qmode==2:
                pass
            #
        #
        
    def bp(self, label_array, debug=0):
        if debug:
            print("HiddenLayer::bp() macOS Metal")
        #
            
        if self._gpu.type==2:
            # delta
            self.delta = self._next.delta @ self._next._weight_matrix

            # slope
            self.dW = (self._pre._output_array.T @ self.delta) / self._batch_size

            # optimaize
            self.dW = self.dW.T
            N = self.dW.shape[0]
            M = self.dW.shape[1]
            for n in range(N):
                for m in range(M):
                    w = self._weight_matrix[n][m]
                    self._weight_matrix[n][m] = w - self.dW[n][m] * np.float32(self.learning_rate)
                    # derivertive of relu
                    #if self.dW[n][m]<=0:
                    #    self.dW[n][m] = 0.0
                    #
                #
            #
            self.dW = self.dW.T
        #
        
class OutputLayer(Layer):
    def __init__(self, i, num_input, num_node, pre, gpu=None, smax=False):
        print("OutputLayer::__init__()")
        super(OutputLayer, self).__init__(i, LAYER_TYPE_OUTPUT, num_input, num_node, pre, gpu)
                
        if self._gpu:
            pass
        else:
            print("error : no gpu")
        #
        self.smax = smax
        self.softmax_scale = 1.0
        self.rate = 0.01
        
    def set_softmax_scale(self, scale):
        self.softmax_scale = scale
        
    def prepare(self, batch_size):
        print("OutputLayer::prepare(%d), %d" % (batch_size, self.qmode))
        self._batch_size = batch_size
        
        if self._gpu.type==0:
            if self.qmode==0:
                #self._momentum = np.zeros( (self._num_node, self._num_input), dtype=np.int8)
                self._weight_index_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.uint8)
                self._weight_matrix = np.zeros((self._num_node, self._num_input), dtype=np.float32)
                self._gpu_weight = self._gpu.dev_malloc(self._weight_matrix)
                
                self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.float32)
                self._gpu_output = self._gpu.dev_malloc(self._output_array)
                                
                self._softmax_array = np.zeros((self._batch_size, self._num_node), dtype=np.float32)
                self._gpu_softmax = self._gpu.dev_malloc(self._softmax_array)
            elif self.qmode==1:
                self._momentum = np.zeros( (self._num_node, self._num_input), dtype=np.int8)
                self._weight_index_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.uint8)
                self._weight_matrix = np.zeros((self._num_node, self._num_input), dtype=np.float16)
                self._gpu_weight = self._gpu.dev_malloc(self._weight_matrix)
                
                self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.float16)
                self._gpu_output = self._gpu.dev_malloc(self._output_array)
                                
                self._softmax_array = np.zeros((self._batch_size, self._num_node), dtype=np.float16)
                self._gpu_softmax = self._gpu.dev_malloc(self._softmax_array)
            elif self.qmode==2:
                self._weight_index_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.uint8)
                self._gpu_weight = self._gpu.dev_malloc(self._weight_index_matrix)
                
                self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.float16)
                self._gpu_output = self._gpu.dev_malloc(self._output_array)
                
                self._softmax_array = np.zeros((self._batch_size, self._num_node), dtype=np.float16)
                self._gpu_softmax = self._gpu.dev_malloc(self._softmax_array)
            #
        elif self._gpu.type==1:
            print("output : nvidia")
            self._weight_matrix = np.zeros((self._num_node, self._num_input), dtype=np.float16)
            self._gpu_weight = self._gpu.allocateArray(self._weight_matrix)
            
            self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.float16)
            self._gpu_output = self._gpu.allocateArray(self._output_array)
            self._softmax_array = np.zeros((self._batch_size, self._num_node), dtype=np.float16)
            self._gpu_softmax = self._gpu.allocateArray(self._softmax_array)
        elif self._gpu.type==2:
            #print("OutputLayer::prepare() macOS Metal")
            if self.qmode==0:
                self._weight_index_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.uint8)
                self._weight_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.float32)
                self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.float32)
                self._softmax_array = np.zeros((self._batch_size, self._num_node), dtype=np.float32)
        
                self._gpu_weight = self._gpu.alloc_buf_from_array(self._weight_matrix)
                self._gpu_output = self._gpu.alloc_buf_from_array(self._output_array)
                self._gpu_softmax = self._gpu.alloc_buf_from_array(self._softmax_array)
            elif self.qmode==1:
                self._weight_index_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.uint8)
                self._weight_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.float16)
                self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.float16)
                self._softmax_array = np.zeros((self._batch_size, self._num_node), dtype=np.float16)
        
                self._gpu_weight = self._gpu.alloc_buf_from_array(self._weight_matrix)
                self._gpu_output = self._gpu.alloc_buf_from_array(self._output_array)
                self._gpu_softmax = self._gpu.alloc_buf_from_array(self._softmax_array)
            elif self.qmode==2:
                pass
            #
        #
        
    def update_weight(self):
        if self._gpu.type==0:
            if self.qmode==0 or self.qmode==1:
                self._gpu.copy(self._gpu_weight, self._weight_matrix)
            elif self.qmode==2:
                self._gpu.copy(self._gpu_weight, self._weight_index_matrix)
            #
        elif self._gpu.type==1:
            self._gpu_weight = self._gpu.allocateArray(self._weight_matrix)
        elif self._gpu.type==2: # macOS Metal
            self._gpu.write_np_to_mbuf(self._weight_matrix, self._gpu_weight)
        #
        
    def propagate(self, array_in, debug=0):
        if debug:
            print(self._index, "OutputLayer::propagate()", debug)
        #
        if self._gpu:
            pass
        else:
            print("OutputLayer::propagate() = error, no gpu")
        #
        
        if self._gpu.type==0: # OpenCL
            if self.qmode==0:
                pass
            elif self.qmode==1:
                self._gpu.macRelu(array_in, self._gpu_weight, self._gpu_output,
                              self._batch_size, self._num_node, self._num_input, 0)
                #if debug:
                #    self._gpu.copy(self._output_array, self._gpu_output)
                #
                self._gpu.softmax(self._gpu_output, self._gpu_softmax, self._num_node, self._batch_size, self.softmax_scale)
                
                if debug:
                    print("softmax", self._softmax_array[0].sum())
                    print(self._softmax_array[0])
                #
            elif self.qmode==2:
                a_mode = 0
                self._gpu.macReluQ(array_in, self._gpu_weight, self._gpu_output, self._batch_size, self._num_node, self._num_input, a_mode)
                if debug:
                    self._gpu.copy(self._output_array, self._gpu_output)
                    print(self._index, "output, mac")
                    print(self._output_array[0])
                #
                self._gpu.softmax(self._gpu_output, self._gpu_softmax, self._num_node, self._batch_size, self.softmax_scale)
                
                if debug:
                    self._gpu.copy(self._softmax_array, self._gpu_softmax)
                    print("output, softmax", self._softmax_array[0].sum())
                    print(self._softmax_array[0])
                #
            #
        elif self._gpu.type==1: # DGX
            self._gpu.macRelu3(array_in, self._gpu_weight, self._gpu_output, self._batch_size, self._num_node, self._num_input, 0)
            if debug:
                print("output::mac", self._index)
                darray = cp.asnumpy(self._gpu_output)
                print(darray[0])
            #
            self._gpu.layerScale(self._gpu_output, self._batch_size, self._num_node, 4.0)
            if debug:
                print("output::scale", self._index)
                darray = cp.asnumpy(self._gpu_output)
                print(darray[0])
            #
            self._gpu.softmax(self._gpu_output, self._gpu_softmax, self._batch_size, self._num_node, self.softmax_scale)
            if debug:
                print("output::softmax", self._index)
                darray = cp.asnumpy(self._gpu_softmax)
                print(darray[0])
                print("sum:", darray[0].sum())
            #
        elif self._gpu.type==2:
            if debug:
                print("OutputLayer::propagate() macOS Metal", self.smax)
            #
            
            if self.qmode==0:
                self._gpu.calc_mac_relu(self._batch_size, array_in, self._gpu_weight, self._gpu_output, self._num_node, self._num_input, 0)
                if debug:
                    out = np.frombuffer(self._gpu_output.contents().as_buffer(self._gpu_output.length()), dtype=np.float32)
                    out = out.view(np.float32).reshape(self._batch_size, self._num_node)
                    print(out.shape)
                    print(out[0])
                #
                if self.smax:
                    self._gpu.softmax(self._batch_size, self._num_node, self.softmax_scale, self._gpu_output, self._gpu_softmax)
                    
                    if debug:
                        out = np.frombuffer(self._gpu_softmax.contents().as_buffer(self._gpu_softmax.length()), dtype=np.float32)
                        out = out.view(np.float32).reshape(self._batch_size, self._num_node)
                        print(out.shape)
                        print(out[0], out[0].sum())
                    #
                #
                
                if self.backprop:
                    self._output_array = np.frombuffer(self._gpu_output.contents().as_buffer(self._gpu_output.length()), dtype=np.float32)
                    self._output_array = self._output_array.view(np.float32).reshape(self._batch_size, self._num_node)
                    if self.smax:
                        self._softmax_array = np.frombuffer(self._gpu_softmax.contents().as_buffer(self._gpu_softmax.length()), dtype=np.float32)
                        self._softmax_array = self._softmax_array.view(np.float32).reshape(self._batch_size, self._num_node)
                    #
                #
                    

            elif self.qmode==1:
                self._gpu.calc_mac_relu(self._batch_size, array_in, self._gpu_weight, self._gpu_output, self._num_node, self._num_input, 0)
                
                if self.backprop:
                    self._output_array = np.frombuffer(self._gpu_output.contents().as_buffer(self._gpu_output.length()), dtype=np.float16)
                    self._output_array = self._output_array.view(np.float16).reshape(self._batch_size, self._num_node)
                    self._softmax_array = np.frombuffer(self._gpu_softmax.contents().as_buffer(self._gpu_softmax.length()), dtype=np.float16)
                    self._softmax_array = self._softmax_array.view(np.float16).reshape(self._batch_size, self._num_node)
                    self._gpu.softmax(self._batch_size, self._num_node, self.softmax_scale, self._gpu_output, self._gpu_softmax)
                    #debug=1
                    if debug:
                        N = self._softmax_array.shape[0]
                        M = self._softmax_array.shape[1]
                        for n in range(N):
                            for m in range(M):
                                if np.isnan(self._softmax_array[n][m]):
                                    print(n, m, self._output_array[n][m], self._softmax_array[n][m])
                                #
                            #
                        #
                    #
                    #debug = 0
                else:
                    self._gpu.softmax(self._batch_size, self._num_node, self.softmax_scale, self._gpu_output, self._gpu_softmax)
                #
                        
                if debug:
                    out = np.frombuffer(self._gpu_output.contents().as_buffer(self._gpu_output.length()), dtype=np.float16)
                    out = out.view(np.float16).reshape(self._batch_size, self._num_node)
                    print(out.shape)
                #
            elif self.qmode==2:
                pass
            #
        #
    
    def bp(self, label_array, debug=0):
        if debug:
            print("OutputLayer::bp() macOS Metal")
        #
        
        if self._gpu.type==2:
            # differencial
            self.delta = (self._softmax_array - label_array) # need no batch avg
            
            
            
            
            
            #ddd = self._softmax_array - label_array
            print("self.delta.shape:", self.delta.shape, type(self.delta[0][0])) # (100, 10)
            
            
            #print("self._pre._output_array.shape:", self._pre._output_array.shape) # (100, 256)
            
            #I = self.delta.shape[0] # 100
            #J = self.delta.shape[1] # 10
            #for i in range(I):
            #    for j in range(J):
            #        if np.isnan(self.delta[i][j]):
            #            #print("NaN", i, j, self.delta[i][j], self._softmax_array[i][j], label_array[i][j])
            #            print("NaN", i, self._softmax_array[i])
            #        #
            #    #
            #
            

            # slope for weights
            self.dW = (self._pre._output_array.T @ self.delta) / self._batch_size
            #N = self.dW.shape[0]
            #M = self.dW.shape[1]
            #for n in range(N): # 100
            #    for m in range(M): # 256
            #        self._pre._output_array[][]
            
            #print("self.dW.shape:", self.dW.shape) # (256, 10)
            
            # change weights and derivertive of relu
            self.dW = self.dW.T # transpose
            #print("self.dW.T.shape:", self.dW.shape) # (10, 256)
            #print("self._weight_matrix.shape:", self._weight_matrix.shape) # (10, 256)
            N = self.dW.shape[0]
            M = self.dW.shape[1]
            #cnt = 0
            for n in range(N):
                for m in range(M):
                    #if np.isnan(self.dW[n][m]):
                    #    print("NaN", n, m, self.delta[m][n], self._pre._output_array[m][n], self.delta[m][n]*self._pre._output_array[m][n]/self._batch_size)
                    #
                    w = self._weight_matrix[n][m]
                    self._weight_matrix[n][m] = w - self.dW[n][m] * np.float32(self.learning_rate)
                    #if self.dW[n][m]<0:
                    #    self.dW[n][m] = 0.0
                    #
                #
            #

            #self._weight_index_matrix[n][m] = idx - 1
            #self._weight_matrix[n][m] = WEIGHT_SET[self._weight_index_matrix[n][m]]
            
            self.dW = self.dW.T # transpose
            if debug:
                print(self.delta[0])
                print("dW", self.dW.shape)
                print(self.dW[0])
            #
            
            #dX = self.delta @ self._weight_matrix
        else:
            print("OutputLayer::bp()")
        #
        
class RegressionOutputLayer(Layer):
    def __init__(self, i, num_input, num_node, pre, gpu=None):
        print("RegressionOutputLayer::__init__()")
        super(RegressionOutputLayer, self).__init__(i, LAYER_TYPE_OUTPUT, num_input, num_node, pre, gpu)
        if gpu:
            pass
        else:
            print("error, no gou")
            return
        #
        
        self._weight_index_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.int32)
        self._weight_matrix = np.zeros( (self._num_node, self._num_input), dtype=np.float16)
        #
        if self._gpu.type==0:
            self._gpu_weight = self._gpu.dev_malloc(self._weight_matrix)
        elif self._gpu.type==1:
            print("error")
        #
        
    def prepare(self, batch_size):
        if gpu:
            pass
        else:
            print("RegressionOutputLayer::prepare(), error, no gou")
            return
        #
        self._batch_size = batch_size
        self._product_matrix = np.zeros( (self._batch_size, self._num_node, self._num_input), dtype=np.float16)
        self._output_array = np.zeros((self._batch_size, self._num_node), dtype=np.float16)
        #
        if self._gpu.type==0:
            self._gpu_product = self._gpu.dev_malloc(self._product_matrix)
            self._gpu_output = self._gpu.dev_malloc(self._output_array)
        elif self._gpu.type==1:
            pass
        #
    
    def update_weight(self):
        if gpu:
            pass
        else:
            print("RegressionOutputLayer::update_weight(), error, no gou")
            return
        #

        if self._gpu.type==0:
            self._gpu.copy(self._gpu_weight, self._weight_matrix)
        elif self._gpu.type==1:
            pass
        #

    def propagate(self, array_in, debug=0):
        if self._gpu:
            pass
        else:
            print("RegressionOutputLayer::propagate() = error, no gpu")
        #
        stride_1 = self._num_node * self._num_input
        stride_2 = self._num_input
        # multiple
        if self._gpu.type==1:
            self._gpu.multiple_x_by_w_batch(array_in, self._gpu_weight, self._gpu_product,                                   self._batch_size, stride_1, stride_2,
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
            print("RegressionOutputLayer::propagate() for nvidia not yet impremented")
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
        self.cache = 0
    
    def reset(self):
        self.cache = 0
        
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
        self._output_array = np.zeros((self._batch_size, self._ch, self._num_node), dtype=np.float16)
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
        if self.cache:
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
        self.cache = 1

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
        super(Conv_4_Layer, self).__init__(i, LAYER_TYPE_CONV, num_input, num_node, pre, gpu)
        #
        # mems for weights
        self._weight_index_matrix = np.zeros( (self._filter, self._num_of_w), dtype=np.int32)
        self._weight_matrix = np.zeros( (self._filter, self._num_of_w), dtype=np.float16)
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
        self._padded_array = np.zeros((self._batch_size, (self._w+2)*(self._h+2)*self._ch), dtype=np.float16)
        self._sum_array = np.zeros((self._batch_size), dtype=np.float16)
        self._dsum_array = np.zeros((self._batch_size), dtype=np.float16)
        #self._max_array = np.zeros((self._batch_size), dtype=np.float16)
        # output
        self._output_array = np.zeros((self._batch_size, self._filter, self._w*self._h), dtype=np.float16)
        if self._gpu:
            if self._gpu.type==0:
                self._gpu_padded = self._gpu.dev_malloc(self._padded_array)
                self._gpu_output = self._gpu.dev_malloc(self._output_array)
                self._gpu_sum = self._gpu.dev_malloc(self._sum_array)
                #self._gpu_max = self._gpu.dev_malloc(self._max_array)
            elif self._gpu.type==1:
                self._gpu_padded = self._gpu.allocateArray(self._padded_array)
                self._gpu_output = self._gpu.allocateArray(self._output_array)
                self._gpu_sum = self._gpu.allocateArray(self._sum_array)
                self._gpu_dsum = self._gpu.allocateArray(self._dsum_array)
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
                #self.save_padded(0, 1, self._padded_array[0])
                #self.save_padded(0, 2, self._padded_array[0])
            #
            self._gpu.conv_4_roll_batch(self._gpu_padded, self._gpu_weight, self._gpu_output,
                                        self._w, self._h, self._ch, self._filter,
                                        self._batch_size, 0)
            if debug:
                print(self._index, "conv ret")
                self._gpu.copy(self._output_array, self._gpu_output)
                print((self._output_array[0][0]))
            #
            
            # normalize
            size = self._w * self._h * self._filter
            #self._gpu.get_sum(self._batch_size, self._gpu_output, self._gpu_sum, size)
            #self._gpu.copy(self._sum_array, self._gpu_sum)
            #mean = self._sum_array.sum() / float(self._batch_size*size)
            #self._gpu.get_dsum(self._batch_size, self._gpu_output, self._gpu_sum, size, mean)
            #self._gpu.copy(self._sum_array, self._gpu_sum)
            #div2 = self._sum_array.sum() / float(self._batch_size*size)
            #div = np.sqrt(div2) +  0.0000001;
            #self._gpu.get_std(self._batch_size, self._gpu_output, size, mean, div)
            #
            
            # relu
            self._gpu.relu(self._gpu_output, self._batch_size, self._filter, size, a_mode)
            
            # scale
            #self._gpu.scale_layer(self._batch_size, self._gpu_output, size)
            if debug:
                print(self._index, "conv, scale")
                self._gpu.copy(self._output_array, self._gpu_output)
                print((self._output_array[0][0]))
                #
                #self.save_output()
                #self.save_filter_out(0, 0, self._output_array[0][0])
                self.save_debug("ocl.txt", self._w, self._h, self._output_array[0][0])
            #
        elif self._gpu.type==1: # GDX
            self._gpu.padding(array_in, self._gpu_padded, self._w, self._h, self._ch, self._batch_size)
            if debug:
                print(self._index, "conv", "padded", )
                darray = cp.asnumpy(self._gpu_padded)
                print((darray.shape, type(darray[0])))
                self.save_padded(0, 0, darray[0])
            #
            self._gpu.convolusion(self._gpu_padded, self._gpu_weight, self._gpu_output, self._w, self._h, self._ch, self._filter, self._batch_size)
            if debug:
                print(self._index, "conv ret")
                darray = cp.asnumpy(self._gpu_output)
                print((darray.shape, type(darray[0])))
                print(darray[0][0])
            #
            
            # normalize
            size = self._w * self._h * self._filter
            self._gpu.get_sum(self._batch_size, self._gpu_output, self._gpu_sum, size)
            mean = float( self._gpu_sum.sum() / float(self._batch_size * size) )
            self._gpu.get_dsum(self._batch_size, self._gpu_output, self._gpu_sum, size, mean)
            div2 = float(self._gpu_sum.sum() / float(self._batch_size * size))
            div = float(np.sqrt(div2) + 0.0000001);
            self._gpu.get_std(self._batch_size, self._gpu_output, size, mean, div)
            
            #relu
            self._gpu.relu(self._batch_size, self._filter, self._gpu_output, size, a_mode)

            # scale
            #self._gpu.layerScale(self._gpu_output, self._batch_size, size)
            
            if debug:
                print(self._index, "conv, scale")
                darray = cp.asnumpy(self._gpu_output)
                print(darray[0][0])
                #
                self.save_filter_out(0, 0, darray[0][0])
                self.save_debug("dgx.txt", self._w, self._h, darray[0][0])
            #
            #self._gpu.filterScale(self._batch_size, self._filter, self._gpu_output, size, self._w * self._h)
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
    
    def save_array_to_png(self, data_array):
        for bi in range(self._batch_size):
            for fi in range(self._filter):
                data = data_array[bi][fi]
                self.save_filter_out(bi, fi, data)
            #
        #
        
    def save_debug(self, name, w, h, darray):
        fname = "./debug/" + name
        print("### fname:",fname)
        size = w * h
        with open(fname, 'wt') as f:
            for i in range(size):
                f.write("%f\n" %(darray[i]))
            #
        #
        
class Conv_5_Layer(Layer):
    def __init__(self, i, w, h, ch, filter, size, stride, pre, gpu=None):
        print("Convolution Layer ver.5 ::__init__()")
        
        self._w = w
        self._h = h
        self._ch = ch # number of inputs
        self._filter = filter # node / # number of outputs
        self._filter_len = size
        self._filter_size = size * size * ch
        self._stride = stride
        self._out_w = self._w - (self._filter_len - self._stride)
        self._out_h = self._h - (self._filter_len - self._stride)
        #
        num_node = self._filter
        num_input = self._filter_size
        super(Conv_5_Layer, self).__init__(i, LAYER_TYPE_CONV, num_input, num_node, pre, gpu)
        
        # mems for weights
        self._weight_index_matrix = np.zeros( (self._filter, self._filter_size), dtype=np.int32)
        print(self._weight_index_matrix.shape)
        
        self._weight_matrix = np.zeros( (self._filter, self._filter_size), dtype=np.float16)
        if self._gpu:
            if self._gpu.type==0:
                self._gpu_weight = self._gpu.dev_malloc(self._weight_matrix)
            elif self._gpu.type==1:
                self._gpu_weight = self._gpu.allocateArray(self._weight_matrix)
            #
        else:
            print("error")
        #
        
    def prepare(self, batch_size):
        print(("Conv_5_Layer::prepare(%d)" %(batch_size)))
        
        self._batch_size = batch_size
        # intermidiate
        self._sum_array = np.zeros((self._batch_size), dtype=np.float16)
        self._dsum_array = np.zeros((self._batch_size), dtype=np.float16)
        # output
        self._output_array = np.zeros((self._batch_size, self._filter, self._out_w*self._out_h), dtype=np.float16)
        
        if self._gpu:
            if self._gpu.type==0:
                self._gpu_output = self._gpu.dev_malloc(self._output_array)
                self._gpu_sum = self._gpu.dev_malloc(self._sum_array)
            elif self._gpu.type==1:
                self._gpu_output = self._gpu.allocateArray(self._output_array)
                self._gpu_sum = self._gpu.allocateArray(self._sum_array)
                self._gpu_dsum = self._gpu.allocateArray(self._dsum_array)
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
        
    def propagate(self, array_in, debug=0):
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
            self._gpu.conv_5_roll_batch(self._batch_size, self._out_w, self._out_h,
                                        array_in, self._gpu_weight, self._gpu_output,
                                        self._w, self._h,
                                        self._ch, self._filter,
                                        self._filter_len, self._stride)
            if debug:
                print(self._index, "conv ret")
                self._gpu.copy(self._output_array, self._gpu_output)
                print((self._output_array[0][0]))
            #
            size = self._out_w * self._out_h * self._filter
            # normalize
            self._gpu.get_sum(self._batch_size, self._gpu_output, self._gpu_sum, size)
            self._gpu.copy(self._sum_array, self._gpu_sum)
            mean = self._sum_array.sum() / float(self._batch_size*size)
            self._gpu.get_dsum(self._batch_size, self._gpu_output, self._gpu_sum, size, mean)
            self._gpu.copy(self._sum_array, self._gpu_sum)
            div2 = self._sum_array.sum() / float(self._batch_size*size)
            div = np.sqrt(div2) +  0.0000001;
            self._gpu.get_std(self._batch_size, self._gpu_output, size, mean, div)
            
            # relu
            self._gpu.relu(self._gpu_output, self._batch_size, self._filter, size, a_mode)
            if debug:
                print(self._index, "conv, scale")
                self._gpu.copy(self._output_array, self._gpu_output)
                print((self._output_array[0][0]))
                #
                for i in range(self._filter):
                    name = "debug_%d" % (i)
                    self.save_png(name, self._output_array[0][i])
                #
            #
        elif self._gpu.type==1: # GDX
            pass
        #
        
    def save_png(self, name, data_array):
        #size = self._w * self._h
        max = np.max(data_array)
        min = np.min(data_array)
        print(("max=%f, min=%f" % (max, min)))
        
        img = Image.new("L", (self._out_w, self._out_h), 0)
        pix = img.load()
        for y in range(self._out_h):
            for x in range(self._out_w):
                v = data_array[self._out_w*y + x]
                if max>0.0:
                    v1 = int(v*255/max)
                    pix[x,y] = v1
                else:
                    pix[x,y] = 0
                #
            #
        #
        spath = "./debug/%s.png" % (name)
        print(spath)
        img.save(spath)

class FCNN_Layer(Layer):
    def __init__(self, i, w, h, ch, filter, pre, gpu=None, type=LAYER_TYPE_FCNN, padding=0):
        print("Fixed CNN Layer::__init__()")
        self.padding = padding
        self.cache = 0
        self.ksize = 3 # kernel size
        stride = 1
        self._w = w
        self._h = h
        if self.padding==1:
            self._w = self._w + 2
            self._h = self._h + 2
        #
        self._ch = ch # number of input channels
        self._filter = filter # node / number of output channels
        self._filter_size = self.ksize * self.ksize * ch
        self._stride = stride
        #self._out_w = self._w - (self.ksize - self._stride)
        #self._out_h = self._h - (self.ksize - self._stride)
        self._out_w = w - (self.ksize - self._stride)
        self._out_h = h - (self.ksize - self._stride)
        #
        num_node = self._filter
        num_input = self._filter_size
        super(FCNN_Layer, self).__init__(i, type, num_input, num_node, pre, gpu)
        
        # mems for weights
        self._weight_index_matrix = np.zeros( (self._filter, self._filter_size), dtype=np.int32)
        self._weight_matrix = np.zeros( (self._filter, self._filter_size), dtype=np.float16)
        if self._gpu:
            if self._gpu.type==0:
                self._gpu_weight = self._gpu.dev_malloc(self._weight_matrix)
            elif self._gpu.type==1:
                self._gpu_weight = self._gpu.allocateArray(self._weight_matrix)
            #
        else:
            print("error")
        #
        
    def reset(self):
        self.cache = 0
        
    def prepare(self, batch_size):
        print(("FCNN_Layer::prepare(%d)" %(batch_size)))
        
        self._batch_size = batch_size
        # intermidiate
        self._sum_array = np.zeros((self._batch_size), dtype=np.float16)
        self._dsum_array = np.zeros((self._batch_size), dtype=np.float16)
        # output
        self._output_array = np.zeros((self._batch_size, self._filter, self._out_w*self._out_h), dtype=np.float16)
        
        if self.padding==1:
            isize = self._w*self._h * self._ch
            self._padded_array = np.zeros((self._batch_size, isize), dtype=np.float16)
        #
        
        if self._gpu:
            if self._gpu.type==0:
                self._gpu_output = self._gpu.dev_malloc(self._output_array)
                self._gpu_sum = self._gpu.dev_malloc(self._sum_array)
                if self.padding==1:
                    self._gpu_padded = self._gpu.dev_malloc(self._padded_array)
                #
            elif self._gpu.type==1:
                self._gpu_output = self._gpu.allocateArray(self._output_array)
                self._gpu_sum = self._gpu.allocateArray(self._sum_array)
                self._gpu_dsum = self._gpu.allocateArray(self._dsum_array)
                if self.padding==1:
                    self._gpu_padded = self._gpu.allocateArray(self._padded_array)
                #
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
        
    def set_filter(self, index, farray, size):
        for i in range(size):
            self._filter[index][i] = farray[i]
        #
        if self._gpu:
            self._gpu.copy(self._gpu_weight, self._weight_matrix)
        #
        
    def propagate(self, array_in, debug=0):
        if self._gpu:
            pass
        else:
            print("FCNN_Layer::propagate() = error, no gpu")
            return
        #
        
        if self.cache:
            return
        #
        
        a_mode = 1
        if self._gpu.type==0: # OpenCL
            if self.padding==0:
                self._gpu.conv_5_roll_batch(self._batch_size,
                                            self._out_w,
                                            self._out_h,
                                            array_in,
                                            self._gpu_weight,
                                            self._gpu_output,
                                            self._w,
                                            self._h,
                                            self._ch,
                                            self._filter,
                                            self.ksize,
                                            self._stride)
            else:
                self._gpu.conv_4_pad_batch(array_in,
                                            self._gpu_padded,
                                            self._w-2, self._h-2,
                                            self._ch,
                                            self._batch_size)
                self._gpu.conv_5_roll_batch(self._batch_size,
                                            self._out_w,
                                            self._out_h,
                                            self._gpu_padded,
                                            self._gpu_weight,
                                            self._gpu_output,
                                            self._w,
                                            self._h,
                                            self._ch,
                                            self._filter,
                                            self.ksize,
                                            self._stride)
            #
            if debug:
                #if self._index==1:
                    #self._pre.debug()
                    #temp = np.zeros(3072, dtype=np.float16)
                    #self._gpu.copy(temp, array_in)
                    #print(temp)
                #    print(self._weight_index_matrix)
                #    print(self._weight_matrix)
                #
                print(self._index, "FCNN")
                self._gpu.copy(self._output_array, self._gpu_output)
                print(self._output_array.shape)
                print(self._output_array[0][0])
            #
            
            # normalize
            #size = self._out_w * self._out_h * self._filter
            #self._gpu.get_sum(self._batch_size, self._gpu_output, self._gpu_sum, size)
            #self._gpu.copy(self._sum_array, self._gpu_sum)
            #mean = self._sum_array.sum() / float(self._batch_size*size)
            #self._gpu.get_dsum(self._batch_size, self._gpu_output, self._gpu_sum, size, mean)
            #self._gpu.copy(self._sum_array, self._gpu_sum)
            #div2 = self._sum_array.sum() / float(self._batch_size*size)
            #div = np.sqrt(div2) +  0.0000001;
            #self._gpu.get_std(self._batch_size, self._gpu_output, size, mean, div)
            
            # relu
            # activation mode
            # 0 : none
            # 1 : normal
            # 2 : 0.000001
            # 3 : y/20
            size = self._out_w * self._out_h
            self._gpu.relu(self._gpu_output, self._batch_size, self._filter, size, a_mode)
            if debug:
                print(self._index, "FCNN, reru(),", self._batch_size, self._filter, size)
                print(self._out_w, self._out_h, self._filter)
                self._gpu.copy(self._output_array, self._gpu_output)
                print(self._output_array.shape)
                print(self._output_array[0][0])
                #
                #for i in range(self._filter):
                #    name = "debug_%d" % (i)
                #    self.save_png(name, self._output_array[0][i])
                #
            #
        elif self._gpu.type==1: # GDX
            pass
        #
        self.cache = 1
        
class Roster:
    def __init__(self):
        self._weight_list = []
        self._gpu = None
        self.layers = []
        self.input = None
        self.output = None
        self._batch_size = 1
        self._data_size = 1
        #self._eval_mode = 0
        self._path = ""
        self.emode = 0
        self.wmode = 0 # 0:index, 1:float
        self.qmode = 0 # 0:32bit, 1:16bit, 2:8bit
        
    def set_backpropagation(self, sw, lr=0.01):
        self.backprop = sw
        self.learning_rate = lr
        c = self.count_layers()
        for i in range(0, c):
            layer = self.get_layer_at(i)
            layer.set_backpropagation(self.backprop, self.learning_rate)
        #
    
    # 0 : wi only / old style
    # 1 : all quantize
    
    # 0 : 32 bit
    # 1 : 16 bit
    # 2 : 8 bit
    def set_qmode(self, q):
        self.qmode = q
        c = self.count_layers()
        for i in range(c):
            layer = self.get_layer_at(i)
            layer.qmode = self.qmode
        #
        
    def set_path(self, path):
        self._path = path
        
    def save(self, mode=0):
        #print("Roster::save(%s, %d)" % (self._path, mode))
        self.export_weight(self._path, mode)
        
    def save_as(self, path, mode=0):
        print("Roster::save(%s, %d)" % (path, mode))
        self.export_weight(path, mode)
    
    def load(self, path=None, mode=-1):
        print("Roster::load(%s, %d)" % (path, mode))
        # 0:index, 1:float
        if mode<0:
            mode = self.wmode
        #
        
        if path is None:
            path = self._path
        #
        
        if os.path.isfile(path):
            self.import_weight(path, mode)
        else:
            value = 0
            self.init_weight(mode, value)
            self.export_weight(path, mode)
        #

    def set_evaluate_mode(self, mode):
        #self._eval_mode = mode
        self.emode = mode
        # 0 : CE for classification
        # 1 : MSE for autoencoder
        # 2 : MSE for regression
    
    def set_gpu(self, gpu):
        self._gpu = gpu
        self._remote = None

    def prepare(self, batch_size, data_size, num_class):
        print("Roster::prepare(), gpu type=%d, qmode=%d" % (self._gpu.type, self.qmode))
        if self._gpu:
            #print("Roster::prepare(), gpu type=%d" % (self._gpu.type))
            pass
        else:
            print("Roster::prepare(), error, no gpu")
        #
        self.num_class = num_class
        self._batch_size = batch_size
        self._data_size = data_size
        #
        if self.qmode==0:
            self._batch_data = np.zeros((self._batch_size, data_size), dtype=np.float32)
            self._labels = np.zeros((batch_size, num_class), dtype=np.float32)
            self._batch_cross_entropy = np.zeros(batch_size, dtype=np.float32)
        if self.qmode==1:
            self._batch_data = np.zeros((self._batch_size, data_size), dtype=np.float16)
            self._labels = np.zeros((batch_size, num_class), dtype=np.float16)
            self._batch_cross_entropy = np.zeros(batch_size, dtype=np.float16)
        elif self.qmode==2:
            self._batch_data = np.zeros((self._batch_size, data_size), dtype=np.uint8)
            self._labels = np.zeros((batch_size, num_class), dtype=np.float16)
            self._batch_cross_entropy = np.zeros(batch_size, dtype=np.float16)
        #

        if self._gpu.type==0: # OpenCL
            print("Roster::prepare(), OpenCL")
            self._gpu_input = self._gpu.dev_malloc(self._batch_data)
            self._gpu_labels = self._gpu.dev_malloc(self._labels)
            self._gpu_entropy = self._gpu.dev_malloc(self._batch_cross_entropy)
        elif self._gpu.type==1: # nvidia
            print("Roster::prepare(), CUPY")
            self._gpu_input = self._gpu.allocateArray(self._batch_data)
            self._gpu_labels = self._gpu.allocateArray(self._labels)
            self._gpu_entropy = self._gpu.allocateArray(self._batch_cross_entropy)
        elif self._gpu.type==2: # macOS Metal
            self._gpu_input = self._gpu.alloc_buf_from_array(self._batch_data)
            self._gpu_labels = self._gpu.alloc_buf_from_array(self._labels)
            self._gpu_entropy = self._gpu.alloc_buf_from_array(self._batch_cross_entropy)
        #
        
        self.input = self.get_layer_at(0)
        for layer in self.layers:
            layer.prepare(batch_size)
        #
        self.output = layer
    
    def direct_set_data(self, data_array):
        if self._gpu.type==0: # opencl
            self._gpu.copy(self._gpu_input, data_array) # copy(dist, src)
            self._gpu.copy(self.input._gpu_output, self._gpu_input)
        elif self._gpu.type==1: # nvidia
            self.input._gpu_output = self._gpu.allocateArray(data_array)
        elif self._gpu.type==2: # macOS Metal
            print("Roster::direct_set_data()")
            print("self._batch_data", type(self._batch_data[0][0]), self._batch_data.shape)
            print("data_array", type(data_array[0][0]), data_array.shape)
            
            self._gpu.write_np_to_mbuf(data_array, self._gpu_input)
            self._gpu.write_np_to_mbuf(data_array, self.input._gpu_output)
            #
            # copy data to np array for bp !!!!!
            #
            self.input._output_array = data_array
        #
    
    def direct_set_label(self, label_array):
        #print("Roster::direct_set_label()")
        if self._gpu.type==0: # opencl
            self._gpu.copy(self._gpu_labels, label_array)
            self.label_array = label_array
        elif self._gpu.type==1: # GDX
            self._gpu_labels = self._gpu.allocateArray(label_array)
        elif self._gpu.type==2: # macOS Metal
            #print(label_array.shape)
            #print(type(self._labels[0][0]), self._labels.shape)
            #print(type(label_array), type(label_array[0][0]), label_array.shape)
            
            self._gpu.write_np_to_mbuf(label_array, self._gpu_labels)
            self.label_array = label_array
        #
    
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
                layer.init_weight_with_mode(mode, value)
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
            print("Roster::get_layer_at(), error: %d > %d" % (i, c))
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
        elif type==LAYER_TYPE_CONV:
            print("not yet")
            return
        elif type==LAYER_TYPE_MAX:
            print("not yet")
            return
 
    def get_inference(self):
        if self._gpu:
            pass
        else:
            print("Roster::get_inference(), error, no gpu")
            return None
        #
        output = self.output
        
        if self._gpu.type==0: # OpenCL
            print("Roster::get_inference(), OpenCL")
            output._gpu.copy(output._output_array, output._gpu_output)
        elif self._gpu.type==1: # nvidia
            print("Roster::get_inference(), CUPY")
        elif self._gpu.type==2: # macOS Metal
            #print("Roster::get_inference(), macOS Metal")
            output._output_array = np.frombuffer(output._gpu_output.contents().as_buffer(output._gpu_output.length()), dtype=np.float32)
            output._output_array = output._output_array.view(np.float32).reshape(self._batch_size, 2)
        #
        return output._output_array

    def get_answer_with_confidence(self):
        ret = []
        output = self.output
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
            ret.append((max_index, max))
        #
        return ret

    def get_answer(self):
        #print("roster::get_answer()")
        ret = []
        #c = self.count_layers()
        output = self.output #self.get_layer_at(c-1)
        if self._gpu:
            if self._gpu.type==0:
                output._gpu.copy(output._output_array, output._gpu_output)
            elif self._gpu.type==1:
                output._output_array = self._gpu.allocateArray(output._gpu_softmax)
            elif self._gpu.type==2:
                if self.qmode==1:
                    output._output_array = self._gpu.read_mbuf_to_numpy(output._gpu_softmax, np.float16, output._output_array.shape)
                else:
                    output._output_array = self._gpu.read_mbuf_to_numpy(output._gpu_softmax, np.float32, output._output_array.shape)
                #
            #
        else:
            pass
        #

        for i in range(self._batch_size):
            if self._gpu.type==0:
                inf = output._output_array[i]
            elif self._gpu.type==1: # cupy
                inf = cp.asnumpy(output._output_array)[i]
            elif self._gpu.type==2:
                inf = output._output_array[i]
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
        #print("Roster::evaluate()", self.emode)
        self.propagate(debug)
        #
        if self.emode==0: # CE for classification
            ce = self.get_cross_entropy(debug)
        elif self.emode==1: # MSE for autoencoder
            self._gpu.mse(self.output._gpu_output, self.input._gpu_output, self._gpu_entropy, self._data_size, self._batch_size)
            self._gpu.copy(self._batch_cross_entropy, self._gpu_entropy)
            ce = np.sum(self._batch_cross_entropy)/np.float16(self._batch_size)
        elif self.emode==2: # MSE for regression
            #print(self._batch_size, self._data_size, self.output._output_array.shape)
            infs = self.get_inference()
            I = self.output._output_array.shape[0]
            J = self.output._output_array.shape[1]
            total = np.float32(0.0)
            #print("I:", I)
            for i in range(I):
                sub_total = np.float32(0.0)
                for j in range(J):
                    se = np.float32(0.0)
                    se = self.label_array[i][j] - self.output._output_array[i][j]
                    sub_total += se * se
                #
                total += sub_total
            #
            ce = total / np.float32(I) #self._batch_size
        #
        return ce
    
    def get_cross_entropy(self, debug=0):
        c = self.count_layers()
        output = self.get_layer_at(c-1)

        if self._gpu:
            if self._gpu.type==0: # OenCL
                #self._gpu.cross_entropy(output._gpu_output, self._gpu_labels, self._gpu_entropy, self.num_class, self._batch_size)
                self._gpu.cross_entropy(output._gpu_softmax, self._gpu_labels, self._gpu_entropy, self.num_class, self._batch_size)
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
            elif self._gpu.type==1: # nvidia
                self._gpu.crossEntropy(output._gpu_softmax, self._gpu_labels, self._gpu_entropy, self._batch_size, output._num_node)
                total = self._gpu_entropy.sum()
                avg = total / float(self._batch_size)
                
                if debug:
                    print("get_cross_entropy")
                    darray = cp.asnumpy(self._gpu_entropy)
                    print(darray)
                #
                return avg
            elif self._gpu.type==2: # macOS Metal
                self._gpu.cross_entropy(self._batch_size, self.num_class, output._gpu_softmax, self._gpu_labels, self._gpu_entropy)
                
                if self.qmode==1: # 16bit
                    self._batch_cross_entropy = self._gpu.read_mbuf_to_numpy(self._gpu_entropy, np.float16, self._batch_cross_entropy.shape)
                else: # 32bit
                    self._batch_cross_entropy = self._gpu.read_mbuf_to_numpy(self._gpu_entropy, np.float32, self._batch_cross_entropy.shape)
                #
                if debug:
                    print("Roster::get_cross_entropy()")
                    print("DEBUG CE:", self._batch_cross_entropy)
                #
                
                s = np.float64(0.0)
                for i in range(self._batch_size):
                    s += np.float64(self._batch_cross_entropy[i])
                #
                s = s / np.float64(self._batch_size)
                return s
            #
        #
        return 0.0

    def export_weight(self, path, mode=0):
        # mode 0:index, 1:value
        #print("Roster : export_weight(%s, %d)" % (path, mode))
        with open(path, "w") as f:
            writer = csv.writer(f, lineterminator='\n')
            c = self.count_layers()
            for i in range(1, c):
                layer = self.get_layer_at(i)
                if mode==0:
                    data = layer.export_weight_index()
                elif mode==1:
                    data = layer.export_weight_value()
                #
                if data:
                    writer.writerows(data)
                #
            # for
        # with
        
    
    def import_weight(self, path, mode=0):
        # mode 0:index, 1:value
        print("Roster::import_weight(%s, %d)" % (path, mode))
        
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
                if mode==0:
                    layer.import_weight_index(block)
                elif mode==1:
                    layer.import_weight_value(block)
                #
            # for
        # with

    def propagate(self, debug=0):
        c = self.count_layers()
        pre = self.get_layer_at(0)
        for i in range(1, c):
            layer = self.get_layer_at(i)
            layer.propagate(pre._gpu_output, debug)
            #
            pre = layer
        #
        #print("end of propagate()")
        
    def bp(self, debug=0):
        c = self.count_layers()
        for i in range(c-1, -1, -1):
            layer = self.get_layer_at(i)
            layer.bp(self.label_array, debug)
        #
        
def main():
    return 0

if __name__=='__main__':
    print(">> start")
    sts = main()
    print(">> end")
    print("\007")
    sys.exit(sts)
#
#
# EOF
