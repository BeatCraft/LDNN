#! c:/Python26/python.exe
# -*- coding: utf-8 -*-

import os, sys, time, math
from stat import *
import random
import copy
import multiprocessing as mp
import pickle
import numpy as np

from PIL import Image
from PIL import ImageFile
from PIL import JpegImagePlugin
from PIL import ImageFile
from PIL import PngImagePlugin
import zlib

sys.setrecursionlimit(10000)
#
# constant values
#
WEIGHT_SET_0 = [-0.90, -0.85, -0.80, -0.75, -0.70, -0.65, -0.60, -0.55, -0.50, -0.45, -0.40, -0.35, -0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

WEIGHT_SET_1 = [-1, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625, -0.0078125, 0, 0.0078125, 0.015625, 0.0625, 0.125, 0.25, 0.5, 1]

WEIGHT_SET_2 = [-1, -0.72363462, -0.52364706, -0.37892914, -0.27420624, -0.19842513, -0.14358729, -0.10390474, -0.07518906, -0.05440941, -0.03937253, -0.02849133, -0.02061731, -0.0149194, -0.01079619, -0.0078125, 0, 0.72363462, 0.52364706, 0.37892914, 0.27420624, 0.19842513, 0.14358729, 0.10390474, 0.07518906, 0.05440941, 0.03937253, 0.02849133, 0.02061731, 0.0149194, 0.01079619, 0.0078125, 1]

lesserWeights = WEIGHT_SET_1
lesserWeightsLen = len(lesserWeights)
WEIGHT_INDEX_ZERO = 8
WEIGHT_INDEX_MAX = lesserWeightsLen-1
WEIGHT_INDEX_MIN = 0
WEIGHT_RANDOM_RANGE = 6
#
#
#
def sigmoid(x):
    try:
        a = math.exp(-x)
    except OverflowError:
        a = float('inf')
        print "fuck"
    
    ret = 1.0 / (1.0 + a)
    print "sigmoid(%f)=%f" % (x, ret)
    return 1.0 / (1.0 + a)
#
#
#
def relu(x):
    if x<=0.0 :
        return 0.0#1
    return x
#
#
#
def softmax(x):
    return math.exp(x)
#return np.exp(x)

#
#
#
class Connection:
    # left and right are Node
    def __init__(self, left, right):
        self._id = -1
        self._product = 0.0
        self._weight_index = 0

        self.left = left
        self.right = right
    
    def set_id(self, id):
        if id>=0:
            self._id = id
    
    def get_id(self):
        return self._id

    def getWeightIndex(self):
        return self._weight_index

    def setWeightIndex(self, index):
        if index>=0 and index<lesserWeightsLen:
            self._weight_index = index
 
        return self._weight_index
            
    def calc_product(self):
        node = self.left
        y = node.getY()
        self._product = y * lesserWeights[self._weight_index]
        return self._product

    def get_product(self):
        return self._product

    def propagate(self):
        return self.calc_product()
#
#
#
class Node:
    # inputs and outputs are lists of Connection
    # x is input value
    # y is output value
    # weights are hold by Connections
    def __init__(self):
        self._id = -1
        self._sum = 0
        self._input_filter_index = 0
        self._output_filter_index = 0
        
        self.x = 0
        self.y = 0
        self.bias = 0
        self.inputs = []
        self.outputs = []

    def set_output_filter(self, i):
        self._output_filter_index = i

    def set_input_filter(self, i):
       self._input_filter_index = i
           
    def set_id(self, id):
        if id>=0:
            self._id = id

    def get_id(self):
        return self._id

    def getInputs(self):
        return self.inputs

    def getOutputs(self):
        return self.outputs

    def addInputs(self, c):
        self.inputs.append(c)

    def addOutputs(self, c):
        self.outputs.append(c)

    def dump(self):
        print"    node : %f" % self.y

    def getX(self):
        return self.x
    
    def setX(self, v):
        self.x = v

    def getY(self):
        return self.y

    def setY(self, y):
        self.y = y

    def getBias(self):
        return self.bias
    
    def setBias(self, b):
        self.bias = b

    def propagate(self):
        sum = 0.0
        num = len(self.inputs)
        
        if num==0:
            if self._input_filter_index==1:
                if self.x>0:
                    self.y = float(self.x)/255.0
                else:
                    self.y = 0.0
        else:
            for i in range(num):
                input = self.inputs[i]
                sum += input.propagate()

            if self._output_filter_index==0:
                self.y = sum
            elif self._output_filter_index==1:
                self.y = relu(sum)
            elif self._output_filter_index==2:
                self.y = softmax(sum)

    def get_weight_array(self):
        wl = []
        for con in self.inputs:
            i = con.getWeightIndex()
            w = lesserWeights[i]
            wl.append(w)

        wa = np.array(wl)
        return wa
#
#
#
class Layer:
    def __init__(self, i, type):
        self._id = -1
        
        self.index = i
        self.type = type # 0 : input, 1 : hidden, 2 : output
        self.nodes = []

    def set_id(self, id):
        if id>=0:
            self._id = id

    def get_id(self):
        return self._id
    
    def getType(self):
        return self.type

    def countNodes(self):
        return len(self.nodes)

    def getNodes(self):
        if self.countNodes() == 0:
            return 0
        return self.nodes

    def addNode(self, node):
        # 2018-12-06 by lesser
        # this part is temporaly implemented for multi-processing, shuod be re-desinged later
        if self.type==0:
            node.set_input_filter(1) # scaling
            #set_output_filter(0) # default=0, no filter
        elif self.type==1:
            node.set_output_filter(1)
        elif self.type==2:
            node.set_output_filter(2)
        #
        self.nodes.append(node)
        return self.countNodes()

    def dump(self):
        out = "layer(%d) : " % self.index
        for node in self.nodes:
            v = "%f, " % node.getY()
            out = out + v
        print out

    def propagate(self):
        c = len(self.nodes)
        if not c>0:
            print "error"
            return
        
        for node in self.nodes:
            node.propagate()

    def get_weight_array(self):

        wl = []
        for node in self.nodes:
            wl.append(node.get_weight_array())

        a = np.array(wl)
        return a
#
#
#
class Roster:
    def __init__(self):
        self.layers = []
        self.connections = []
    
    def init_connections(self):
        i = 0
        for con in self.connections:
            con.set_id(i)
            i = i + 1
    
    def countLayers(self):
        return len(self.layers)

    def getLayers(self):
        if self.countLayers() == 0:
            return 0
        return self.layers

    def getLayerAt(self, i):
        c = self.countLayers()
        if i>=c:
            print "error : Roster : getLayerAt"
            return None
        return self.layers[i]

    def addLayer(self, num_of_nodes, type):
        c = self.countLayers()
        # create a layer
        layer = Layer(c, type)
        
        # create nodes
        for n in range(num_of_nodes):
            node = Node()
            layer.addNode(node)

        self.layers.append(layer)
        return layer

    def connectNodes(self, leftNode, rightNode):
        c = Connection(leftNode, rightNode)
        #
        #i = WEIGHT_INDEX_ZERO + 4
        i = random.randrange(lesserWeightsLen)
        #i = random.randrange(WEIGHT_INDEX_ZERO, WEIGHT_INDEX_ZERO+WEIGHT_RANDOM_RANGE, 1)
        #
        c.setWeightIndex(i)
        rightNode.addInputs(c)
        leftNode.addOutputs(c)
        self.connections.append(c)

    def countConnections(self):
        return len(self.connections)
    
    def getConnections(self):
        return self.connections

    def connectLayers(self, leftLayer, rightLayer):
        for rightNode in rightLayer.getNodes():
            for leftNode in leftLayer.getNodes():
                self.connectNodes(leftNode, rightNode)

    def setInputData(self, data): # data is a list
        inputLayer = self.getLayerAt(0)
        c = inputLayer.countNodes()
        n = len(data)
        if n == c:
            pass
        else:
            print "error : num of data : %d : %d" % (n, c)
            return 0

        nodes =  inputLayer.getNodes()
        i = 0
        for node in nodes:
            node.setX(data[i])
            i = i + 1
        return 1

    def dump(self):
        for layer in self.layers:
            layer.dump()

    def dumpLayer(self, index):
        layer = self.getLayerAt(index)
        layer.dump()

    def propagate(self):
        c = self.countLayers()
        if c<=0:
            print "error : Roster : propagate"
            return
        
        #prev = self.getLayerAt(0)
        for i in range(0, c):
            layer = self.getLayerAt(i)
            layer.propagate()

    def getInferences(self, softmax=0):
        c = self.countLayers()
        if c<=0:
            return None
        
        output = self.getLayerAt(c-1)
        nodes = output.getNodes()
        ret = []
        sum = 0.0
        if softmax:
            for node in nodes:
                sum += node.getY()
            
            if sum==0.0:
                print "FUCK FUCK"
                for node in nodes:
                    print node.getY()
                    ret.append(0.0)
            else:
                for node in nodes:
                    ret.append(node.getY()/sum)
                        
        else: # usually, this part is not used
            for node in nodes:
                ret.append(node.getY())
        
        return ret

    def get_weight_array(self):
        c = self.countLayers()
        if c<=0:
            print "error : Roster : get_weight_array"
            return
        
        for i in range(0, c):
            layer = self.getLayerAt(i)
            a = layer.get_weight_array()
            print a
#
#
#
def main():
    r = Roster()
    inputLayer = r.addLayer(196, 0)    # 0 : input
    hiddenLayer_1 = r.addLayer(10, 1)  # 1 : hiddeh
    hiddenLayer_2 = r.addLayer(10, 1)  # 2 : hiddeh
    outputLayer = r.addLayer(10, 2)    # 3 : output
        
    r.connectLayers(inputLayer, hiddenLayer_1)
    r.connectLayers(hiddenLayer_1, hiddenLayer_2)
    r.connectLayers(hiddenLayer_2, outputLayer)
    
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
# memo / todo
#

# 2018/11/19
# scaling function to inpout layer

# 2018/11/19
# select method for activate functions

# 2018/11/20
# try simple model : (28*28)*16*16

# 2018/11/20
# simplify algo1 : try 10 inc by random, and then 10 dec
