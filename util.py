#! c:/Python26/python.exe
# -*- coding: utf-8 -*-

import os, sys, time, math
from stat import *
import random
import copy

import pickle
import numpy as np

from PIL import Image
from PIL import ImageFile
from PIL import JpegImagePlugin
from PIL import ImageFile
from PIL import PngImagePlugin
import zlib

import csv
#
# constant values
#
C_PURPLE = (255, 0, 255)
C_RED = (255, 0, 0)
C_ORANGE = (255, 128, 0)
C_YELLOW = (255, 255, 0)
C_RIGHT_GREEN = (128, 255, 128)
C_GREEN = (0, 255, 0)
C_RIGHT_BLUE = (128, 128, 255)
C_BLUE = (0, 0, 255)
COLOR_PALLET = [C_BLUE, C_RIGHT_BLUE, C_GREEN, C_RIGHT_GREEN,
                C_YELLOW, C_ORANGE, C_RED, C_PURPLE]
#
#
#
def mean_squared_error(y, y_len, t, t_len):
    if y_len != t_len:
        return None
    
    s = 0.0
    for i in range(y_len):
        s += (y[i]-t[i])**2.0
    
    s = s * 0.5
    return s
#
#
#
def mean_squared_error_B(y, y_len, t, t_len):
    if y_len != t_len:
        return None
    
    s = 0
    for i in range(y_len):
        s += abs(y[i]-t[i])

    return s
#
# mostly, mean_absolute_error is used
#
def mean_absolute_error(y, y_len, t, t_len):
    if y_len != t_len:
        return None
    
    s = 0.0
    sum = 0.0
    
    for i in range(y_len):
        sum += y[i]
        s += abs(y[i]-t[i])
    
    if sum<=0:
        print "FUCK(%f)" % sum
    
    return s/float(y_len)
#    if sum>0.0:
#       return s/y_len
#    return 100.0
#
#
#
def img2List(img):
    pix = img.load()
    w = img.size[0]
    h = img.size[1]
    
    ret = []
    for y in range(h):
        for x in range(w):
            k = pix[x, y] #/ 255.0
            ret.append(k)
    return ret
#
#
#
def loadData(path):
    img = Image.open(path)
    img = img.convert("L")
    img = img.resize((14,14))
    data = img2List(img)
    return data
#
#
#
def pickle_save(path, data):
    with open(path, mode='wb') as f:
        pickle.dump(data, f)
#
#
#
def pickle_load(path):
    try:
        with open(path, mode='rb') as f:
            data = pickle.load(f)
            return data
    except:
        return None
#
#
#
def exportPng(r, num_of_processed):
    connections = r.getConnections()
    wlist = []
    for con in connections:
        w = con.getWeightIndex()
        wlist.append(w)
        
        lc = r.countLayers()
        bc = lc - 1
        width = 0
        height = 0
    
    for i in range(bc):
        left = r.getLayerAt(i)
        left_nc = left.countNodes()
        right = r.getLayerAt(i+1)
        right_nc = right.countNodes()
        width = width + right_nc*4
        if left_nc + right_nc > height:
            height = left_nc*4 + right_nc*4
    
    img = Image.new("RGB", (width+100, height+10))
    pix = img.load()
    windex = 0
    w = 0
    h = 0
    
    for i in range(bc):
        left = r.getLayerAt(i)
        left_nc = left.countNodes()
        right = r.getLayerAt(i+1)
        right_nc = right.countNodes()
        
        start_w = left_nc*4*i + 10*i
        start_h = 0
        
        for x in range(right_nc):
            w = start_w + x*4
            for y in range(left_nc):
                h = start_h + y*4
                #print "[%d]%d, %d : %d" % (i, w, h, windex)
                wv = wlist[windex]
                v = COLOR_PALLET[wv]
                pix[w,   h  ] = v
                pix[w,   h+1] = v
                pix[w,   h+2] = v
                pix[w+1, h  ] = v
                pix[w+1, h+1] = v
                pix[w+1, h+2] = v
                pix[w+2, h  ] = v
                pix[w+2, h+1] = v
                pix[w+2, h+2] = v
                windex = windex + 1

    start_h = 0

    save_name = "./%05d.png" % (num_of_processed)
    img.save(save_name)
#
#
#
def list_to_csv(path, data_list):
    with open(path, 'wb') as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)
        wr.writerow(data_list)
#
#
#
def csv_to_list(path):
    print "csv_to_list()"
    data_list = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            for cell in row:
                data_list.append(cell)

    return data_list
#
#
#
