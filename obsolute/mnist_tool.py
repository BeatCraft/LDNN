#! c:/Python26/python.exe
# -*- coding: utf-8 -*-

from PIL import Image
from PIL import ImageFile
from PIL import JpegImagePlugin
from PIL import ImageFile
from PIL import PngImagePlugin

import zlib

import os, sys, time
from stat import *
import struct
import csv
import pickle

import util

path_train_image = "./MNIST/train-images-idx3-ubyte"
path_train_label = "./MNIST/train-labels-idx1-ubyte"
path_test_image  = "./MNIST/t10k-images-idx3-ubyte"
path_test_label  = "./MNIST/t10k-labels-idx1-ubyte"

num_train_set = 60000
num_test_set  = 10000

img_width  = 28
img_height = 28
img_size   = img_width*img_height

path_data = "./data/"
path_train_save = "./data/train/"
path_test_save  = "./data/test/"

#
#
#
def check_dir():
    if not os.path.isdir(path_data):
        os.makedirs(path_data)

    if not os.path.isdir(path_train_save):
        os.makedirs(path_train_save)
    for i in range(10):
        path = path_train_save + str(i) + "/"
        if not os.path.isdir(path):
            os.makedirs(path)
#        print path

    if not os.path.isdir(path_test_save):
        os.makedirs(path_test_save)
    for i in range(10):
        path = path_test_save + str(i) + "/"
        if not os.path.isdir(path):
            os.makedirs(path)
#        print path
#
#
#
def save_png(img, path):
    img.save(path)
#
#
#
def save_raw(img, path):
    file = None
    
    try:
        file = open(path, 'wb')
    except:
        print "file error(%s)" % path_out
        return
    
    file.write( bytearray( img.getdata() ) )
    file.close()


def convert_image(path, save_dir, num):
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    file_in = open(path)
    header = file_in.read(16)
    
    for i in range(num):
        data = file_in.read(img_size)
        imgSize = (img_width, img_height)
        img = Image.frombytes('L', imgSize, data, 'raw')
        
        name = "%05d" % i
        # png
        path_out = save_dir + name + ".png"
        save_png(img, path_out)
        # raw
        #path_out = save_dir + name + ".raw"
        #save_raw(img, path_out)
    
    file_in.close()
#
#
#
def convert_label_to_list(path, num):
    file_in = open(path)
    data = file_in.read()
    
    label_list = [0 for i in range(num)]
    
    for i in range(8, num+8):
        label = struct.unpack('>B', data[i])
        label_list[i-8] = label[0]

    return label_list
#
#
#
def covert_list_to_csv(path, list):
    with open(path, 'wb') as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)
        wr.writerow(list)
#
#
#
def save_pickele(path_out, obj):
    with open(path_out, mode='wb') as f:
        pickle.dump(obj, f)
#
#
#
def sort_samples(path_labels, max, path_target):
    labels = []
    with open(path_labels, mode='rb') as f:
        labels = pickle.load(f)
    
    for i in range(10000, max):
        name = "%05d" % i
        path_from = path_target + name + ".png"
        path_to = path_target + str(labels[i])+ "/"
        if not os.path.isdir(path_to):
            print os.makedirs(path_to)
        shutil.move(path_from, path_to)
#
#
#
def extract_image(labels, num, path, path_save):
    file_in = open(path)
    header = file_in.read(16)
    
    for i in range(num):
        data = file_in.read(img_size)
        imgSize = (img_width, img_height)
        img = Image.frombytes('L', imgSize, data, 'raw')
        name = "%05d" % i
        path_out = path_save + str(labels[i])+ "/" + name + ".png"
        print path_out
        save_png(img, path_out)
    
    file_in.close()
#
#
#
def prepare_samples():
    check_dir()
    
    labels = convert_label_to_list(path_train_label, num_train_set)
    extract_image(labels, num_train_set, path_train_image, path_train_save)
    
    labels = convert_label_to_list(path_test_label, num_test_set)
    extract_image(labels, num_test_set, path_test_image, path_test_save)
#
#
#
def get_pixmap(path):
    img = Image.open(path)
    img = img.convert("L")

    pix = img.load()
    w = img.size[0]
    h = img.size[1]

    for y in range(h):
        for x in range(w):
            print pix[x, y]

#
#
#
def process_data():
    
    base_path = "/Users/lesser/ldnn/data/train/"
    for i in range(10):
        path = base_path + "%d/" % i
        print path
        
        files = []
        save_list = [0 for p in range(28*28)]
        cnt = 0
        for f in os.listdir(path):
            file_path = os.path.join(path, f)
            file_name, file_extension = os.path.splitext(f)
            #print file_name
            #print file_extension
            if file_extension == ".png":
                cnt = cnt + 1
                data_list = util.loadData(file_path)
                for p in range(28*28):
                    save_list[p] = save_list[p] + data_list[p]
    
        #print save_list
        #print cnt
        for j in range(28*28):
            save_list[j] = save_list[j]/cnt
        #print save_list
        
        save_img = Image.new('L', (28, 28))
        pix = save_img.load()


        k = 0
        for y in range(28):
            for x in range(28):
                pix[x, y] = save_list[k]
                k = k + 1

        save_name = "%d.png" % (i)
        save_path = os.path.join("/Users/lesser/ldnn/mini/", save_name)
        print save_path
        save_img.save(save_path)



#print len(files)

#
#
#
def main():
    argvs = sys.argv
    argc = len(argvs)

    process_data()
    
    
    
#
#    prepare_samples()
#    get_pixmap("./data/train/0/00001.png")
#
    return 0
#
#
#
if __name__=='__main__':
    print ">> start"
    sts = main()
    print ">> end"
    sys.exit(sts)
