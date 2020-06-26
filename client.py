#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import traceback
import csv
import socket
import time
import multiprocessing
import struct
import binascii
#
import numpy as np
import sys,os
import core
import util
import gpu
import netutil
#
#
#
class ClientLooper(netutil.Looper):
    def __init__(self, local_addr, local_port, remote_addr, remote_port, platform_id, device_id, client_id):
        print "ClientLooper::__init__()"
        super(ClientLooper, self).__init__(local_addr, local_port, remote_addr, remote_port)
        #
        #self._batch_size = batch_size
        #self._part_start = part_start
        #self._part_size = part_size
        self._platform_id = platform_id
        self._device_id = device_id
        #self._package_id = package_id
        self._client_id = client_id
        self._it_cnt = 0
        #
        self._send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
    def setup(self):
        self._mini_cnt = 0
        my_gpu = gpu.Gpu(self._platform_id, self._device_id)
        my_gpu.set_kernel_code()
        #
        self._package = util.Package(self._package_id)
        self._roster = self._package.setup_dnn(my_gpu)
        if self._roster is None:
            print "fatal DNN error"
            return 0
        #
        self._package.load_batch()
        #
        self._data_size = self._package._image_size
        self._num_class = self._package._num_class
        self._data_array = np.zeros((self._mini_batch_size , self._data_size), dtype=np.float32)
        self._class_array = np.zeros(self._mini_batch_size , dtype=np.int32)
        #
        self._roster.prepare(self._mini_batch_size, self._data_size, self._num_class)
        
        #it = self._it_cnt * self._num_client + self._client_id
        #self._package_id = b
        #self._mini_batch_size = c
        #self._num_client = d
        return 1
        #
#
#
#        #
#        self.set_batch(0)
        #
        # evaluate
        #
#        start_time = time.time()
#        self._roster.propagate()
#        ce = self._roster.get_cross_entropy()
#        elapsed_time = time.time() - start_time
#        t = format(elapsed_time, "0")
#        print "time = %s" % (t)
#        print ce
        #
        
    def set_batch(self):
        start = self._it_cnt * self._mini_batch_size * self._num_client + self._mini_batch_size * self._client_id
        for i in range(self._mini_batch_size):
            self._data_array[i] = self._package._train_image_batch[start + i]
            self._class_array[i] = self._package._train_label_batch[start + i]
        #
        print "set batch : %d + %d" % (start, self._mini_batch_size)
        self._roster.set_data(self._data_array, self._data_size, self._class_array, self._mini_batch_size)
        
        self._roster.propagate()
        ce = self._roster.get_cross_entropy()
        print "index=%d, entropy=%f" % (self._it_cnt, ce)
        self._it_cnt = self._it_cnt + 1
        
#        batch_size = self._batch_size
#        part_start = self._part_start
#        part_size = self._part_size
#        load_path = "../ldnn_config/%s/mini/%d/%03d.pickle" % (self._package._name, batch_size, mini_index)
#        print "loading a batch : %s" % (load_path)
        #
#        random_index = util.pickle_load(load_path)
        #print random_index
#        for i in range(part_size):
#            bi = random_index[part_start+i]
            #print bi
#            self._data_array[i] = self._package._train_image_batch[bi]
#            self._class_array[i] = self._package._train_label_batch[bi]
        #
#        self._roster.set_data(self._data_array, self._data_size, self._class_array, part_size)
        #
        
    def loop(self):
        print "ClientLooper::loop() - start"
        #
        while not self.is_quite_requested():
            # recv a packet
            res, addr = self.recv()
            if res==None:
                #print "**** recv sock timeout"
                time.sleep(0.01)
                continue
            #
            # decode and response
            #
            seq, a, b, c, d, e = netutil.unpack_i6(res)
            #print "recv seq = %d" % (seq)
            if a==10:   # init
                print "init"
                self._package_id = b
                self._mini_batch_size = c
                self._num_client = d
                self._it_cnt = 0
                print "package_id=%d, mini_batch_size=%d, num_client=%d" % (b, c, d)
                #
                cmd = netutil.pack_if(15, 1.0)
                self.send(cmd)
                #
                self.setup()
            elif a==20: # evaluate
                #print "N/A"
                self._roster.propagate()
                ce = self._roster.get_cross_entropy()
                cmd = netutil.pack_if(seq, ce)
                self.send(cmd)
            elif a==30: # alt
                self._roster.propagate(b, c, d, e, 0)
                ce = self._roster.get_cross_entropy()
                cmd = netutil.pack_if(seq, ce)
                self.send(cmd)
            elif a==40: # update
                layer = self._roster.getLayerAt(b)
                layer.set_weight_index(c, d, e)
                layer.update_weight()
                self._roster.propagate()
                ce = self._roster.get_cross_entropy()
                cmd = netutil.pack_if(seq, ce)
                self.send(cmd)
            elif a==60: # debug
                print "debug"
                cmd = netutil.pack_if(seq, 1.0)
                self.send(cmd)
            elif a==70: # set_batch()
                print "set_batch(%d|%d)" % (b, self._it_cnt)
                self.set_batch()
                cmd = netutil.pack_if(seq, 1.0)
                self.send(cmd)
            else:
                print "unknown command"
                pass
            #
        # while
    # end of loop()
#
#
#
def client(BC_ADDR, BC_PORT, SERVER_ADDR, SERVER_PORT, platform_id, device_id, client_id):
    print "client()"
    #
    c = ClientLooper(BC_ADDR, BC_PORT, SERVER_ADDR, SERVER_PORT, platform_id, device_id, client_id)
    c.init()
    c.run()
    #
    loop = 1
    while loop:
        key = raw_input("cmd >")
        print key
        if key=='q' or key=='Q':
            loop = 0
            c.quit()
    #
    print "main() : end"
#
#
#
def main():
    print "main() : start"
    #
    BC_ADDR = "127.0.0.1"
    BC_PORT = 5000
    SERVER_ADDR = "127.0.0.1"
    SERVER_PORT = 5005
    #
    #batch_size = 10000
    #part_start = 0
    #part_size = 2000
    #
    platform_id = 0
    device_id = 1
    client_id = 0
    #
    client(BC_ADDR, BC_PORT, SERVER_ADDR, SERVER_PORT, platform_id, device_id, client_id)
    #
#
#
#
if __name__ == '__main__':
	main()


