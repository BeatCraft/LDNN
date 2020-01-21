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
    def __init__(self, local_addr, local_port, remote_addr, remote_port):
        print "ClientLooper::__init__()"
        super(ClientLooper, self).__init__(local_addr, local_port, remote_addr, remote_port)
        #
        #self._state = 0
        
    def setup(self):
        package_id = 0 # MNIST
        # 0 : AMD Server"
        # 1 : Intel on MBP"
        # 2 : eGPU (AMD Radeon Pro 580)"
        platform_id = 0
        device_id = 0
        my_gpu = gpu.Gpu(platform_id, device_id)
        my_gpu.set_kernel_code()
        self._package = util.Package(package_id)
        self._roster = self._package.setup_dnn(my_gpu)
        if self._roster is None:
            print "fatal DNN error"
        #
        # set batch
        batch_start = 0
        batch_size = 500
        self._package.load_batch()
#        self._roster.set_batch(self._package._train_image_batch, self._package._train_label_batch, batch_size, self._package._image_size, self._package._num_class)
        self._roster.set_batch(self._package._train_image_batch, self._package._train_label_batch, batch_start, batch_size, self._package._image_size, self._package._num_class, 0)
        
        # evaluate
        self._roster.propagate()
        ce = self._roster.get_cross_entropy()
        print ce
        
        
    def loop(self):
        print "ClientLooper::loop() - start"
        
        self.setup()
        #
        while not self.is_quite_requested():
            # recv a packet
            res, addr = self.recv()
            if res==None:
                time.sleep(0.01)
                continue
            #
            # decode and response
            #
            a, b, c, d, e = netutil.unpack_i5(res)
            #print a
            if a==10: # init
                cmd = netutil.pack_if(15, 1.0)
                self.send(cmd)
            elif a==20: # evaluate
                self._roster.propagate()
                ce = self._roster.get_cross_entropy()
                cmd = netutil.pack_if(25, ce)
                self.send(cmd)
            elif a==30: # alt
                self._roster.propagate(b, c, d, e, 0)
                ce = self._roster.get_cross_entropy()
                #print "set_alt(%d, %d, %d, %d)=%f" %(b, c, d, e, ce)
                cmd = netutil.pack_if(35, ce)
                self.send(cmd)
            elif a==40: # update
                layer = self._roster.getLayerAt(b)
                layer.set_weight_index(c, d, e)
                layer.update_weight_gpu()
                self._roster.propagate()
                ce = self._roster.get_cross_entropy()
                #print "update(%d, %d, %d, %d)=%f" %(b, c, d, e, ce)
                #print ce
                cmd = netutil.pack_if(45, ce)
                self.send(cmd)
            else:
                print "unknown command"
                pass
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
    c = ClientLooper(BC_ADDR, BC_PORT, SERVER_ADDR, SERVER_PORT)
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
if __name__ == '__main__':
	main()


