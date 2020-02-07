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
    def __init__(self, local_addr, local_port, remote_addr, remote_port, batch_size, batch_start, device_id, package_id):
        print "ClientLooper::__init__()"
        super(ClientLooper, self).__init__(local_addr, local_port, remote_addr, remote_port)
        #
        self._batch_size = batch_size
        self._batch_start = batch_start
        self._device_id = device_id
        self._package_id = package_id
        #
        self._send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
    def setup(self):
        platform_id = 0
        my_gpu = gpu.Gpu(platform_id, self._device_id)
        my_gpu.set_kernel_code()
        self._package = util.Package(self._package_id)
        self._roster = self._package.setup_dnn(my_gpu)
        if self._roster is None:
            print "fatal DNN error"
        #
        # set batch
        batch_start = self._batch_start
        batch_size = self._batch_size
        self._package.load_batch()
        self._roster.set_batch(self._package._train_image_batch, self._package._train_label_batch, batch_start, batch_size, self._package._image_size, self._package._num_class, 0)
        
        # evaluate
        start_time = time.time()
        self._roster.propagate()
        ce = self._roster.get_cross_entropy()
        elapsed_time = time.time() - start_time
        t = format(elapsed_time, "0")
        print "time = %s" % (t)
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
            if a==10:   # init
                print "init"
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
                cmd = netutil.pack_if(35, ce)
                self.send(cmd)
            elif a==40: # update
                layer = self._roster.getLayerAt(b)
                layer.set_weight_index(c, d, e)
                layer.update_weight_gpu()
                self._roster.propagate()
                ce = self._roster.get_cross_entropy()
                cmd = netutil.pack_if(45, ce)
                self.send(cmd)
            elif a==60 : # debug
                print "debug"
                cmd = netutil.pack_if(65, 1.0)
                self.send(cmd)
            else:
                print "unknown command"
                pass
            # if
        # while
    # end of loop()
#
#
#
def client(BC_ADDR, BC_PORT, SERVER_ADDR, SERVER_PORT, batch_size, batch_start, device_id, package_id):
    print "client()"
    #
    c = ClientLooper(BC_ADDR, BC_PORT, SERVER_ADDR, SERVER_PORT, batch_size, batch_start, device_id, package_id)
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
    #BC_ADDR = "192.168.200.255"
    BC_PORT = 5000
    #SERVER_ADDR = "192.168.200.10"
    SERVER_ADDR = "127.0.0.1"
    SERVER_PORT = 5005
    #
    batch_size = 5000
    batch_start = 0
    device_id = 1#0
    # 0 : AMD Server"
    # 1 : Intel on MBP"
    # 2 : eGPU (AMD Radeon Pro 580)"
    package_id = 0 # MNIST
    #
    client(BC_ADDR, BC_PORT, SERVER_ADDR, SERVER_PORT, batch_size, batch_start, device_id, package_id)
#
#
#
if __name__ == '__main__':
	main()


