#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import traceback
import csv
import socket
import time
#import command
import multiprocessing
import struct
import binascii
#
import numpy as np
import sys,os
import random
#
#import main
import core
import util
import gpu
import netutil
import train
#
#
#
class ServerLooper(netutil.Looper):
    def __init__(self, local_addr, local_port, remote_addr, remote_port,
                 package_id, config_id, mini_batch_size, num_client, epoc):
        print("ServerLooper::__init__()")
        #
        super(ServerLooper, self).__init__(local_addr, local_port, remote_addr, remote_port)
        #
        self._package_id = package_id
        self._config_id = config_id
        self._send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._send_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._mini_batch_size = mini_batch_size
        self._num_client = num_client
        self._client_num = 0
        self._client_list = []
        #
        #package_id = 0 # MNIST
        my_gpu = None
        self._package = util.Package(package_id)
        self._roster = self._package.setup_dnn(my_gpu, self._config_id)
        if self._roster is None:
            print("fatal DNN error")
        #
        self._roster.set_remote(self)
        self._seq = 0
        self._epoc = epoc
        
    def set_client_num(self, num):
        self._client_num = num

    def recv_multi(self):
        ret = 0.0
        timeout = 0
        error = 0.0
        for i in range(self._client_num):
            self._client_timeout[i] = 0
        #
        start_time = time.time()
        max = self._client_num*2
        cnt = 0
        for i in range(max):
            res, addr = self.recv()
            if res:
                pass
            else: # timeout
                for k in range(self._client_num):
                    print("    timeout:")
                    print("        %s, %d" % (self._client_list[k], self._client_timeout[k]))
                #
                timeout = timeout + 1
                continue
            #
            if addr[0] in self._client_list:
                k = self._client_list.index(addr[0])
                self._client_timeout[k] = 1
            else: # address is not registered
                print("    bad address : %s, %d" % (addr[0], addr[1]))
                error = error + 1
                continue
            #
            seq, ce = netutil.unpack_if(res)
            if seq==self._seq:
                ret =  ret + ce
            else: # bad seq
                print("   seq error : %s, %d, %d (%d)" % (addr[0], addr[1], seq, self._seq))
                error = error + 1
                continue
            #
            cnt = cnt + 1
            if self._client_num==cnt:
                break
        #
        if timeout + error>0 and cnt<self._client_num:
            return -1.0
        #
        self._seq = self._seq + 1
        return ret
        #return ret/float(self._client_num)
    
    def execute_cmd(self, mode, a, b, c, d):
        self.set_mode(mode)
        cmd = netutil.pack_i6(self._seq, mode, a, b, c, d)
        #print "send seq=%d" %(self._seq)
        self.send(cmd)
        #
        self.set_mode(mode+5)
        ret = self.recv_multi()
        #
        self.set_mode(mode+5)
        #self._seq = self._seq + 1
        return ret
    
    def evaluate(self):
        mode = 20
        #
        e = self.execute_cmd(mode, 0, 0, 0, 0)
        return e / float(self._num_client)
        
    def set_alt(self, li, ni, ii, wi):
        mode = 30
        #
        e = self.execute_cmd(mode, li, ni, ii, wi)
        return e / float(self._num_client)
        
    def update(self, li, ni, ii, wi):
        mode = 40
        #
        e = self.execute_cmd(mode, li, ni, ii, wi)
        return e / float(self._num_client)
    
    def set_batch(self, i):
        mode = 70
        return self.execute_cmd(mode, i, 0, 0, 0)
    
    def loop(self):
        print("ServerLooper::loop() - start")
        while not self.is_quite_requested():
            mode = self.mode()
            if mode==0:
                time.sleep(0.01)
                continue
            elif mode==10: # init
                #
                cmd = netutil.pack_i6(self._seq, mode,
                                      self._package_id, self._config_id, self._mini_batch_size, self._num_client)
                self.send(cmd)
                self.set_mode(mode+5)
                #
                res, addr = self.recv()
                while res:
                    print("%s : %d" % (addr[0], addr[1]))
                    self._client_list.append(addr[0])
                    res, addr = self.recv()
                #
                self._client_num = len(self._client_list)
                self._client_timeout = [0] * self._client_num
                print(self._client_num)
                if self._num_client!=self._client_num:
                    print("error : %d expected, %d answered." % (self._num_client, self._client_num))
                    #self.quit()
                #
            elif mode==20: # evaluate
                ce = self.evaluate()
                print("evaluate :%f" % (ce))
            elif mode==30: # alt
                cmd = netutil.pack_i6(self._seq, mode, 0, 1, 0, 0)
                self.send(cmd)
                self.set_mode(mode+5)
                #
                ret = self.recv_multi()
                print(ret)
                self.set_mode(0)
            elif mode==40: # update
                cmd = netutil.pack_i6(self._seq, mode, 0, 1, 0, 0)
                self.send(cmd)
                self.set_mode(mode+5)
                #
                ret = self.recv_multi()
                print(ret)
                self.set_mode(0)
            elif mode==50: # train
                t = train.Train(self._package, self._roster)
                t.set_limit(0.000001)
                t.set_mini_batch_size(self._mini_batch_size)
                #t.set_divider(64)
                it = self._package._train_batch_size / (self._mini_batch_size*self._num_client)
                t.set_iteration(it)
                t.set_epoc(self._epoc)
                t.set_layer_direction(1) # output to input
                #print "train_batch_size : %d" % (self._package._train_batch_size)
                #print "mini_batch_size : %d" % (self._mini_batch_size)
                #print "num_client : %d" % (self._num_client)
                #print "div : %d" % (self._mini_batch_size*self._num_client)
                #print "train it : %d" % (it)
                t.loop()
                self.set_mode(0)
            elif mode==60: # debug / ping
                cmd = netutil.pack_i6(self._seq, mode, 0, 0, 0, 0)
                self.send(cmd)
                self.set_mode(mode+5)
                ret = self.recv_multi()
                print(ret)
                self.set_mode(0)
            elif mode==70: # set_batch()
                print("set_batch()")
                cmd = netutil.pack_i6(self._seq, mode, 0, 0, 0, 0)
                self.send(cmd)
                #
                time.sleep(10)
                #
                ret = self.recv_multi()
                print(ret)
                self.set_mode(0)
            else:
                self.set_mode(0)
            #
        #
        print("ServerLooper::loop() - end")
#
#
#
def server(SERVER_ADDR, SERVER_PORT, BC_ADDR, BC_PORT, package_id, config_id, mini_batch_size, num_client, epoc):
    print("main() : start")
    #
    s = ServerLooper(SERVER_ADDR, SERVER_PORT, BC_ADDR, BC_PORT, package_id, config_id, mini_batch_size, num_client, epoc)
    s.init()
    s.run()
    #
    loop = 1
    while loop:
        key = input("cmd >")
        print(key)
        if key=='q' or key=='Q':
            loop = 0
            s.quit()
        elif key=='i' or key=='I':
            s.set_mode(10)
        elif key=='e' or key=='E':
            s.set_mode(20)
        elif key=='a' or key=='A':
            s.set_mode(30)
        elif key=='u' or key=='U':
            s.set_mode(40)
        elif key=='t' or key=='T':
            s.set_mode(50)
        elif key=='d' or key=='D':
            s.set_mode(60)
        elif key=='m' or key=='M':
            s.set_mode(70)
        #
    #
    print("main() : end")
#
#
#
def main():
    print("main() : start")
    #
    BC_ADDR = "127.0.0.1"
    BC_PORT = 5000
    SERVER_ADDR = "127.0.0.1"
    SERVER_PORT = 5005
    #
    package_id = 0 # MNIST
    config_id = 1 # CNN
    mini_batch_size = 400
    num_client = 1
    epoc = 12
    #
    s = server(SERVER_ADDR, SERVER_PORT, BC_ADDR, BC_PORT, package_id, config_id, mini_batch_size, num_client, epoc)
    #
    print("main() : end")
#
#
#
if __name__ == '__main__':
	main()


