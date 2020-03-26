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
import main
import core
import util
import gpu
import netutil
#
#
#
class ServerLooper(netutil.Looper):
    def __init__(self, local_addr, local_port, remote_addr, remote_port, package_id):
        print "ServerLooper::__init__()"
        #
        super(ServerLooper, self).__init__(local_addr, local_port, remote_addr, remote_port)
        #
        self._package_id = package_id
        self._send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._send_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._client_num = 0
        self._client_list = []
        #
        #package_id = 0 # MNIST
        my_gpu = None
        self._package = util.Package(package_id)
        self._roster = self._package.setup_dnn(my_gpu)
        if self._roster is None:
            print "fatal DNN error"
        #
        self._roster.set_remote(self)
        self._seq = 0
        
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
                    print "    timeout:"
                    print "        %s, %d" % (self._client_list[k], self._client_timeout[k])
                #
                timeout = timeout + 1
                continue
            #
            if addr[0] in self._client_list:
                k = self._client_list.index(addr[0])
                self._client_timeout[k] = 1
            else: # address is not registered
                print "    bad address : %s, %d" % (addr[0], addr[1])
                error = error + 1
                continue
            #
            seq, ce = netutil.unpack_if(res)
            if seq==self._seq:
                ret =  ret + ce
            else: # bad seq
                print "   seq error : %s, %d, %d (%d)" % (addr[0], addr[1], seq, self._seq)
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
        return ret/float(self._client_num)
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
        return self.execute_cmd(mode, 0, 0, 0, 0)
        
    def set_alt(self, li, ni, ii, wi):
        mode = 30
        return self.execute_cmd(mode, li, ni, ii, wi)
        
    def update(self, li, ni, ii, wi):
        mode = 40
        return self.execute_cmd(mode, li, ni, ii, wi)
        
    def loop(self):
        print "ServerLooper::loop() - start"
        while not self.is_quite_requested():
            mode = self.mode()
            if mode==0:
                time.sleep(0.01)
                continue
            elif mode==10:  # init
                #
                cmd = netutil.pack_i6(self._seq, mode, 0, 0, 0, 0)
                self.send(cmd)
                self.set_mode(mode+5)
                #
                res, addr = self.recv()
                while res:
                    print "%s : %d" % (addr[0], addr[1])
                    self._client_list.append(addr[0])
                    res, addr = self.recv()
                #
                self._client_num = len(self._client_list)
                self._client_timeout = [0] * self._client_num
                print self._client_num
            elif mode==60: # debug
                cmd = netutil.pack_i6(self._seq, mode, 0, 0, 0, 0)
                self.send(cmd)
                self.set_mode(mode+5)
                ret = self.recv_multi()
                print ret
                self.set_mode(0)
            elif mode==20: # evaluate
                ce = self.evaluate()
                print "evaluate :%f" % (ce)
            elif mode==30: # alt
                cmd = netutil.pack_i6(self._seq, mode, 0, 1, 0, 0)
                self.send(cmd)
                self.set_mode(mode+5)
                #
                ret = self.recv_multi()
                print ret
                self.set_mode(0)
            elif mode==40: # update
                cmd = netutil.pack_i6(self._seq, mode, 0, 1, 0, 0)
                self.send(cmd)
                self.set_mode(mode+5)
                #
                ret = self.recv_multi()
                print ret
                self.set_mode(0)
            elif mode==50: # train
                it = 400
                debug = 0
                #main.echo(self._package_id)
                main.loop(it, r, self._package_id, debug)
                self.set_mode(0)
            else:
                self.set_mode(0)
            #
        #
        print "ServerLooper::loop() - end"
#
#
#
def server(SERVER_ADDR, SERVER_PORT, BC_ADDR, BC_PORT, package_id):
    print "main() : start"
    #
    s = ServerLooper(SERVER_ADDR, SERVER_PORT, BC_ADDR, BC_PORT, package_id)
    s.init()
    s.run()
    #
    loop = 1
    while loop:
        key = raw_input("cmd >")
        print key
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
        #
    #
    print "main() : end"
#
#
#
def main2():
    print "main() : start"
    #
    BC_ADDR = "127.0.0.1"
    BC_PORT = 5000
    SERVER_ADDR = "127.0.0.1"
    SERVER_PORT = 5005
    package_id = 0
    #
    s = server(SERVER_ADDR, SERVER_PORT, BC_ADDR, BC_PORT, package_id)
    #
    print "main() : end"
#
#
#
if __name__ == '__main__':
	main2()


