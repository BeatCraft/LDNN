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
import core
import util
import gpu
import netutil
#
#
#
class ServerLooper(netutil.Looper):
    def __init__(self, local_addr, local_port, remote_addr, remote_port):
        print "ServerLooper::__init__()"
        #
        super(ServerLooper, self).__init__(local_addr, local_port, remote_addr, remote_port)
        #
        #self._local_addr = local_addr
        #self._local_port = local_port
        #self._remoto_addr = remote_addr
        #self._remote_port = remote_port
        #self._send_sock = None
        #self._recv_sock = None
        self._send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._send_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._client_num = 0
        #
        package_id = 0 # MNIST
        my_gpu = None
        self._package = util.Package(package_id)
        self._roster = self._package.setup_dnn(my_gpu)
        if self._roster is None:
            print "fatal DNN error"
        #
    def set_client_num(self, num):
        self._client_num = num
        
    def recv_multi(self):
        ret = 0.0
        timeout = 0
        error = 0.0
        #
        for i in range(self._client_num):
            res, addr = self.recv()
            if res==None:
                timeout = timeout + 1
                continue
            #
            a, b = netutil.unpack_if(res)
            if a==self.mode():
                ret =  ret + b
            else:
                error = error +1
        #
        if timeout + error>0:
            return -1.0
        #
        return ret/float(self._client_num)
    
    
    def execute_cmd(self, mode, a, b, c, d):
        self.set_mode(mode)
        cmd = netutil.pack_i5(mode, a, b, c, d)
        self.send(cmd)
        #
        self.set_mode(mode+5)
        ret = self.recv_multi()
        #
        self.set_mode(mode+5)
        return ret
    
    def evaluate(self):
        mode = 20
        return self.execute_cmd(mode, 0, 0, 0, 0)
        
#        self.set_mode(mode)
#        cmd = pack_i5(mode, 0, 0, 0, 0)
#        self.send(cmd)
#        #
#        self.set_mode(mode+5)
#        ret = self.recv_multi()
#        #
#        self.set_mode(mode+5)
#        return ret
        
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
                cmd = netutil.pack_i5(mode, 0, 0, 0, 0)
                self.send(cmd)
                self.set_mode(mode+5)
                #
                ret = self.recv_multi()
                print ret
                self.set_mode(0)
            elif mode==20: # evaluate
                ce = self.evaluate()
                print "evaluate :%f" % (ce)
#                cmd = pack_i5(mode, 0, 1, 2, 0)
#                self.send(cmd)
#                self.set_mode(mode+5)
#                #
#                ret = self.recv_multi()
#                print ret
#                self.set_mode(0)
            elif mode==30: # alt
                cmd = netutil.pack_i5(mode, 0, 1, 0, 0)
                self.send(cmd)
                self.set_mode(mode+5)
                #
                ret = self.recv_multi()
                print ret
                self.set_mode(0)
            elif mode==40: # update
                cmd = netutil.pack_i5(mode, 0, 1, 0, 0)
                self.send(cmd)
                self.set_mode(mode+5)
                #
                ret = self.recv_multi()
                print ret
                self.set_mode(0)
            elif mode==50: # train
                self.train_loop()
#                r = self._roster
#                c = r.countLayers()
#                print c
                self.set_mode(0)
            else:
                self.set_mode(0)
            #
        #
        print "ServerLooper::loop() - end"


    def weight_shift_mode(self, li, ni, ii, mse_base, mode):
        #print "weight_shift_mode"
        r =  self._roster
        layer = r.getLayerAt(li)
        wp = layer.get_weight_property(ni, ii) # default : 0
        lock = layer.get_weight_lock(ni, ii)   # default : 0
        wi = layer.get_weight_index(ni, ii)
        #print wi
        wi_alt = wi
        #
        if lock>0:
            return mse_base, 0
        #
        if mode>0: # heat
            if wp<0:
                print "    skip"
                return mse_base, 0
            #
            if wi==core.WEIGHT_INDEX_MAX:
                layer.set_weight_property(ni, ii, 0)
                #
                layer.set_weight_index(ni, ii, wi-1)
                #
                #layer.update_weight_gpu()
                #r.propagate()
                #mse_base = r.get_cross_entropy()
                mse_base = self.update(li, ni, ii, wi-1)
                #
                #print "    max:%d" % (wi)
                return mse_base, 1
            #
        else:
            if wp>0:
                return mse_base, 0
            #
            if wi==core.WEIGHT_INDEX_MIN:
                layer.set_weight_property(ni, ii, 0)
                #
                layer.set_weight_index(ni, ii, wi+1)
                #
                #layer.update_weight_gpu()
                #r.propagate()
                #mse_base = r.get_cross_entropy()
                mse_base = self.update(li, ni, ii, wi+1)
                #
                return mse_base, 1
            #
        #
        wi_alt = wi + mode
        #
        #r.propagate(li, ni, ii, wi_alt, 0)
        #mse_alt = r.get_cross_entropy()
        mse_alt = self.set_alt(li, ni, ii, wi_alt)
        #print mse_alt
        #
        if mse_alt<mse_base:
            #print "    update"
            layer.set_weight_property(ni, ii, mode)
            layer.set_weight_index(ni, ii, wi_alt)
            #
            #layer.update_weight_gpu()
            #
            mse_alt = self.update(li, ni, ii, wi_alt)
            return mse_alt, 1
        #
        layer.set_weight_property(ni, ii, 0)
        #
        return mse_base, 0

    def train(self, it, limit):
        print "train"
        r = self._roster
        divider = 4
        t_cnt = 0
        h_cnt = 0
        c_cnt = 0
        w_list = []
        #
        # > > > >r.propagate()
        # > > > >mse_base = r.get_cross_entropy()
        mse_base = self.evaluate()
        print mse_base
        #
        c = r.countLayers()
        for li in range(1, c):
            layer = r.getLayerAt(li)
            num_node = layer._num_node
            num_w = layer._num_input
            #
            node_index_list = list(range(num_node))
            random.shuffle(node_index_list)
            nc = 0
            for ni in node_index_list:
                nc = nc + 1
                w_p = num_w/divider
                for p in range(w_p):
                    ii = random.randrange(num_w)
                    mse_base, ret = self.weight_shift_mode(li, ni, ii, mse_base, 1)
                    if ret>0:
                        print "[%d] H=%d/%d, N(%d/%d), W(%d/%d) : W(%d,%d,%d), CE:%f" % (it, h_cnt, t_cnt, nc, num_node, p, w_p, li, ni, ii, mse_base)
                    #print "    %f" % mse_base
                    #
                    h_cnt = h_cnt + ret
                    t_cnt = t_cnt +1
                    if mse_base<limit:
                        print "exit iterations"
                        return h_cnt, c_cnt, mse_base
                    #
                #
            #
        #
        t_cnt = 0
        c = r.countLayers()
        for li in range(1, c):
            layer = r.getLayerAt(li)
            num_node = layer._num_node
            num_w = layer._num_input
            #
            node_index_list = list(range(num_node))
            random.shuffle(node_index_list)
            nc = 0
            for ni in node_index_list:
                nc = nc + 1
                w_p = num_w/divider
                for p in range(w_p):
                    ii = random.randrange(num_w)
                    mse_base, ret = self.weight_shift_mode(li, ni, ii, mse_base, -1)
                    if ret>0:
                        print "[%d] C=%d/%d, N(%d/%d), W(%d/%d) : W(%d,%d,%d), CE:%f" % (it, c_cnt, t_cnt, nc, num_node, p, w_p, li, ni, ii, mse_base)
                    #print "    %f" % mse_base
                    #
                    c_cnt = c_cnt + ret
                    t_cnt = t_cnt + 1
                    if mse_base<limit:
                        print "exit iterations"
                        return h_cnt, c_cnt, mse_base
                    #
                #
            #
        #
        return h_cnt, c_cnt, mse_base
    #
    def train_loop(self, debug=0):
        print "train_loop"
        it = 20*20
        r = self._roster
        #
        h_cnt_list = []
        c_cnt_list = []
        ce_list = []
        #
        limit = 0.000001
        pre_ce = 0.0
        lim_cnt = 0
        #
        start_time = time.time()
        #
        for i in range(it):
            h_cnt, c_cnt, ce = self.train(i, limit)
            #
            h_cnt_list.append(h_cnt)
            c_cnt_list.append(c_cnt)
            ce_list.append(ce)
            # debug
#            if debug==1:
#                save_path = "./debug/wi.csv.%f" % ce
#                r.export_weight_index(save_path)
            #
            #r.export_weight_index(self._package._wi_csv_path)
            #
            if pre_ce == ce:
                lim_cnt = lim_cnt + 1
                if lim_cnt>5:
                    print "locked with local optimum"
                    print "exit iterations"
                    break
                #
            #
            if ce<limit:
                print "exit iterations"
                break
            #
            pre_ce = ce
        # for
        elapsed_time = time.time() - start_time
        t = format(elapsed_time, "0")
        print "time = %s" % (t)
        #
        k = len(h_cnt_list)
        for j in range(k):
            print "%d, %d, %d, %f," % (j, h_cnt_list[j], c_cnt_list[j], ce_list[j])
        #
    # end of train_loop()
#
#
#
def main():
    print "main() : start"
    #
    BC_ADDR = "192.168.0.255"
    BC_PORT = 5000
    SERVER_ADDR = "192.168.0.152"
    SERVER_PORT = 5005
    #
    s = ServerLooper(SERVER_ADDR, SERVER_PORT, BC_ADDR, BC_PORT)
    s.set_client_num(3)
    s.init()
    s.run()
    
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
        #
    #
    print "main() : end"
#
#
#
if __name__ == '__main__':
	main()


