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
#
import util
#
#
def broadcast(sock, addr, port, data):
    sock.sendto(data, (addr, port))

def recv(sock):
    msg, address = sock.recvfrom(1024)
    return msg, address

def pack_iiii(a=0, b=0, c=0, d=0):
    values = (a, b, c, d)
    packer = struct.Struct('I I I I')
    data = packer.pack(*values)
    return data
    
def unpack_iiii(data):
    unpacker = struct.Struct('I I I I')
    unpacked_data = unpacker.unpack(data)
    return unpacked_data

def pack_i5(a=0, b=0, c=0, d=0, e=0):
    values = (a, b, c, d, e)
    packer = struct.Struct('I I I I I')
    data = packer.pack(*values)
    return data
    
def unpack_i5(data):
    unpacker = struct.Struct('I I I I I')
    unpacked_data = unpacker.unpack(data)
    return unpacked_data

def pack_if(a=0, b=0.0):
    values = (a, b)
    packer = struct.Struct('I f')
    data = packer.pack(*values)
    return data
    
def unpack_if(data):
    unpacker = struct.Struct('I f')
    unpacked_data = unpacker.unpack(data)
    return unpacked_data

def pack_iif(a=0, b=0, c=0.0):
    values = (a, b, c)
    packer = struct.Struct('I I f')
    data = packer.pack(*values)
    return data
    
def unpack_iif(data):
    unpacker = struct.Struct('I I f')
    unpacked_data = unpacker.unpack(data)
    return unpacked_data
#
#
#
class Looper(object):
    def __init__(self, local_addr, local_port, remote_addr, remote_port):
        print "Looper::__init__()"
        self._stop = multiprocessing.Value('i', 0)
        self._mode = multiprocessing.Value('i', 0)
        self._process = None
        self._local_addr = local_addr
        self._local_port = local_port
        self._remoto_addr = remote_addr
        self._remote_port = remote_port
        self._send_sock = None
        self._recv_sock = None
        #
        self.init()

    def init(self):
        print "Looper::init()"
        #self._send_sock = None#socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        #self._send_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._recv_sock.settimeout(1)
        self._recv_sock.bind((self._local_addr, self._local_port))
    
    def send(self, cmd):
        self._send_sock.sendto(cmd, (self._remoto_addr, self._remote_port))
    
    def recv(self):
        try:
            return recv(self._recv_sock)
        except:
            return None, None
        
    def loop(self): # MUST override this
        print "Looper::loop()"
        while not self._stop.value:
            print "  loop"
            print self._stop.value
        #
        print "  stopped."

    def run(self):
        print "Looper::run()"
        self._process = multiprocessing.Process(target=self.loop, args=())
        self._process.start()

    def quit(self):
        print "Looper::quit()"
        self._stop.value = 1
    
    def is_quite_requested(self):
        return self._stop.value
        
    def set_mode(self, mode):
        self._mode.value = mode
        
    def mode(self):
        return self._mode.value
#
#
#
def main():
    print "main() : start"
    #
    BC_ADDR = "127.0.0.1"
    BC_PORT = 5000
    RECV_ADDR = "127.0.0.1"
    RECV_PORT = 5005
    
    s = ServerLooper(RECV_ADDR, RECV_PORT, BC_ADDR, BC_PORT)
    s.init()
    s.run()
    
    loop = 1
    while loop:
        key = raw_input("cmd >")
        print key
        if key=='q' or key=='Q':
            loop = 0
            s.quit()
    
#    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #
#    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#    recv_sock.bind((RECV_ADDR, RECV_PORT))
    #
#    state = 0
#    while True:
#        if state == 0: # init
#            cmd = pack_iiii()
#            broadcast(send_sock, BC_ADDR, BC_PORT, cmd)
#        elif state == 1:
#            pass
#        elif state == 10: #
#            pass
#        elif state == 20:
#            pass
#
#        res, addr = recv(recv_sock)
#        print addr
#        print unpack_iiii(res)
#
    
#    values = (1, 'ab', 2.7)
#    packer = struct.Struct('I 2s f')
#    packed_data = packer.pack(*values)
#
    # 0x : init
    #   00 : init : 0
    #   01 : ack : 1 c_id
    # 1x : Map WI (TBD)
    #   10 : map
    #   11 : OK
    # 2x : train
    #   20 : set alt : 20 li ni wi
    #   21 : return cr : 21 c_id cr
    #   22 : update : 22 li ni wi
    #   23 : OK
    
    #cmd = struct.Struct('I I f f')
    #cmd_init = (0, 0)
    #cmd_ack = ()
    

    print "main() : end"
#
#
#
if __name__ == '__main__':
	main()
#
