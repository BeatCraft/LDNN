#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import traceback
import csv
import socket
import time
import command
import multiprocessing
import struct
import binascii
#
import numpy as np
import sys,os
import core
import util
import gpu
#
#
#
def cmd_broadcast(sock, addr, port, data):
    sock.sendto(data, (addr, port))

def cmd_recv(sock):
    msg, address = sock.recvfrom(1024)
    #print("message: {msg}\nfrom: {address}")
    return msg

def cmd_pack(a=0, b=0, c=0, d=0):
    values = (a, b, c, d)
    packer = struct.Struct('I I I I')
    data = packer.pack(*values)
    return data
    
def cmd_unpack_i4(data):
    unpacker = struct.Struct('I I I I')
    unpacked_data = unpacker.unpack(data)
    return unpacked_data

def cmd_unpack(data):
    unpacker = struct.Struct('I I f')
    unpacked_data = unpacker.unpack(data)
    return unpacked_data

def main():
    print "main() : start"
    #
    BC_ADDR = "127.0.0.1"
    BC_PORT = 5000
    RECV_ADDR = "127.0.0.1"
    RECV_PORT = 5005
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind((RECV_ADDR, RECV_PORT))
    #
    state = 0
    while True:
        if state == 0: # init
            cmd = cmd_pack()
            cmd_broadcast(send_sock, BC_ADDR, BC_PORT, cmd)
        elif state == 1:
            pass
        elif state == 10: #
            pass
        elif state == 20:
            pass
    
        res = cmd_recv(recv_sock)
        print cmd_unpack_i4(res)
    
    
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
	# -r remote host
	# -p send port
	# -s source port
	# (ex) python Node.py -r 127.0.0.1 -p 8000 -s 9000
	# python Node.py -r 192.0.0.127 -p 8000 -s 9000
	main()


