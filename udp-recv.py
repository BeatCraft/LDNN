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
def main():
    print "main() : start"
    #
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #sock.settimeout(1)
    msg = "test"
    ip = "127.0.0.1"
    port = 5000
    sock.bind((ip, port))
    #unpacker = struct.Struct('I 2s f')
    unpacker = struct.Struct('I I I I')

    while True:
        data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
        #print "received message:", data
        print data
        unpacked_data = unpacker.unpack(data)
        print unpacked_data
        print addr
        #print >>sys.stderr, 'rcv "%s"' % #binascii.hexlify(unpacked_data), values

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
