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
    sock.settimeout(1)
    #self._sock.bind(("", self._local_port))
    msg = "test"
    ip = "127.0.0.1"
    port = 5000
    #sock.sendto(msg, (ip, port))
    
    
    values = (1, 'ab', 2.7)
    packer = struct.Struct('I 2s f')
    packed_data = packer.pack(*values)
    sock.sendto(packed_data, (ip, port))
    
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
    
    cmd = struct.Struct('I I f f')
    cmd_init = (0, 0)
    cmd_ack = ()
    
    

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


