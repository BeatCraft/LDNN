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
    port = 5005
    #sock.sendto(msg, (ip, port))
    
    values = (1, 'ab', 2.7)
    packer = struct.Struct('I 2s f')
    packed_data = packer.pack(*values)
    sock.sendto(packed_data, (ip, port))

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


