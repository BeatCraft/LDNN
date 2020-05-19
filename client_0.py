#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
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
import client
#
#
#
def main():
    cid = 0
    num = 4
    batch_size = 20000
    print "main() : start : %d" % (cid)
    # amd-0 : 192.168.0.150 / 192.168.200.10
    # amd-1 : 192.168.0.151 / 192.168.200.11
    # amd-2 : 192.168.0.152 / 192.168.200.12
    # amd-3 : 192.168.0.153 / 192.168.200.13
    BC_ADDR = "192.168.200.255"
    BC_PORT = 5000
    SERVER_ADDR = "192.168.200.10"
    SERVER_PORT = 5005
    #
    batch_size = 20000
    part_size = batch_size/num
    part_start = part_size*cid
    platform_id = 0
    device_id = 0 # AMD Server"
    package_id = 0 # MNIST
    #
    client.client(BC_ADDR, BC_PORT, SERVER_ADDR, SERVER_PORT,
                  batch_size, part_start, part_size,
                  platform_id, device_id, package_id)
    #
#
#
#
if __name__ == '__main__':
	main()


