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
    print "main() : start"
    #
    BC_ADDR = "192.168.200.255"
    BC_PORT = 5000
    SERVER_ADDR = "192.168.200.10"
    SERVER_PORT = 5005
    # 0 : 192.168.0.150 / 192.168.200.10
    batch_size = 8000
    batch_start = 0
    device_id = 0
    # 1-0 : 192.168.0.150 / 192.168.200.11
#   batch_size = 6000
#   batch_start = 8000
#   device_id = 0
    # 1-1 : 192.168.0.151 / 192.168.200.11
#   batch_size = 8000
#   batch_start = 14000
#   device_id = 1
    # 2-0 : 192.168.0.152 / 192.168.200.12
#   batch_size = 20000
#   batch_start = 22000
#   device_id = 0
    # 0 : AMD Server"
    # 1 : Intel on MBP"
    # 2 : eGPU (AMD Radeon Pro 580)"
    package_id = 0 # MNIST
    #
    client.client(BC_ADDR, BC_PORT, SERVER_ADDR, SERVER_PORT, batch_size, batch_start, device_id, package_id)
#
#
#
if __name__ == '__main__':
	main()


