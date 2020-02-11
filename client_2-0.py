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
    # amd-0 : 192.168.0.150 / 192.168.200.10
    # amd-1 : 192.168.0.151 / 192.168.200.11
    # amd-2 : 192.168.0.152 / 192.168.200.12
    # amd-3 : 192.168.0.153 / 192.168.200.13
    BC_ADDR = "192.168.200.255"
    BC_PORT = 5000
    SERVER_ADDR = "192.168.200.10"
    SERVER_PORT = 5005
    batch_size = 15000
    # 0 : 0
    # 1 : 15000
    # 2 : 30000
    # 3 : 45000
    batch_start = 30000
    # 0 : AMD Server"
    # 1 : Intel on MBP"
    # 2 : eGPU (AMD Radeon Pro 580)"
    device_id = 0
    # 0 : MNIST
    package_id = 0
    #
    client.client(BC_ADDR, BC_PORT, SERVER_ADDR, SERVER_PORT, batch_size, batch_start, device_id, package_id)
#
#
#
if __name__ == '__main__':
	main()


