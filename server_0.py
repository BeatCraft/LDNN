#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
import random
#
import core
import util
import gpu
import netutil
import server
#
#
#
def main():
    print("main() : start")
    #
    BC_ADDR = "192.168.200.255"
    BC_PORT = 5000
    SERVER_ADDR = "192.168.200.10"
    SERVER_PORT = 5005
    #
    package_id = 0
    config_id = 1
    mini_batch_size = 625
    num_client = 4
    epoc = 16
    #
    server.server(SERVER_ADDR, SERVER_PORT, BC_ADDR, BC_PORT, package_id, config_id, mini_batch_size, num_client, epoc)
    #
    print("main() : end")
#
#
#
if __name__ == '__main__':
	main()


