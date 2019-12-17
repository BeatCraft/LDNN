#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import traceback
import csv
import socket
import time
import command
import multiprocessing
#
import numpy as np
import sys,os
import core
import util
import gpu

#class Looper(object):
#    def __init__(self):
#        print "Looper::__init__()"
#        self._stop = multiprocessing.Value('i', 0)
#        self._process = None
#        #
#        self.init()
#
#    def init(self):
#        print "Looper::init()"
#
#    def loop(self):
#        print "Looper::loop()"
#        while not self._stop.value:
#            print "  loop"
#            print self._stop.value
#
#        print "  stopped."
#
#    def run(self):
#        print "Looper::run()"
#        self._process = multiprocessing.Process(target=self.loop, args=())
#        self._process.start()
#
#    def quit(self):
#        print "Looper::quit()"
#        self._stop.value = 1;
#


#class SocketLooper(Looper):
#    def __init__(self, local_port):
#        self._local_port = local_port
#        self._sock = None
#        super(SocketLooper, self).__init__()
#
#    def init(self):
#        print "SocketLooper::init()"
#        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#        self._sock.settimeout(1)
#        self._sock.bind(("", self._local_port))

#
#
#
class ClientLooper():
    def __init__(self, remote_address, remote_port, local_port):
        self._remote_address = remote_address
        self._remote_port = remote_port
        self._local_port = local_port
        #
        self._local_port = local_port
        self._sock = None
        #
        self._stop = multiprocessing.Value('i', 0)
        self._process = None
    
    def run(self):
        self._process = multiprocessing.Process(target=self.loop, args=())
        self._process.start()
        
    def quit(self):
        print "    quit()"
        self._stop.value = 1;
        
    def command(self, msg, address):
        print "    command()"
        cmd = command.decode_command(msg)
        print "  type : ", cmd.cmd_type
        # evaluate
        if cmd.cmd_type == command.CMD_TYPE_EVAL_WEIGHT:
            pass
        elif cmd.cmd_type == command.CMD_TYPE_CHANGE_WEIGHT:
            pass
        elif cmd.cmd_type == command.CMD_TYPE_NODE_QUIT:
            self.quit()
        else:
            print "  command not found :", cmd.cmd_type
                
    def loop(self):
        print "loop() : start"
        #
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(1)
        self._sock.bind(("", self._local_port))
        #
        while not self._stop.value:
            data = None
            address = None
            try:
                data, address = self._sock.recvfrom(8192)
            except:
                # time out
                pass
            #
            try:
                if data:
                    pass
                    #self.command(data, address)
            except:
                traceback.print_exc()
        # while
        self._sock.close()
        self._sock = None
        print "loop() : end"
#
#
#
def get_args():
	DEFAULT_HOST = '127.0.0.1'
	DEFAULT_R_PORT = 8000
	DEFAULT_S_PORT = 9000
	DEFALUT_WEIGHT_PATH = "./weight.data"

	parser = argparse.ArgumentParser()
	parser.add_argument("-r", "--remote_host", type=str, default=DEFAULT_HOST)
	parser.add_argument("-p", "--port", type=int, default=DEFAULT_R_PORT)
	parser.add_argument("-s", "--sorce", type=int, default=DEFAULT_S_PORT)
	parser.add_argument("-w", "--weght", type=str, default=DEFALUT_WEIGHT_PATH)

	args = parser.parse_args()
	return args
#
#
#
def main():
    args = get_args()
    #
    client = ClientLooper('127.0.0.1', 8000, 9000)
    client.run()
    #
    try:
        c = input("test")
    except:
        c = -1
        
    key = util.get_key_input("test> ")
    
    client.quit()
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
