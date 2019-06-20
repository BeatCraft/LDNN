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
#sys.path.append('../ldnn/')
import core
import util
import gpu

class Looper(object):
    def __init__(self):
        print "Looper::__init__()"
        self._stop = multiprocessing.Value('i', 0)
        self._process = None
        #
        self.init()

    def init(self):
        print "Looper::init()"

    def loop(self):
        print "Looper::loop()"
        while not self._stop.value:
            print "  loop"
            print self._stop.value

        print "  stopped."
    
    def run(self):
        print "Looper::run()"
        self._process = multiprocessing.Process(target=self.loop, args=())
        self._process.start()

    def quit(self):
        print "Looper::quit()"
        self._stop.value = 1;

class SocketLooper(Looper):
    def __init__(self, local_port):
        self._local_port = local_port
        self._sock = None
        super(SocketLooper, self).__init__()

    def init(self):
        print "SocketLooper::init()"
#        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#        self._sock.settimeout(1)
#        self._sock.bind(("", self._local_port))

class ClientLooper(SocketLooper):
    def __init__(self, roster, remote_address, remote_port, local_port):
        self._remote_address = remote_address
        self._remote_port = remote_port
        self._local_port = local_port
        self._roster = roster
        super(ClientLooper, self).__init__(local_port)

    def loop(self):
        print "ClientLooper::loop() start"
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
                    self.command(data, address)
            except:
                traceback.print_exc()
                    
        self._sock.close()
        self._sock = None
        print "ClientLooper::loop() end"


    def command(self, msg, address):
        print "ClientLooper::command()"
        cmd = command.decode_command(msg)
        print "  type : ", cmd.cmd_type
        # evaluate
        if cmd.cmd_type == command.CMD_TYPE_EVAL_WEIGHT:
            pass
            #self.evaluate(cmd)
        # change weight
        elif cmd.cmd_type == command.CMD_TYPE_CHANGE_WEIGHT:
            pass
#            weight_list = []
#            for data in cmd.weight_list:
#            weight_list.append((data.index, data.value))
            # 結果を送信
#            self.changeWeight(weight_list)
#            result = command.ChangeWeightResult(command.RES_OK)
#            self.send(result.encode_bin())
        elif cmd.cmd_type == command.CMD_TYPE_NODE_QUIT:
            self.quit()
        else:
            print "  command not found :", cmd.cmd_type
#
#ServerLooper
#
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


def batch_evaluate(r, batch, batch_start, batch_size, labels):
    sum = 0.0
    for i in range(batch_start, batch_start+batch_size, 1):
        entry = batch[i]
        data_class = entry[1]
        labels[data_class] = 1.0
        inf = r._batch_out[i]
        mse =  util.cross_emtropy_error(inf, len(inf), labels, len(labels))
        sum = sum + mse
        labels[data_class] = 0.0
    
    return sum/float(batch_size)

def main():
    args = get_args()
    #
    my_gpu = gpu.Gpu()
    my_gpu.set_kernel_code()
    r = core.Roster()
    r.set_gpu(my_gpu)
    #
    input_layer = r.add_layer(0, 784, 784)
    hidden_layer_1 = r.add_layer(1, 784, 32)
    hidden_layer_2 = r.add_layer(1, 32, 32)
    output_layer = r.add_layer(2, 32, 10)
    #
    wi = util.csv_to_list("./client_data/wi.csv")
    if len(wi)>0:
        print "restore weights"
        r.restore_weighgt(wi)
    else:
        print "init weights"
        r.init_weight()
    
    if my_gpu:
        r.update_weight()

    batch = util.pickle_load("./client_data/train_batch.pickle")
    batch_size = 100
    batch_start = 0
    r.makeBatchBufferIn(28*28, batch_size)
    r.makeBatchBufferOut(10, batch_size)
    labels = np.zeros(10, dtype=np.float32)
    #
    for i in range(batch_start, batch_start+batch_size, 1):
        entry = batch[i]
        r._batch_in[i] = entry[0].copy()

    r._gpu.copy(r._gpu_batch_in, r._batch_in)
    #
    r.batch_propagate_all(None, None, 0)
    mse_base = batch_evaluate(r, batch, batch_start, batch_size, labels)
    print mse_base

    client = ClientLooper(r, '127.0.0.1', 8000, 9000)
    client.run()

    try:
        c = input("test")
    except:
        c = -1
    
    client.quit()



    #test = SocketLooper(5000)

#	node = LNode(args.remote_host, args.port, args.sorce,args.weght)
#	print node.run()


if __name__ == '__main__':
	# -r remote host
	# -p send port
	# -s source port
	# (ex) python Node.py -r 127.0.0.1 -p 8000 -s 9000
	# python Node.py -r 192.0.0.127 -p 8000 -s 9000
	main()	
