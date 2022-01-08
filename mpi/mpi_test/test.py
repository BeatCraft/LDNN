import os, sys, time, math
from stat import *
import copy
import math
import numpy as np
import pickle



HOST_NAMES = ["tr", "amd-1", "amd-2", "amd-3"]
MINI_BATCH_SIZE = [1000, 250, 250, 250]

 
package_id = 0 
config_id = 0
mini_batch_size = MINI_BATCH_SIZE 
b_start = 0

def get_host_index_by_name(name):
    return HOST_NAMES.index(name)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"comm : rank={rank}, size={size}")
    my_self = MPI.COMM_SELF
    my_size = my_self.group.Get_size() 
    my_rank = my_self.group.Get_rank()
    print(f"self : rank={my_rank}, size={my_size}")

    pname = MPI.Get_processor_name()
    print(f"Hello, world! from {pname}")

    i = get_host_index_by_name(pname)
    print(f"index={i}") 


    if rank == 0: # server
        #print(f"server({rank})={pname}")
        print("server : %s, %d, %d" % (pname, rank, i))
        comm.send((10, "test"), dest=1, tag=1)
        if i==0:
            pass
            #comm.send((rank, size, pname, my_rank, my_size), dest=0, tag=1)
        else:
            pass
        #
    else: # client
        print("client : %s, %d, %d" % (pname, rank, i))
        k, t = comm.recv(source=0, tag=1)
        print("%d : %s" % (k, t))
        if i==0:
            pass
        else:
            pass
            #world_rank, world_size, your_name, your_rank, your_size = comm.recv(source=i, tag=1)
            #print("%d, %d, %d, %d,%d" % (world_rank, world_size, your_name, your_rank, your_size))
        #
    #
#
#
#
if __name__=='__main__':
    sts = main()
    sys.exit(sts)
#
