#! /usr/bin/python
# -*- coding: utf-8 -*-
#
import os, sys, time, math
import core


def main():
    argvs = sys.argv
    argc = len(argvs)
    print(argc)
    #

    import plat
    if plat.ID==0: # MBP
        import opencl
        platform_id = 0
        device_id = 1
        my_gpu = opencl.OpenCL(platform_id, device_id)
        my_gpu.set_kernel_code()
    else:
        return 0
    #

    r = Roster()
    r.set_gpu(my_gpu)
    #
    c = r.count_layers()
    input = InputLayer(c, 128, 128, None, my_gpu)
    r.layers.append(input)
    
    c = r.count_layers()
    hidden_1 = HiddenLayer(c, 128, 64, input, my_gpu)
    r.layers.append(hidden_1)
    
    c = r.count_layers()
    hidden_2 = HiddenLayer(c, 64, 64, hidden_1, my_gpu)
    r.layers.append(hidden_2)

    c = r.count_layers()
    output = OutputLayer(c, 64, 10, hidden_2, my_gpu)
    r.layers.append(output)

    r.init_weight()
    #
    return 0
#
#
#
if __name__=='__main__':
    print(">> start")
    sts = main()
    print(">> end")
    print("\007")
    sys.exit(sts)
#
#
#
