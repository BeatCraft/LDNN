#! c:/Python26/python.exe
# -*- coding: utf-8 -*-

import os
import sys

ID = 0

# 0 : MacBook Pro (13-inch, 2017, Two Thunderbolt 3 ports)
#     macOS Monterey Version 12.3.1
#     opencl
# 1 : Threadripper
#     ubuntu
#     opencl
# 2 : Nvidia DGX
#     ubuntu
#     cupy

if sys.platform.startswith('darwin'):
    import opencl
else:
    if ID==1:
        import opencl
    elif ID==2:
        import dgx
    #
#

def getGpu(idx=0):
    if ID==0: # MBP
        platform_id = 0
        device_id = 1
        my_gpu = opencl.OpenCL(platform_id, device_id)
        my_gpu.set_kernel_code()
    elif ID==1: # tr
        platform_id = 1
        device_id = 0
        my_gpu = opencl.OpenCL(platform_id, device_id)
        my_gpu.set_kernel_code()
    elif ID==2: # nvidia
        my_gpu = dgx.Dgx(idx)
    else:
        print("error : undefined platform")
        my_gpu = None
    #
    return my_gpu

