import pyopencl as cl
import os, sys, time
from time import time
import numpy as np
#
#
#
KERNEL_CODE = """
__kernel void scale(
    __global float* x,
    __global float* y,
    const int stride,
    const float max,
    const int debug)
{
    int i = get_global_id(0); // data index
    int j = get_global_id(1); // bathch index
    
    y[stride*j+i] = x[stride*j+i]/max;
    
    if (debug==1){
        printf(\"%d, %d\\n\",i, j);
    }
};

__kernel void multiple_x_by_w(
    __global float* x,
    __global float* w,
    __global float* y,
    const int bi,
    const int stride_1,
    const int stride_2)
{
    int i = get_global_id(0); // num_input
    int j = get_global_id(1); // num_node
    
    y[stride_1*bi + stride_2*j+i] = x[stride_2*bi+i] * w[stride_2*j+i];
    
//    if (j==9){
//        printf(\"gpu(%f)(%f) [%d]\\n\", x[stride_2*bi + i], w[stride_2*j+i], i);
//    }
};

__kernel void multiple_x_by_w_alt(
    __global float* x,
    __global float* w,
    __global float* y,
    const int bi,
    const int stride_1,
    const int stride_2,
    const int alt_ni,
    const int alt_ii,
    const float alt_w)
{
    int i = get_global_id(0); // num_input
    int j = get_global_id(1); // num_node

    if (j==alt_ni && i==alt_ii){
        y[stride_1*bi + stride_2*j + i] = x[stride_2*bi + i] * alt_w;
    }else{
        y[stride_1*bi + stride_2*j + i] = x[stride_2*bi + i] * w[stride_2*j + i];
    }
};

"""
#
#
#
class Gpu:
    def __init__(self):
        #
        #self._ctx = cl.create_some_context()
        #
        platform = cl.get_platforms()[0]
        # AMD Server
        #device = platform.get_devices()[0]
        # Intel on MBP
        #device = platform.get_devices()[1]
        # AMD on eGPU
        device = platform.get_devices()[2]
        print platform
        print device
    
        self._ctx = cl.Context([device])
        for dev in self._ctx.devices:
            assert dev.local_mem_size > 0
        self._queue = cl.CommandQueue(self._ctx)
        self._bufs = []

    def set_kernel_code(self):
        self.prg = cl.Program(self._ctx, KERNEL_CODE).build()
    
    def get_buffer_list(self):
        return self._bufs
    
    def dev_malloc(self, host_array):
        mf = cl.mem_flags
        buf = cl.Buffer(self._ctx,
                        mf.READ_WRITE|mf.COPY_HOST_PTR,
                        hostbuf=host_array,
                        size=host_array.nbytes)
        self._bufs.append(buf)
        return buf

    def scale(self, d_x, d_y, stride, max, row, batch_size, debug):
        event = self.prg.scale(self._queue, (row, batch_size), None,
                               d_x, d_y, np.int32(stride),
                               np.float32(max), np.int32(debug))
        event.wait()

    def multiple_x_by_w(self, d_x, d_w, d_y, bi, stride_1, stride_2, row, col):
        event = self.prg.multiple_x_by_w(self._queue,(row,col), None,
                                         d_x, d_w, d_y, np.int32(bi),
                                         np.int32(stride_1), np.int32(stride_2))
        event.wait()

    def multiple_x_by_w_alt(self, d_x, d_w, d_y, bi, stride_1, stride_2, row, col, ni, ii, wv):
        event = self.prg.multiple_x_by_w_alt(self._queue,(row,col), None,
                                             d_x, d_w, d_y, np.int32(bi),
                                             np.int32(stride_1), np.int32(stride_2),
                                             np.int32(ni), np.int32(ii), np.float32(wv))
        event.wait()
    
    def copy(self, dist, src):
        event = cl.enqueue_copy(self._queue, dist, src)
        event.wait()
#
#
#
def main():
    data_x = np.array([0.1, 0.2, 0.3, 0.4]).astype(np.float32)
    data_w = np.array([[0.5, 0.5, 0.5, 0.5],
                       [0.1, 0.1, 0.1, 0.1]]).astype(np.float32)
    data_y = np.array([[0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0]]).astype(np.float32)
    num = data_x.shape[0]
    print num
    
    data_a = np.array([8, 16, 32, 64]).astype(np.int32)
    data_b = np.array([0.0, 0.0, 0.0, 0.0]).astype(np.float32)

    print data_x
    print data_w
    print data_y
    
    g = Gpu()
    g.dev_malloc(data_x) # 0
    g.dev_malloc(data_w) # 1
    g.dev_malloc(data_y) # 2
    g.dev_malloc(data_a) # 3
    g.dev_malloc(data_b) # 4
    
    g.set_kernel_code()
    bufs = g.get_buffer_list()
    
    g.multiple_x_by_w(bufs[0], bufs[1], bufs[2], num)
    g.read(data_y, bufs[2])
    print data_y
    #data_y[0][0]=0.999
    #print data_y
    
    
    g.scale(bufs[3], bufs[4], 255.0, 4)
    g.read(data_x, bufs[4])
    print data_x
    
    #g.write(bufs[2], data_a)
    
    #for row in data_y:
    #    print np.sum(row)
    
    return 0

if __name__=='__main__':
    print ">> start"
    sts = main()
    print ">> end"
    print("\007")
    sys.exit(sts)

