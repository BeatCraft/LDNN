import pyopencl as cl
from pyopencl.reduction import ReductionKernel

import os, sys, time
from time import time
import numpy as np

block_size = 1

KERNEL_CODE = """

__kernel void multiple_x_by_w(
    __global float* x,
    __global float* w,
    __global float* y,
    const int num_x,
    const int num_y)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

//    float k = 0.0;
//    k = x[i] * w[j*num_x+i];
//    printf(\"(%d, %d) k = %f\\n\", i, j, k);
//    k = 0.0;
    for (int h=0; h<4; h++){
        y[j] += x[i+h] * w[j*num_x+i+h];
    }
//    printf(\"%f\\n\", y[j]);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    printf(\"%f\\n\", y[0]);
    printf(\"%f\\n\", y[1]);
}

"""

#y[j*num_x+i] = x[i] * w[j*num_x+i];

#    int dim = get_work_dim();
#printf(\"get_work_dim= %d\\n\", dim);
#    //int gid = get_global_id(0);
#//y[gid] = 0.2;
#//y[0] = 0.3;
#
#
#
class Gpu:
    def __init__(self):
        #self._ctx = cl.create_some_context()
        platform = cl.get_platforms()[0]
        device = platform.get_devices()[2]
        self._ctx = cl.Context([device])
        
        for dev in self._ctx.devices:
            assert dev.local_mem_size > 0

        self._queue = cl.CommandQueue(self._ctx)
        self._bufs = []

        self.test = ReductionKernel(self._ctx,
                                    np.float32,
                                    neutral='0',
                                    reduce_expr='a+b',
                                    map_expr='x[i]',
                                    arguments='__global float* x')


            #from pyopencl import clrandom
            #a = clrandom.rand(self._queue, 10, np.float32)
            #print a
            #print self.test(a)


    def set_kernel_code(self):
        self.prg = cl.Program(self._ctx, KERNEL_CODE).build()
#
#                            map_expr="i",
#                              arguments=[])

                              #
                              #
            


#result_dev = red(range=slice(n), queue=queue).get()
#result_ref = n*(n-1)//2
    
    
    def get_buffer_list(self):
        return self._bufs
    
    def dev_malloc(self, host_array):
        mf = cl.mem_flags
        buf = cl.Buffer(self._ctx, mf.READ_WRITE|mf.COPY_HOST_PTR, hostbuf=host_array, size=host_array.nbytes)
        self._bufs.append(buf)
    
    def multiple_x_by_w(self, d_x, d_w, d_y, num_x, num_y):
        test_data = np.array([0.1, 0.2, 0.3, 0.4]).astype(np.float32)
        #print self._sum(test_data, queue=self._queue).get()
        
        #        from pyopencl.reduction import ReductionKernel
        #sum = ReductionKernel(
        #                      self._ctx,
        #                      np.float32,
        #                      neutral="0",
        #                      reduce_expr="a+b",
        #                      arguments='__global float* x')
        #print sum(test_data)

                                  
        
        
        
        event = self.prg.multiple_x_by_w(
                                         self._queue,(4,2), None,
                                         d_x, d_w, d_y,
                                         np.int32(num_x), np.int32(num_y))
        event.wait()
    
    
    
#
#        from pyopencl.reduction import ReductionKernel
#        k = ReductionKernel(self._ctx,
#                            np.float32,
#                            neutral='0',
#                            reduce_expr='a+b',
#                            map_expr='x[i]',
#                            arguments='__global float* x')
#
#        print k(d_y)
    
    def read(self, dev_buf, host_array):
        cl.enqueue_copy(self._queue, host_array, dev_buf)
#
#
#
def main():
    data_x = np.array([0.1, 0.2, 0.3, 0.4]).astype(np.float32)
    data_w = np.array(
                      [[0.5, 0.5, 0.5, 0.5],
                       [0.1, 0.1, 0.1, 0.1]]).astype(np.float32)
    #data_y = np.array([[0.0, 0.0, 0.0, 0.0],
    #                   [0.0, 0.0, 0.0, 0.0]]).astype(np.float32)
    data_y = np.array([0.0, 0.0]).astype(np.float32)
    #data_y_sum = np.array([0.0, 0.0, 0., 0.0]).astype(np.float32)
                       #data_y_sum = np.array([0.0, 0.0]).astype(np.float32)
    print data_x
    print data_w
    print data_y
    #print data_y_sum
    
    g = Gpu()
    g.dev_malloc(data_x)
    g.dev_malloc(data_w)
    g.dev_malloc(data_y)
    #g.dev_malloc(data_y_sum)
    #g.set_kernel_code()
    bufs = g.get_buffer_list()

    #   g.multiple_x_by_w(bufs[0], bufs[1], bufs[2], 4, 2)
    #g.read(bufs[2], data_y)

    from pyopencl import array as clarray
    a = clarray.to_device(g._queue, data_x)

    print g.test(a)


    
    return 0

if __name__=='__main__':
    print ">> start"
    sts = main()
    print ">> end"
    print("\007")
    sys.exit(sts)

