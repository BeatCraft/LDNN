import pyopencl as cl
import os, sys, time
from time import time
import numpy as np
#
#
#
KERNEL_CODE = """

__kernel void multiple_x_by_w(
    __global float* x,
    __global float* w,
    __global float* y,
    const int num)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    y[j*num + i] = x[i] * w[j*num + i];
};

__kernel void scale(
    __global int* x,
    __global float* y,
    const float max)
{
    int i = get_global_id(0);

    y[i] = x[i] / max;    
};

"""
#
# printf(\"%f\\n\", y[i]);
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
    
    def multiple_x_by_w(self, d_x, d_w, d_y, num):
        event = self.prg.multiple_x_by_w(self._queue,(4,2), None,
                                         d_x, d_w, d_y, np.int32(num))
        event.wait()

    def scale(self, d_x, d_y, max, row):
        event = self.prg.scale(self._queue, (row,), None,
                               d_x, d_y, np.float32(max))
        event.wait()
    
    
    def read(self, host_array, dev_buf):
        cl.enqueue_copy(self._queue, host_array, dev_buf) # dist, src

    def write(self, dev_buf, host_array):
        cl.enqueue_copy(self._queue, dev_buf, host_array)

#
#        host_data = np.array([0.0, 0.0, 0.0, 0.0]).astype(np.float32)
#        import pyopencl.enqueue_read_buffer
#        event = enqueue_read_buffer(self._queue,
#                                       dev_buf,
#                                       host_data,
#                                       dev_offset=1)
#        print host_data
#        event.wait()


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

