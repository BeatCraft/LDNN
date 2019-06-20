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
    const float max,
    const int debug)
{
    int i = get_global_id(0);
    y[i] = x[i]/max;
    if (debug==1){
        printf(\"[%d] %f, %f\\n\",i,  x[i], y[i]);
    }
};

__kernel void batch_scale(
    __global float* x,
    __global float* y,
    const int index,
    const int stride,
    const float max,
    const int debug)
{
    int i = get_global_id(0);
    y[stride*index+i] = x[stride*index+i]/max;
    if (debug==1){
        printf(\"[%d] %f, %f\\n\",i,  x[stride*index+i], y[i]);
    }
};

__kernel void batch_multiple_x_by_w(
    __global float* x,
    __global float* w,
    __global float* y,
    const int num_w,
    const int index,
    const int stride)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    y[stride*index+j*num_w + i] = x[i] * w[j*num_w + i];
};

__kernel void multiple_x_by_w(
__global float* x,
__global float* w,
__global float* y,
const int num_w)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    y[j*num_w + i] = x[i] * w[j*num_w + i];
};

__kernel void multiple_x_by_w_alt(
__global float* x,
__global float* w,
__global float* y,
const int num_w,
const int alt_row,
const int alt_col,
const float alt_w)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    if (j==alt_col && i==alt_row){
        y[j*num_w + i] = x[i] * alt_w;
    }else{
        y[j*num_w + i] = x[i] * w[j*num_w + i];
    }
};

"""
#y[i] = x[stride*index+i]/max;
#//printf(\"(%d, %d)\\n\", i, j);
#y[j*num_w + i] = x[i] * w[j*num_w + i];
#//    printf(\"%f\\n\", x[i]);
#             //
#             //    printf(\"%f\\n\", y[j*num_w + i]);
#
#
#    __global unsigned char* x,
#    float v = x[i] + 1.0;
#    y[i] = v/(max+1.0);
#
# printf(\"GPU :(%d, %d) = %f\\n\", j, i, x[i]);
# printf(\"%f\\n\", y[i]);
#
class Gpu:
    def __init__(self):
        #self._ctx = cl.create_some_context()
        platform = cl.get_platforms()[0]
        # AMD Server
        #device = platform.get_devices()[0]
        # Intel on MBP
        device = platform.get_devices()[1]
        # AMD on eGPU
        #device = platform.get_devices()[2]
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
    
    def multiple_x_by_w(self, d_x, d_w, d_y, row, col):
        event = self.prg.multiple_x_by_w(self._queue,(row,col), None,
                                         d_x, d_w, d_y, np.int32(row))
        event.wait()

    def multiple_x_by_w_alt(self, d_x, d_w, d_y, row, col, layer_i, node_i, w):
        event = self.prg.multiple_x_by_w_alt(self._queue,(row,col), None,
                                             d_x, d_w, d_y, np.int32(row),
                                             np.int32(layer_i), np.int32(node_i),
                                             np.float32(w))
        event.wait()
    
    def scale(self, d_x, d_y, max, row, debug):
        event = self.prg.scale(self._queue, (row,), None,
                               d_x, d_y, np.float32(max), np.int32(debug))
        event.wait()
    
    def batch_scale(self, d_x, d_y, index, stride, max, row, debug):
        event = self.prg.batch_scale(self._queue, (row,), None,
                                     d_x, d_y, np.int32(index), np.int32(stride),
                                     np.float32(max), np.int32(debug))
        event.wait()
    
    def copy(self, dist, src):
        event = cl.enqueue_copy(self._queue, dist, src)
        event.wait()
    
#    def read(self, dist, src):
#        cl.enqueue_copy(self._queue, dist, src)
#
#    def write(self, dist, src):
#        cl.enqueue_copy(self._queue, dist, src)

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

