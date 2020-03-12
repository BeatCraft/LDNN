import pyopencl as cl
import os, sys, time
from time import time
import numpy as np
#
#
# WEIGHT_SET_0 = [-1.0, -0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625, -0.0078125,
#                  0.0,
#                  0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]
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

__kernel void p_softmax(__global float* in, int num)
{
    int bi = get_global_id(0);
    float max;
    float sum;
    
    max = 0.0;
    sum = 0.0;
    
//    for (int i=0;i<num;i++){
//        if (in[bi*num+i]>max){
//            max = in[bi*num+i];
//        }
//    }

    for (int i=0;i<num;i++){
        in[bi*num+i] = exp(in[bi*num+i]);
    }

    for (int i=0;i<num;i++){
        sum += in[bi*num+i];
    }

    for (int i=0;i<num;i++){
        in[bi*num+i] = in[bi*num+i]/sum;
    }

//    for (int i=0;i<num;i++){
//        sum += in[bi*num+i];
//    }
//
//   for (int i=0;i<num;i++){
//        in[bi*num+i] = in[bi*num+i]/sum;
//    }
}


__kernel void k_cross_entropy(__global float* infs,
                              __global float* output,
                              __global int* labels,
                              int num)
{
    int bi = get_global_id(0);
    float delta;
    float k;
    int i;
    
    delta = 0.0000001;
    i = labels[bi];
    k = infs[bi*10+i]+delta;    
    output[bi] = -log(k);
}

__kernel void k_softmax(__global float* in, int num)
{
    int bi = get_global_id(0);
    float max;
    float sum;
    
    max = 0.0;
    sum = 0.0;
    
    for (int i=0;i<num;i++){
        if (in[bi*num+i]>max){
            max = in[bi*num+i];
        }
    }
    
    for (int i=0;i<num;i++){
        sum += exp(in[bi*num+i]-max);
    }

    for (int i=0;i<num;i++){
        in[bi*num+i] = exp(in[bi*num+i]-max)/sum;
    }
}

__kernel void k_sum(__global float* in,
                    __global float* out,
                    int num_input,
                    int num_node,
                    int activation)
{
    int ni = get_global_id(0);
    int bi = get_global_id(1);
    int ii;
    float sum;

    ii = 0;
    sum = 0.0;
    
    for (ii=0;ii<num_input;ii++){
        sum += in[num_node*num_input*bi + num_input*ni + ii];
    }
    
    // relu
    if (activation==0 && sum<0.0){
        out[num_node*bi + ni] = 0.0;
    }else{
        out[num_node*bi + ni] = sum;
    }
}

__kernel void multiple_x_by_w_batch(
    __global float* x,
    __global float* w,
    __global float* y,
    const int stride_1,
    const int stride_2)
{
    int i = get_global_id(0);  // num_input
    int j = get_global_id(1);  // num_node
    int bi = get_global_id(2); // batch id
    
    y[stride_1*bi + stride_2*j+i] = x[stride_2*bi+i] * w[stride_2*j+i];
};

// stride_1 : num_node * num_input
// stride_2 : num_input

__kernel void multiple_x_by_w_batch_alt(
    __global float* x,
    __global float* w,
    __global float* y,
    const int stride_1,
    const int stride_2,
    const int alt_ni,
    const int alt_ii,
    const float alt_w)
{
    int i = get_global_id(0); // num_input
    int j = get_global_id(1); // num_node
    int bi = get_global_id(2); // batch id

    if (j==alt_ni && i==alt_ii){
        y[stride_1*bi + stride_2*j + i] = x[stride_2*bi + i] * alt_w;
    }else{
        y[stride_1*bi + stride_2*j + i] = x[stride_2*bi + i] * w[stride_2*j + i];
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
    def __init__(self, platform_id, device_id):
        platform = cl.get_platforms()[platform_id]
        device = platform.get_devices()[device_id]
        print platform
        print device
        #
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

    def multiple_x_by_w_batch(self, d_x, d_w, d_y, bsize, stride_1, stride_2, row, col):
        event = self.prg.multiple_x_by_w_batch(self._queue,(row,col,bsize), None,
                                               d_x, d_w, d_y,
                                               np.int32(stride_1),
                                               np.int32(stride_2))
        event.wait()
        
    def multiple_x_by_w_batch_alt(self, d_x, d_w, d_y, bsize, stride_1, stride_2, row, col, ni, ii, wv):
        event = self.prg.multiple_x_by_w_batch_alt(self._queue,(row,col,bsize), None,
                                                   d_x, d_w, d_y,
                                                   np.int32(stride_1),
                                                   np.int32(stride_2),
                                                   np.int32(ni), np.int32(ii), np.float32(wv))
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
        
    def k_sum(self, data_in, data_out, num_input, num_node, activation, num_batch):
        event = self.prg.k_sum(self._queue, (num_node, num_batch), None,
                               data_in, data_out,
                               np.int32(num_input), np.int32(num_node), np.int32(activation))
        event.wait()
    
    def k_softmax(self, data, size, num_batch):
        event = self.prg.p_softmax(self._queue, (num_batch,), None, data, np.int32(size))
        event.wait()
    
    def k_cross_entropy(self, infs, output, labels, size, num_batch):
        event = self.prg.k_cross_entropy(self._queue, (num_batch,), None,
                                        infs, output, labels, np.int32(size))
        event.wait()
#
#
#
def main():
    data_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5]).astype(np.float32)
    data_w = np.array([[0.5, 0.5, 0.5, 0.5, 0.2, 0.5, 0.5, 0.5, 0.5, 0.2, 0.5, 0.5, 0.5, 0.5, 0.2],
                       [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                       [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5]]).astype(np.float32)
    data_y = np.array([[0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0]]).astype(np.float32)
    data_a = np.array([8, 16, 32, 64]).astype(np.int32)
    data_b = np.array([0.0, 0.0, 0.0, 0.0]).astype(np.float32)

    print data_x
    print data_w
    print data_y
    
    platform_id = 0
    device_id = 1
    g = Gpu(platform_id, device_id)
    
    g.dev_malloc(data_x) # 0
    g.dev_malloc(data_w) # 1
    g.dev_malloc(data_y) # 2
    g.dev_malloc(data_a) # 3
    g.dev_malloc(data_b) # 4
    
    g.set_kernel_code()
    bufs = g.get_buffer_list()

    num_node = data_w.shape[0]
    num_input =  data_w[0].shape[0]
    
    stride = num_input / 2
    left = num_input % 2
    
    print "num=%d, stride=%d, left=%d" % (num_input, stride, left)
    #g.k_sum(bufs[1], bufs[4], stride, left, 0, num_input, num_node)
    g.k_sum(bufs[1], bufs[4], num_input, num_node, 0, 1)

    #g.testp(bufs[0], num, bufs[4], num)
    #g.k_sum(bufs[0], bufs[4], stride, left)
    #g.multiple_x_by_w(bufs[0], bufs[1], bufs[2], num)
    #g.read(data_y, bufs[2])
    #print data_y
    #data_y[0][0]=0.999
    #print data_y
    
    #g.scale(bufs[3], bufs[4], 255.0, 4)
    #g.read(data_x, bufs[4])
    #print data_x
    
    #g.write(bufs[2], data_a)
    #
    #for row in data_y:
    #    print np.sum(row)
    return 0
#
#
#
if __name__=='__main__':
    print ">> start"
    sts = main()
    print ">> end"
    print("\007")
    sys.exit(sts)

