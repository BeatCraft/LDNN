#! /usr/bin/python
# -*- coding: utf-8 -*-
#

import os
import sys
#import time
#from time import time
import numpy as np
import pyopencl as cl

import gpu
#os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

KERNEL_CODE = """
__kernel void conv_4_pad_batch(
    __global float* input,
    __global float* output,
    const int w,
    const int h,
    const int ch)
{
    int bi = get_global_id(0);
    int xi = get_global_id(1);
    int yi = get_global_id(2);
    
    int index = 0;
    int out_index = 0;
    
    int b_stride = w*h*ch;
    int ch_stride = w*h;
    int y_stride = yi*w;
    
    int out_b_stride = (w+2)*(h+2)*ch;
    int out_ch_stride = (w+2)*(h+2);
    int out_y_stride = yi*(w+2);
    
    for (int i=0; i<ch;i++){
        index = b_stride*bi + ch_stride*i + y_stride + xi;
        out_index = out_b_stride*bi + + out_ch_stride*i + out_y_stride + xi;
        output[out_index] = input[index];
    }
};

__kernel void conv_4_roll_batch(
    __global float* input,
    __global float* weight,
    __global float* output,
    const int w,
    const int h,
    const int ch,
    const int filter)
{
    int bi = get_global_id(0);
    int xi = get_global_id(1);
    int yi = get_global_id(2);

    int ch_stride = (w+2)*(h+2);
    int b_stride = ch_stride*ch;
    int y_stride = yi*(w+2);

    //printf(\"CL : bi=%d\\n\", bi);
    
    for (int fi=0; fi<filter; fi++){
        float sum = 0.0;
        //printf(\"CL : fi=%d\\n\", fi);
        
        for (int i=0; i<ch; i++){
            int start = b_stride*bi + ch_stride*i;
            
            sum += input[start + y_stride + xi    ] * weight[fi*ch*3*3 + i*3*3    ];
            sum += input[start + y_stride + xi + 1] * weight[fi*ch*3*3 + i*3*3 + 1];
            sum += input[start + y_stride + xi + 2] * weight[fi*ch*3*3 + i*3*3 + 2];
        
            sum += input[start + y_stride + (w+2) + xi    ] * weight[fi*ch*3*3 + i*3*3 + 3];
            sum += input[start + y_stride + (w+2) + xi + 1] * weight[fi*ch*3*3 + i*3*3 + 4];
            sum += input[start + y_stride + (w+2) + xi + 2] * weight[fi*ch*3*3 + i*3*3 + 5];
        
            sum += input[start + y_stride + (w+2)*2 + xi    ] * weight[fi*ch*3*3 + i*3*3 + 6];
            sum += input[start + y_stride + (w+2)*2 + xi + 1] * weight[fi*ch*3*3 + i*3*3 + 7];
            sum += input[start + y_stride + (w+2)*2 + xi + 2] * weight[fi*ch*3*3 + i*3*3 + 8];
        }
        // relu
        if (sum<0.0){
            output[bi*w*h*filter + w*h*fi + yi*w+xi] = 0.0;
        }else{
            output[bi*w*h*filter + w*h*fi + yi*w+xi] = sum;
        }
    }
};

__kernel void layer_mse_batch(
    __global const float* input,
    __global float* output,
    const int ch,
    const int w,
    const int h)
{
    int bi = get_global_id(0);
    
    int ch_stride = w*h;
    int input_offset = ch_stride*ch*bi;

    for (int c=1;c<ch;c++){
        int ch_start = input_offset + (ch_stride*c);
        float sum_d = 0.0;
        for (int i=0;i<ch_stride;i++){
            float d = input[input_offset+i] - input[ch_start+i];
            sum_d += (d*d);
        }
        output[bi] = sum_d/float(ch_stride);
    }
}

__kernel void max_batch(
    __global float* input,
    __global float* output,
    const int ch,
    const int w, // output w
    const int h)
{
    int bi = get_global_id(0);
    int y = get_global_id(1);
    int x = get_global_id(2);
    
    int input_w = w*2;
    int input_h = h*2;
    int ich_stride = input_w * input_h;
    int input_stride = ich_stride * ch;
    int input_offset = input_stride * bi;
    
    int output_w = w;
    int output_h = h;
    int och_stride = output_w * output_h;
    int output_stride = och_stride * ch;
    int output_offset = output_stride * bi;

    float max = 0.0;
    float a[4];

    for (int c=0;c<ch;c++){
        int k = input_offset + (ich_stride*c) + (w*2)*y + x*2;
        a[0] = input[k];
        a[1] = input[k + 1];
        a[2] = input[k + w*2];
        a[3] = input[k + w*2 + 1];
        for (int i=0;i<4;i++){
            if (a[i]>max){
                max = a[i];
            }
        }
        output[output_offset + och_stride*c + w*y + x] = max;
    }
}

__kernel void scale_exp(
    __global float* x,
    __global float* y,
    const int stride,
    const int debug)
{
    int i = get_global_id(0); // data index
    int j = get_global_id(1); // bathch index
    
    y[stride*j+i] = exp(x[stride*j+i]);
    
    if (debug==1){
        printf(\"%d, %d\\n\",i, j);
    }
};

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

__kernel void normalize_layer(__global float* data, int size)
{
    int bi = get_global_id(0);
    float sum = 0.0;
    float mean = 0.0;
    float delta = 0.0000001;
    float div2 = 0.0;
    float div = 0.0;

    for (int i=0; i<size; i++){
        sum += data[bi*size+i];
    }
    mean = sum / size;
    
    sum = 0.0;
    for (int i=0; i<size; i++){
        float k = data[bi*size+i] - mean;
        sum += k * k;
    }
    div2 = sum / size + delta;
    div =  sqrt(div2);
    
    for (int i=0; i<size; i++){
        float k = data[bi*size+i] - mean;
        data[bi*size+i] = k / div;
    }
}

__kernel void scale_layer(__global float* data, int size)
{
    int bi = get_global_id(0);
    float max = 0.0;
    
    for (int i=0;i<size;i++){
        if (data[bi*size+i]>max){
            max = data[bi*size+i];
        }
    }
    
    for (int i=0;i<size;i++){
        if (max > 0.0){
            data[bi*size+i] = (data[bi*size+i]/max);
        }else{
            data[bi*size+i] = 0.0;
        }
    }
}


__kernel void mse(__global const float* infs,
                            __global const float* labels,
                            __global float* output,
                            int num)
{
    int bi = get_global_id(0); // batch index
    
    float k;
    float t;
    float d;
    float sum;

    sum = 0.0;

    for (int i=0;i<num;i++){
        t = labels[bi*num + i];
        k = infs[bi*num + i];
        d = t - k;
        sum += d * d;
    }
    
    output[bi] = sum/2.0;
}

__kernel void cross_entropy_rg(__global const float* infs,
                            __global const float* labels,
                            __global float* output,
                            int num)
{
    int bi = get_global_id(0); // batch index
    //int ni = get_global_id(1); // node index
    
    float delta;
    float k;
    float t;
    float sum;
    
    delta = 0.0000001;
    sum = 0.0;
    
//    printf(\"[%d](%d)\\n\", bi, num);

    for (int i=0;i<num;i++){
        //printf(\"[%d](%d)\\n\", bi, i);
        t = labels[bi*num + i];
        k = infs[bi*num + i] + delta;
        //printf(\"[%d](%i) %f\\n\", bi, i, t);
        sum += t * log(k);
        //printf(\"[%d](%i) %f\\n\", bi, i, t * log(k));
    }
    
    output[bi] = (-1.0)*sum;
    //printf(\"%d | %f\\n\", bi, output[bi]);
}



__kernel void cross_entropy(__global const float* infs,
                            __global const float* labels,
                            __global float* output,
                            int num)
{
    int bi = get_global_id(0); // batch index
    //int ni = get_global_id(1); // node index
    
    float delta;
    float k;
    float t;
    float sum;
    
    delta = 0.0000001;
    sum = 0.0;
    
    for (int i=0;i<num;i++){
        t = labels[bi*num + i];
        k = infs[bi*num + i] + delta;
        //printf(\"%d-%d | %f\\n\", bi, i, t * log(k));
        sum += t * log(k);
    }
    
    output[bi] = (-1.0)*sum;
    //printf(\"%d | %f\\n\", bi, output[bi]);
}

__kernel void k_cross_entropy(__global const float* infs,
                              __global float* output,
                              __global const int* labels,
                              int num)
{
    int bi = get_global_id(0);
    float delta;
    float k;
    int i;
    
    delta = 0.0000001;
    i = labels[bi];
    k = infs[bi*num+i]+delta;
    output[bi] = -log(k);
}

__kernel void p_softmax(__global float* in, int num)
{
    int bi = get_global_id(0);
    float sum = 0.0;

    for (int i=0;i<num;i++){
        in[bi*num+i] = exp(in[bi*num+i]);
    }

    for (int i=0;i<num;i++){
        sum += in[bi*num+i];
    }
    //sum += 0.0000001;
    //printf(\"%d : %f\\n\", bi, sum);

    for (int i=0;i<num;i++){
        in[bi*num+i] = in[bi*num+i]/sum;
    }
}

__kernel void k_softmax(__global float* in, int num)
{
    int bi = get_global_id(0);
    float max = 0.0;
    float sum = 0.0;
    
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

__kernel void k_sum(__global const float* in,
                    __global float* out,
                    int num_input,
                    int num_node,
                    int activation)
{
    int ni = get_global_id(0);
    int bi = get_global_id(1);
    int ii = 0;
    float sum = 0.0;
    
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

__kernel void relu(__global float* out, int num, int stride)
{
    int bi = get_global_id(0);
    int ni = get_global_id(1);
    float k = 0.0;
    
    for (int i=0;i<num;i++){
        k = out[stride*bi + ni + i];
        if (k<0.0){
            out[stride * bi + ni + i] = 0.0;
        }else{
            out[stride * bi + ni + i] = k;
        }
    }
}

__kernel void multiple_x_by_w_batch(
    __global const float* x,
    __global const float* w,
    __global float* y,
    const int stride_1,
    const int stride_2)
{
    int i = get_global_id(0);  // num_input
    int j = get_global_id(1);  // num_node
    int bi = get_global_id(2); // batch id

    y[stride_1*bi + stride_2*j+i] = x[stride_2*bi+i] * w[stride_2*j+i];
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

__kernel void k_test(const float in)
{
    int i = get_global_id(0);
    float out = 0.0;
    out = exp(in);
    printf(\"%d : exp(%If) = %If\\n\", i, in, out);
};

"""
#
#
#
class OpenCL(gpu.Gpu):
    def __init__(self, platform_id, device_id):
        super(gpu.Gpu, self).__init__()
        self.name = "OpenCL"
        self.type = 0 # -1:unknown, 0:OpenCL, 1:CuPy/GDX
        self.platform_id = platform_id
        self.device_id = device_id
        platform = cl.get_platforms()[platform_id]
        device = platform.get_devices()[device_id]
        print(platform)
        print(device)
        #
        self._ctx = cl.Context([device])
        for dev in self._ctx.devices:
            assert dev.local_mem_size > 0
        #
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

    def scale_exp(self, d_x, d_y, stride, row, batch_size, debug):
        event = self.prg.scale_exp(self._queue, (row, batch_size), None,
                               d_x, d_y, np.int32(stride), np.int32(debug))
        event.wait()

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
        
    #def int_multiple_x_by_w_batch(self, d_x, d_w, d_y, bsize, stride_1, stride_2, row, col):
    #    event = self.prg.int_multiple_x_by_w_batch(self._queue,(row,col,bsize), None,
    #                                           d_x, d_w, d_y,
    #                                           np.int32(stride_1),
    #                                           np.int32(stride_2))
    #    event.wait()
    
    #def int_sum(self, data_in, data_out, num_input, num_node, activation, num_batch):
     #   event = self.prg.int_sum(self._queue, (num_node, num_batch), None,
     #                          data_in, data_out,
     #                          np.int32(num_input), np.int32(num_node), np.int32(activation))
     #   event.wait()
    
    #def int_scale_layer(self, data, size, batch_size):
    #    event = self.prg.int_scale_layer(self._queue, (batch_size,), None, data, np.int32(size))
    #    event.wait()
    
    def copy(self, dist, src):
        event = cl.enqueue_copy(self._queue, dist, src)
        event.wait()
        
    def sum(self, data_in, data_out, num_input, num_node, activation, num_batch):
        event = self.prg.k_sum(self._queue, (num_node, num_batch), None,
                               data_in, data_out,
                               np.int32(num_input), np.int32(num_node), np.int32(activation))
        event.wait()
    
    def relu(self, data_out, batch_size, num_node, size):
        event = self.prg.relu(self._queue, (batch_size, num_node), None,
                              data_out, np.int32(size), np.int32(num_node))
        event.wait()
    
    def scale_layer(self, data, size, batch_size):
        event = self.prg.scale_layer(self._queue, (batch_size,), None, data, np.int32(size))
        event.wait()
    
    def normalize_layer(self, data, size, batch_size):
        event = self.prg.normalize_layer(self._queue, (batch_size,), None, data, np.int32(size))
        event.wait()
    
    def softmax(self, data, size, num_batch):
        event = self.prg.p_softmax(self._queue, (num_batch,), None, data, np.int32(size))
        event.wait()

    #def int_softmax(self, data, data_out, size, num_batch):
    #    event = self.prg.int_softmax(self._queue, (num_batch,), None, data, data_out, np.int32(size))
    #    event.wait()

    #def int_cross_entropy(self, infs, labels, output, num_node, num_batch):
    #    event = self.prg.int_cross_entropy(self._queue, (num_batch,), None,
    #                                   infs, labels, output, np.int32(num_node))
    #    event.wait()
    
    def mse(self, infs, labels, output, num_node, num_batch):
        event = self.prg.mse(self._queue, (num_batch,), None, infs, labels, output, np.int32(num_node))
        event.wait()
        
    def cross_entropy_rg(self, infs, labels, output, num_node, num_batch):
        event = self.prg.cross_entropy_rg(self._queue, (num_batch,), None,
                                       infs, labels, output, np.int32(num_node))
        event.wait()
        
    def cross_entropy(self, infs, labels, output, num_node, num_batch):
        event = self.prg.cross_entropy(self._queue, (num_batch,), None,
                                       infs, labels, output, np.int32(num_node))
        event.wait()
    
    def k_cross_entropy(self, infs, output, labels, size, num_batch):
        event = self.prg.k_cross_entropy(self._queue, (num_batch,), None,
                                        infs, output, labels, np.int32(size))
        event.wait()
    
    
    
    def layer_mse_batch(self, input, output, ch, w, h, num_batch):
        event = self.prg.layer_mse_batch(self._queue, (num_batch,), None,
                                   input, output, np.int32(ch), np.int32(w), np.int32(h))
        event.wait()
    
    def max_batch(self, input, output, ch, w, h, num_batch):
        event = self.prg.max_batch(self._queue, (num_batch, w, h), None,
                                   input, output, np.int32(ch), np.int32(w), np.int32(h))
        event.wait()
    #
    # new CNN implementations
    #
    def conv_4_pad_batch(self, input, output, w, h, ch, batch_size):
        event = self.prg.conv_4_pad_batch(self._queue, (batch_size, w, h), None,
                                          input, output, np.int32(w), np.int32(h), np.int32(ch))
        event.wait()
            
    def conv_4_roll_batch(self, input, weight, output, w, h, ch, filter, batch_size):
        event = self.prg.conv_4_roll_batch(self._queue, (batch_size, w, h), None,
                                           input, weight, output, np.int32(w), np.int32(h), np.int32(ch), np.int32(filter))
        event.wait()
    
    def k_test(self, value):
        event = self.prg.k_test(self._queue, (1,), None, np.float32(value))
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
    data_b = np.array([0.0, 0.0, 0.0, 0.0]).astype(np.float64)

    print(data_x)
    print(data_w)
    print(data_y)
    
    platform_id = 0
    device_id = 1
    g = Gpu(platform_id, device_id)
    g.set_kernel_code()
    #
    p = 0.0
    for i in range(100):
        g.k_test(p)
        p = p + 1.0
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

