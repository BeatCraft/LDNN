#! /usr/bin/python
# -*- coding: utf-8 -*-
#

import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
import sys
#import time
#from time import time
import numpy as np
import pyopencl as cl

import gpu
#os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

KERNEL_CODE = """
__kernel void normalize_batch(
    __global float* data, 
    const int b_num, 
    const int ch_size, 
    const int ch_num)
{
    int ci = get_global_id(0);
    int stride = ch_size * ch_num;

    float sum = 0.0;
    float mean = 0.0;
    float dsum = 0.0;
    float delta = 0.0000001;
    float div2 = 0.0;
    float div = 0.0;
    int cnt = 0;

    for (int bi=0; bi<b_num; bi++){
        int start = bi * stride + ci * ch_size;
        for (int i=0; i<ch_size; i++){
            sum += data[start + i];
            cnt++;
        }
    }
    mean = sum / (float)cnt;

    for (int bi=0; bi<b_num; bi++){
        int start = bi * stride + ci * ch_size;
        for (int i=0; i<ch_size; i++){
            float k = data[start + i] - mean;
            dsum += (k*k);
        }
    }        
    div2 = dsum / (float)cnt;
    div = sqrt(div2) + delta;
    
    for (int bi=0; bi<b_num; bi++){
        int start = bi * stride + ci * ch_size;
        for (int i=0; i<ch_size; i++){
            float k = data[start + i] - mean;
            data[start + i] = k / div;
        }
    }
}

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
    int out_y_stride = (yi+1)*(w+2);
    
    for (int i=0; i<ch;i++){
        index = b_stride*bi + ch_stride*i + y_stride + xi;
        out_index = out_b_stride*bi + out_ch_stride*i + out_y_stride + xi + 1;
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
    const int filter,
    const int activation)
{
    int bi = get_global_id(0);
    int xi = get_global_id(1);
    int yi = get_global_id(2);

    int ch_stride = (w+2)*(h+2);
    int b_stride = ch_stride*ch;
    int y_stride = yi*(w+2);
    int i_start = b_stride*bi + y_stride;
    //printf(\"CL : bi=%d\\n\", bi);
    
    for (int fi=0; fi<filter; fi++){
        //printf(\"CL : fi=%d\\n\", fi);
        float sum = 0.0;
        int f_start = 3*3*fi*ch;
                    
        for (int i=0; i<ch; i++){
            //int start = b_stride*bi + ch_stride*i;
            int start = i_start + ch_stride*i;
            //int c_start = 3*3*i;
            int w_start = + f_start + i*3*3;
            
            sum += input[start + xi + 0] * weight[w_start + 0];
            sum += input[start + xi + 1] * weight[w_start + 1];
            sum += input[start + xi + 2] * weight[w_start + 2];
        
            sum += input[start + (w+2) + xi + 0] * weight[w_start + 3];
            sum += input[start + (w+2) + xi + 1] * weight[w_start + 4];
            sum += input[start + (w+2) + xi + 2] * weight[w_start + 5];
        
            sum += input[start + (w+2)*2 + xi + 0] * weight[w_start + 6];
            sum += input[start + (w+2)*2 + xi + 1] * weight[w_start + 7];
            sum += input[start + (w+2)*2 + xi + 2] * weight[w_start + 8];
        }

        //sum = sum / 9.0;
        
        // activation
        if (activation==0){
            output[bi*w*h*filter + w*h*fi + yi*w+xi] = sum;
        }else{ // relu
            if (sum<0.0){
                output[bi*w*h*filter + w*h*fi + yi*w+xi] = 0.0;
            }else{
                output[bi*w*h*filter + w*h*fi + yi*w+xi] = sum;
            }
        }
    }
};

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
    /*
    if (x[stride*j+i]==0){
        y[stride*j+i] = 0.0001;
    }else{
        y[stride*j+i] = x[stride*j+i]/max;
    }
    */
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
    float delta = 0.000001;
    float div2 = 0.0;
    float div = 0.0;

    for (int i=0; i<size; i++){
        sum += data[bi*size+i];
    }
    mean = sum / (float)size;
    
    sum = 0.0;
    for (int i=0; i<size; i++){
        float k = data[bi*size+i] - mean;
        sum += k * k;
    }
    div2 = sum / (float)size;
    div =  sqrt(div2 + delta);
    
    for (int i=0; i<size; i++){
        float k = data[bi*size+i] - mean;
        data[bi*size+i] = k / div;
    }
}

__kernel void scale_layer(__global float* data, int size)
{
    int bi = get_global_id(0);
    int start = bi*size;
    float max = 0.0;
    
    for (int i=0;i<size;i++){
        if (data[start+i]>max){
            max = data[start+i];
        }
    }
    
    if (max>0.0){
        for (int i=0;i<size;i++){
            data[start+i] = (data[start+i]/max);
        }
    }
}

__kernel void mse(__global const float* infs, __global const float* labels, __global float* output, int num)
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
    
    //output[bi] = sum/2.0;
    output[bi] = sum/(float)num;
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

__kernel void p_softmax(__global float* in, int num)
{
    int bi = get_global_id(0);
    float temp = 0.0;
    float total = 0.0;
    int start = bi*num;

    for (int i=0;i<num;i++){
        temp = in[start+i];
        temp = exp(temp);
        if (isinf(temp)){
            temp = 3.402823e+38;
        }else if (isnan(temp)){
            temp = 0;
        }
        in[start+i] = temp;
        total += temp;
    }

    //for (int i=0;i<num;i++){
    //    in[bi*num+i] = exp(in[bi*num+i]);
    //}
    //
    //for (int i=0;i<num;i++){
    //    sum += in[bi*num+i];
    //}
    
    
    //sum += 0.0000001;
    //printf(\"%d : %f\\n\", bi, sum);

    for (int i=0;i<num;i++){
        //printf(\"%f : %f\\n\", in[bi*num+i], sum);
        in[start+i] = in[start+i]/total;
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
        //out[num_node*bi + ni] = 0.0;
        out[num_node*bi + ni] = 0.000001;
    }else{
        out[num_node*bi + ni] = sum;
    }
}

__kernel void relu(__global float* out, int num, int stride, int mode)
{
    int bi = get_global_id(0);
    int ni = get_global_id(1);
    float k = 0.0;
    
    for (int i=0;i<num;i++){
        int idx = stride*bi + ni + i;
        if (out[idx]<0){
            if (mode==1){
                out[idx] = 0.0;
            }else if (mode==2){
                out[idx] = 0.000001;
            }else if (mode==3){
                out[idx] = out[idx]/20;
            }
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

__kernel void calc_mac_relu(
    __global float* x,
    __global float* w,
    __global float* y,
    int xsize, // node
    int wsize, // input
    int act)
{
    int bi = get_global_id(0); // batch
    int xi = get_global_id(1); // node
    
    int x_start = wsize * bi;
    int w_start = wsize * xi;
    int y_start = (xsize * bi) + xi;
    float temp = 0.0;

    for (int i=0;i<wsize;i++){
        temp += (x[x_start+i] * w[w_start+i]);
    }

    // activation
    if (temp>=0){
        y[y_start] = temp;
    }else{
        if (act==0){ // no
            y[y_start] = temp;
        }else if (act==1){ // relu
            y[y_start] = 0;
        }else if (act==2){
            y[y_start] = 0.000001;
        }else if (act==3){
            y[y_start] = temp/20;
        }else{
            y[y_start] = temp;
        }
    }
}

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
        
    def copy(self, dist, src):
        event = cl.enqueue_copy(self._queue, dist, src)
        event.wait()

    #
    #
    #

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
        
    def macRelu(self, buf_x, buf_w, buf_y, size_batch, size_node, size_input, act): # <<
        event = self.prg.calc_mac_relu(self._queue,(size_batch, size_node), None,
                                        buf_x, buf_w, buf_y,
                                        np.int32(size_node), np.int32(size_input),
                                        np.int32(act))
        event.wait()

    def multiple_x_by_w_batch(self, d_x, d_w, d_y, bsize, stride_1, stride_2, row, col):
        event = self.prg.multiple_x_by_w_batch(self._queue,(row,col,bsize), None,
                                               d_x, d_w, d_y,
                                               np.int32(stride_1),
                                               np.int32(stride_2))
        event.wait()
        
    def sum(self, data_in, data_out, num_input, num_node, activation, num_batch):
        event = self.prg.k_sum(self._queue, (num_node, num_batch), None,
                               data_in, data_out,
                               np.int32(num_input), np.int32(num_node), np.int32(activation))
        event.wait()
    
    def relu(self, data_out, batch_size, num_node, size, mode):
        event = self.prg.relu(self._queue, (batch_size, num_node), None,
                              data_out, np.int32(size), np.int32(num_node), np.int32(mode))
        event.wait()
    
    def scale_layer(self, data, size, batch_size):
        event = self.prg.scale_layer(self._queue, (batch_size,), None, data, np.int32(size))
        event.wait()
    
    def normalize_layer(self, data, size, batch_size):
        event = self.prg.normalize_layer(self._queue, (batch_size,), None, data, np.int32(size))
        event.wait()
    
    def softmax(self, data, size, num_batch): # <<
        event = self.prg.p_softmax(self._queue, (num_batch,), None, data, np.int32(size))
        event.wait()
    
    def mse(self, infs, labels, output, num_node, num_batch):
        event = self.prg.mse(self._queue, (num_batch,), None, infs, labels, output, np.int32(num_node))
        event.wait()
        
    def cross_entropy(self, infs, labels, output, num_node, num_batch):
        event = self.prg.cross_entropy(self._queue, (num_batch,), None,
                                       infs, labels, output, np.int32(num_node))
        event.wait()
    
    def max_batch(self, input, output, ch, w, h, num_batch): # <<
        event = self.prg.max_batch(self._queue, (num_batch, w, h), None,
                                   input, output, np.int32(ch), np.int32(w), np.int32(h))
        event.wait()
    #
    # CNN
    #
    def conv_4_pad_batch(self, input, output, w, h, ch, batch_size):
        event = self.prg.conv_4_pad_batch(self._queue, (batch_size, w, h), None,
                                          input, output, np.int32(w), np.int32(h), np.int32(ch))
        event.wait()
            
    def conv_4_roll_batch(self, input, weight, output, w, h, ch, filter, batch_size, activation=0):
        event = self.prg.conv_4_roll_batch(self._queue, (batch_size, w, h), None,
                                           input, weight, output, np.int32(w), np.int32(h), np.int32(ch), np.int32(filter), np.int32(activation))
        event.wait()

    #normalize_batch(__global float* data, int b_num, int ch_size, int ch_num)
    def normalize_batch(self, buf, b_num, ch_size, ch_num):
        event = self.prg.normalize_batch(self._queue, (ch_num,), None, 
                                         buf, np.int32(b_num), np.int32(ch_size), np.int32(ch_num))
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

