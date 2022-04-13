import os
import sys
import time
import numpy as np
import cupy as cp
import cupyx

import gpu

calc_mac = cp.RawKernel(r'''
extern "C" __global__
void calc_mac(const float* x, const float* w, float* y, int size) {
    int x_start = size * blockIdx.x;
    int w_start = size * threadIdx.x;
    int y_start = blockDim.x * blockIdx.x +  threadIdx.x;
    float temp = 0.0;

    for (int i=0;i<size;i++){
        temp += (x[x_start+i] * w[w_start+i]);
    }
    y[y_start] = temp;
}
''', 'calc_mac')

calc_mac_relu = cp.RawKernel(r'''
extern "C" __global__
void calc_mac_relu(const float* x, const float* w, float* y, int size) {
    int x_start = size * blockIdx.x;
    int w_start = size * threadIdx.x;
    int y_start = blockDim.x * blockIdx.x  + threadIdx.x;
    float temp = 0.0;

    for (int i=0;i<size;i++){
        temp += (x[x_start+i] * w[w_start+i]);
    }

    if (temp>=0){
        y[y_start] = temp;
    } else {
        y[y_start] = 0;
    }
}
''', 'calc_mac_relu')

cals_layer_scale = cp.RawKernel(r'''
extern "C" __global__
void cals_layer_scale(float* x, int size) {
    int x_start = size * blockIdx.x;
    float temp = 0.0;

    for (int i=0;i<size;i++){
        if (x[x_start+i]>temp){
            temp = x[x_start+i];
        }
    }

    if (temp>0.0){
        for (int i=0;i<size;i++){
            x[x_start+i] = x[x_start+i] / temp;
        }
    }
}
''', 'cals_layer_scale')

calc_softmax = cp.RawKernel(r'''
extern "C" __global__
void calc_softmax(const float* x, double* y, int size) {
    int x_start = size * blockIdx.x;
    int y_start = x_start;
    double temp = 0.0;
    double total = 0.0;
    
    for (int i=0;i<size;i++){
        temp = x[x_start+i];
        temp = exp(temp);
        if (isinf(temp)){
            //printf("[softmax] inf\n");
            temp = 3.402823e+38;
        }else if (isnan(temp)){
            //printf("[softmax] nan\n");
            temp = 0;
        }
        y[y_start+i] = temp;
        total += temp;
    }

    for (int i=0;i<size;i++){
        //double k = y[y_start+i];
        y[y_start+i] = y[y_start+i] / total;
        //printf("[%d, %d] %f\n", blockIdx.x, i, y[y_start+i]);
    }
}
''', 'calc_softmax')

calc_entropy = cp.RawKernel(r'''
extern "C" __global__
void calc_entropy(const double* x, const float *a, double* y, int size) {
    int x_start = size * blockIdx.x;
    int a_start = x_start;
    int y_start = blockIdx.x;

    float t = 0.0;
    float k = 0.0;
    float total = 0.0;
    float delta = 0.0000001;
    
    for (int i=0;i<size;i++){
        t = a[a_start+i];
        k = x[x_start+i] + delta;
        //printf("[%d, %d]%f, %f, %f\n", blockIdx.x, i, t, k,  log(k));
        total += t * log(k);
        //printf("%f, %f, %f\n", total, t*log(k), t);
    }

    y[y_start] = (-1.0) * total;
    //printf("[%d] %f, %f\n", blockIdx.x, y[y_start], total);
}
''', 'calc_entropy')

class Gdx(gpu.Gpu):
    def __init__(self, device_id):
        super(gpu.Gpu, self).__init__()
        self.name = "GDX V100"
        self.id = device_id
        self.type = 1 # -1:unknown, 0:OpenCL, 1:CuPy/GDX

    def allocateArray(self, np_array):
        #with cp.cuda.Device(self.id):
        return cp.asarray(np_array)
        #
        #return None

    def crossEntropy(self, buf_x, buf_l, buf_y, size_batch, size_node):
        #with cp.cuda.Device(self.id):
        calc_entropy((size_batch,), (1,), (buf_x, buf_l, buf_y, size_node))
        #

    def layerScale(self, buf, size_batch, size_node):
        #with cp.cuda.Device(self.id):
        cals_layer_scale((size_batch,), (1,), (buf, size_node))
        #

    def mac(self, buf_x, buf_w, buf_y, size_batch, size_node, size_input):
        #with cp.cuda.Device(self.id):
        calc_mac((size_batch,), (size_node,), (buf_x, buf_w, buf_y, size_input))
        #

    def macRelu(self, buf_x, buf_w, buf_y, size_batch, size_node, size_input):
        #with cp.cuda.Device(self.id):
         calc_mac_relu((size_batch,), (size_node,), (buf_x, buf_w, buf_y, size_input))
        #

    def softmax(self, buf_x, buf_y, size_batch, size_node):
        #with cp.cuda.Device(self.id):
        calc_softmax((size_batch,), (1,), (buf_x, buf_y, size_node))
        #

def main():
    return 0

if __name__=='__main__':
    print(">> start")
    sts = main()
    print(">> end")
    print("\007")
    sys.exit(sts)

