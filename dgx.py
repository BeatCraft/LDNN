import os
import sys
import time
import numpy as np
import cupy as cp
import cupyx

import gpu

calc_layer_mse = cp.RawKernel(r'''
extern "C" __global__
void calc_layer_mse(
    const float* input,
    float* output,
    const int ch,
    const int w,
    const int h)
{
    //int bi = get_global_id(0);
    int bi = blockIdx.x;
    
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
''', 'calc_layer_mse')


calc_cnn_max = cp.RawKernel(r'''
extern "C" __global__
void calc_cnn_max(
    const float* input,
    float* output,
    const int ch,
    const int w,
    const int h)
{
    //int bi = blockDim.x;
    int bi = blockIdx.x;
    int x = threadIdx.x;
    int y = threadIdx.y;

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
''', 'calc_cnn_max')


calc_cnn_roll = cp.RawKernel(r'''
extern "C" __global__
void calc_cnn_roll(
    const float* input,
    const float* weight,
    float* output,
    const int w,
    const int h,
    const int ch,
    const int filter)
{
    int bi = blockIdx.x;
    int xi = threadIdx.x;
    int yi = threadIdx.y;

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
        output[bi*w*h*filter + w*h*fi + yi*w+xi] = sum;
        // relu
        //if (sum<0.0){
        //    output[bi*w*h*filter + w*h*fi + yi*w+xi] = 0.0;
        //}else{
        //    output[bi*w*h*filter + w*h*fi + yi*w+xi] = sum;
        //}
    }
}
''', 'calc_cnn_roll')

# blockDim.x * blockIdx.x  + threadIdx.x;
# grid, block
calc_cnn_pad = cp.RawKernel(r'''
extern "C" __global__
void calc_cnn_pad(
    const float* input,
    float* output,
    const int w,
    const int h,
    const int ch)
{
    int bi = blockIdx.x;
    int xi = threadIdx.x;
    int yi = threadIdx.y;
    
    int index = 0;
    int out_index = 0;
    
    int b_stride = w*h*ch;
    int ch_stride = w*h;
    int y_stride = yi*w;
    
    int out_b_stride = (w+2)*(h+2)*ch;
    int out_ch_stride = (w+2)*(h+2);
    int out_y_stride = (yi+1)*(w+2);

    //printf("PAD(%d)(%d, %d)\n", bi, xi, yi);
    //printf("[softmax] inf\n");

    for (int i=0; i<ch;i++){
        index = b_stride*bi + ch_stride*i + y_stride + xi;
        out_index = out_b_stride*bi + + out_ch_stride*i + out_y_stride + xi + 1;
        output[out_index] = input[index];
    }
}
''', 'calc_cnn_pad')

calc_mac = cp.RawKernel(r'''
extern "C" __global__
void calc_mac(const float* x, const float* w, float* y, int size) {
    int x_start = size * blockIdx.x;
    int w_start = size * threadIdx.x;
    int y_start = blockDim.x * blockIdx.x + threadIdx.x;
    float temp = 0.0;

    for (int i=0;i<size;i++){
        temp += (x[x_start+i] * w[w_start+i]);
    }
    y[y_start] = temp;
}
''', 'calc_mac')


# size : size_input = number of weights
calc_mac_relu3 = cp.RawKernel(r'''
extern "C" __global__
void calc_mac_relu3(const float* x, const float* w, float* y, int xsize, int wsize, int act) {
    int bi = blockIdx.x;
    int xi = threadIdx.x;
    int x_start = (wsize * bi);
    int w_start = wsize * xi;
    int y_start = (xsize * bi) + xi;
    float temp = 0.0;

    for (int i=0;i<wsize;i++){
        temp += (x[x_start+i] * w[w_start+i]);
    }

    // activation
    if (act==0){
        y[y_start] = temp;
        return;
    }
    
    // relu
    if (temp>=0){
        y[y_start] = temp;
    } else {
        //y[y_start] = 0;
        //y[y_start] = 0.000001;
        y[y_start] = temp/20;
    }
}
''', 'calc_mac_relu3')

calc_mac_relu2 = cp.RawKernel(r'''
extern "C" __global__
void calc_mac_relu2(const float* x, const float* w, float* y, int xsize, int wsize) {
    int bi = blockIdx.x;
    int xi = threadIdx.x;
    int x_start = (wsize * bi);
    int w_start = wsize * xi;
    int y_start = (xsize * bi) + xi;
    float temp = 0.0;

    for (int i=0;i<wsize;i++){
        temp += (x[x_start+i] * w[w_start+i]);
    }

    if (temp>=0){
        y[y_start] = temp;
    } else {
        //y[y_start] = 0;
        //y[y_start] = 0.000001;
        y[y_start] = temp/20;
    }
}
''', 'calc_mac_relu2')

calc_mac_relu = cp.RawKernel(r'''
extern "C" __global__
void calc_mac_relu(const float* x, const float* w, float* y, int size) {
    int x_start = size * blockIdx.x;
    int w_start = size * threadIdx.x;
    int y_start = blockDim.x * blockIdx.x + threadIdx.x;
    float temp = 0.0;

    printf("[%d, %d, %d]\n", blockDim.x, blockIdx.x, threadIdx.x);

    for (int i=0;i<size;i++){
        temp += (x[x_start+i] * w[w_start+i]);
    }

    if (temp>=0){
        y[y_start] = temp;
    } else {
        //y[y_start] = 0;
        //y[y_start] = 0.000001;
        y[y_start] = temp/20;
    }
}
''', 'calc_mac_relu')

calc_relu = cp.RawKernel(r'''
extern "C" __global__
void calc_relu(float* buf, int size_input, int mode){
    int bi = blockIdx.x;
    int ni = threadIdx.x;

    for (int i=0;i<size_input;i++){
        int idx = bi*size_input;
        if (buf[idx+i]<0){
            if (mode==1){
                buf[idx+i] = 0.0;
            }else if (mode==2){
                buf[idx+i] = 0.000001;
            }else if (mode==3){
                buf[idx+i] = buf[idx+i]/20;
            }
        }
    }
}
''', 'calc_relu')

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

cals_layer_normalize = cp.RawKernel(r'''
extern "C" __global__
void cals_layer_normalize(float* x, int size) {
    int x_start = size * blockIdx.x;

    float u = 0.0;
    float s = 0.0;
    
    for (int i=0;i<size;i++){
        u += x[x_start+i];
    }
    u = u / float(size);

    for (int i=0;i<size;i++){
        s += (x[x_start+i] - u)*(x[x_start+i] - u);
    }
    s = s / float(size);
    s = sqrtf(s + 0.000001);
    
    for (int i=0;i<size;i++){
        x[x_start+i] = (x[x_start+i] - u) / s;
    }
}
''', 'cals_layer_normalize')

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

calc_batch_normalize = cp.RawKernel(r'''
extern "C" __global__
void calc_batch_normalize(float* data, const int b_num, int ch_size, int ch_num) {
    int ci = blockIdx.x;//get_global_id(0);
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
''', 'calc_batch_normalize')


class Dgx(gpu.Gpu):
    def __init__(self, device_id):
        super(gpu.Gpu, self).__init__()
        self.name = "nvidia DGX V100"
        self.id = device_id
        cp.cuda.Device(device_id).use()
        # -1:unknown, 0:OpenCL, 1:CuPy/DGX
        self.type = 1

    def allocateArray(self, np_array):
        return cp.asarray(np_array)

    def crossEntropy(self, buf_x, buf_l, buf_y, size_batch, size_node):
        calc_entropy((size_batch,), (1,), (buf_x, buf_l, buf_y, size_node))

    def layerScale(self, buf, size_batch, size_node):
        cals_layer_scale((size_batch,), (1,), (buf, size_node))
        
    def layerNormalize(self, buf, size_batch, size_node):
        cals_layer_normalize((size_batch,), (1,), (buf, size_node))
        
        
    def batchNormalize(self, buf, size_batch, size_rect, size_node):
        calc_batch_normalize((size_node,), (1,),
                             (buf, np.int32(size_batch), np.int32(size_rect), np.int32(size_node)))
        
    def mac(self, buf_x, buf_w, buf_y, size_batch, size_node, size_input):
        calc_mac((size_batch,), (size_node,), (buf_x, buf_w, buf_y, size_input))

    def macRelu(self, buf_x, buf_w, buf_y, size_batch, size_node, size_input):
        calc_mac_relu((size_batch,), (size_node,), (buf_x, buf_w, buf_y, size_input))
        
    def macRelu2(self, buf_x, buf_w, buf_y, size_batch, size_node, size_input):
        calc_mac_relu2((size_batch,), (size_node,), (buf_x, buf_w, buf_y, size_node, size_input))
    
    def macRelu3(self, buf_x, buf_w, buf_y, size_batch, size_node, size_input, act): # 0 : no activation
        calc_mac_relu3((size_batch,), (size_node,), (buf_x, buf_w, buf_y, size_node, size_input, act))
        
    def softmax(self, buf_x, buf_y, size_batch, size_node):
        calc_softmax((size_batch,), (1,), (buf_x, buf_y, size_node))
    #
    # cnn
    #
    def padding(self, buf_x, buf_y, w, h, ch, batch_size):
        calc_cnn_pad((batch_size,), (w, h,), (buf_x, buf_y, w, h, ch))
        
    def convolusion(self, buf_x, weight, buf_y, w, h, ch, filter, batch_size):
        calc_cnn_roll((batch_size,), (w, h,), (buf_x, weight, buf_y, w, h, ch, filter))
        
    def max(self, buf_x, buf_y, ch, w, h, batch_size):
        calc_cnn_max((batch_size,), (w, h,), (buf_x, buf_y, ch, w, h))

    def layer_mse(self, buf_x, buf_y, ch, w, h, batch_size):
        calc_layer_mse((batch_size,), (1,), (buf_x, buf_y, ch, w, h))
    
    def relu(self, buf, size_input, mode, batch_size, size_node):
        calc_relu((batch_size,), (size_node,), (buf, size_input, mode))

    def make_attack_list(self, div, mode, w_list, result_gpu):
        pass
    
    
def main():
    return 0

if __name__=='__main__':
    print(">> start")
    sts = main()
    print(">> end")
    print("\007")
    sys.exit(sts)

