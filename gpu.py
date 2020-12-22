import pyopencl as cl
import os, sys, time
from time import time
import numpy as np

import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
#
#
#
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

    int index = 0;
    int b_stride = w*h*ch;
    int ch_stride = w*h;
    int y_stride = yi*w;
    
    for (int fi=0; fi<filter; fi++){
        float sum = 0.0;
        for (int i=0; i<ch; i++){
            int start = b_stride*bi + ch_stride*i;
            sum += input[start + y_stride - w + xi - 1] * weight[fi*ch*3*3 + i*3*3    ];
            sum += input[start + y_stride - w + xi    ] * weight[fi*ch*3*3 + i*3*3 + 1];
            sum += input[start + y_stride - w + xi + 1] * weight[fi*ch*3*3 + i*3*3 + 2];
        
            sum += input[start + y_stride + xi - 1] * weight[fi*ch*3*3 + i*3*3 + 3];
            sum += input[start + y_stride + xi    ] * weight[fi*ch*3*3 + i*3*3 + 4];
            sum += input[start + y_stride + xi + 1] * weight[fi*ch*3*3 + i*3*3 + 5];
        
            sum += input[start + y_stride + w + xi - 1] * weight[fi*ch*3*3 + i*3*3 + 6];
            sum += input[start + y_stride + w + xi    ] * weight[fi*ch*3*3 + i*3*3 + 7];
            sum += input[start + y_stride + w + xi + 1] * weight[fi*ch*3*3 + i*3*3 + 8];
        }
        // relu
        if (sum<0.0){
            output[bi*w*h*filter + w*h*fi + yi*w+xi] = 0.0;
        }else{
            output[bi*w*h*filter + w*h*fi + yi*w+xi] = sum;
        }
    }
};

__kernel void conv3d_batch(
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
    int image_stride = w * h;
    int batch_stride = image_stride * ch;
    int ofset_input = batch_stride * bi;
    int output_stride = w * h * ch * filter;
    int offset_out = output_stride * bi + yi * w + xi;
    int filetr_size = 3*3*ch;
    int offset_w = 0;
    float sum = 0.0;
    float a[9];
    int x = xi - 1;
    int y = yi - 1;
    //
    for (int f=0;f<filter;f++){
        //sum = 0.0;
        offset_w = filetr_size * f;
        for (int c=0;c<ch;c++){
            int offset = ofset_input + image_stride * c*3*3;
            if (yi==0){ // on the top
                if (xi==0){ // top left corner
                    a[0] = 0.0;
                    a[1] = 0.0;
                    a[2] = 0.0;
                    a[3] = 0.0;
                    a[4] = input[offset+w*(y+1)+x+1] * weight[offset_w+4];
                    a[5] = input[offset+w*(y+1)+x+2] * weight[offset_w+5];
                    a[6] = 0;
                    a[7] = input[offset+w*(y+2)+x+1] * weight[offset_w+7];
                    a[8] = input[offset+w*(y+2)+x+2] * weight[offset_w+8];
                }else if (xi==w-1){ // top right corner
                    a[0] = 0;
                    a[1] = 0;
                    a[2] = 0;
                    a[3] = input[offset+w*(y+1)+x] * weight[offset_w+3];
                    a[4] = input[offset+w*(y+1)+x+1] * weight[offset_w+4];
                    a[5] = 0;
                    a[6] = input[offset+w*(y+2)+x] * weight[offset_w+6];
                    a[7] = input[offset+w*(y+2)+x+1] * weight[offset_w+7];
                    a[8] = 0;
                }else{ // top line
                    a[0] = 0;
                    a[1] = 0;
                    a[2] = 0;
                    a[3] = input[offset+w*(y+1)+x] * weight[offset_w+3];
                    a[4] = input[offset+w*(y+1)+x+1] * weight[offset_w+4];
                    a[5] = input[offset+w*(y+1)+x+2] * weight[offset_w+5];
                    a[6] = input[offset+w*(y+2)+x] * weight[offset_w+6];
                    a[7] = input[offset+w*(y+2)+x+1] * weight[offset_w+7];
                    a[8] = input[offset+w*(y+2)+x+2] * weight[offset_w+8];
                }
            }else if (yi==h-1){ //on the bottom
                if (xi==0){ // bottom left corner
                    a[0] = 0;
                    a[1] = input[offset+w*y+x+1] * weight[offset_w+1];
                    a[2] = input[offset+w*y+x+2] * weight[offset_w+2];
                    a[3] = 0;
                    a[4] = input[offset+w*(y+1)+x+1] * weight[offset_w+4];
                    a[5] = input[offset+w*(y+1)+x+2] * weight[offset_w+5];
                    a[6] = 0;
                    a[7] = 0;
                    a[8] = 0;
                }else if (xi==w-1){ // bottom right corner
                    a[0] = input[offset+w*y+x] * weight[offset_w+0];
                    a[1] = input[offset+w*y+x+1] * weight[offset_w+1];
                    a[2] = 0;
                    a[3] = input[offset+w*(y+1)+x] * weight[offset_w+3];
                    a[4] = input[offset+w*(y+1)+x+1] * weight[offset_w+4];
                    a[5] = 0;
                    a[6] = 0;
                    a[7] = 0;
                    a[8] = 0;
                }else{ // bottom line
                    a[0] = input[offset+w*y+x] * weight[offset_w+0];
                    a[1] = input[offset+w*y+x+1] * weight[offset_w+1];
                    a[2] = input[offset+w*y+x+2] * weight[offset_w+2];
                    a[3] = input[offset+w*(y+1)+x] * weight[offset_w+3];
                    a[4] = input[offset+w*(y+1)+x+1] * weight[offset_w+4];
                    a[5] = input[offset+w*(y+1)+x+2] * weight[offset_w+5];
                    a[6] = 0;
                    a[7] = 0;
                    a[8] = 0;
                }
            }else{ // in the middle
                a[0] = input[offset+w*y+x] * weight[offset_w+0];
                a[1] = input[offset+w*y+x+1] * weight[offset_w+1];
                a[2] = input[offset+w*y+x+2] * weight[offset_w+2];
                a[3] = input[offset+w*(y+1)+x] * weight[offset_w+3];
                a[4] = input[offset+w*(y+1)+x+1] * weight[offset_w+4];
                a[5] = input[offset+w*(y+1)+x+2] * weight[offset_w+5];
                a[6] = input[offset+w*(y+2)+x] * weight[offset_w+6];
                a[7] = input[offset+w*(y+2)+x+1] * weight[offset_w+7];
                a[8] = input[offset+w*(y+2)+x+2] * weight[offset_w+8];
            }
            sum = 0.0;
            for (int i=0;i<9;i++){
                sum += a[i];
            }
            //
            // relu
            //
            if (sum<0.0){
                output[offset_out + image_stride * ch * f + image_stride * c] = 0.0;
            }else{
                output[offset_out + image_stride * ch * f + image_stride * c] = sum;
            }
            //printf(\"%f\\n\", sum);
        } // ch loop
    } // filter loop
};

__kernel void conv2d_batch_alt(
    __global float* input,
    __global float* weight,
    __global float* output,
    const int w,
    const int h,
    const int ch,
    const int filter,
    const int ni,
    const int ii,
    const float alt)
{
    int bi = get_global_id(0);
    int xi = get_global_id(1);
    int yi = get_global_id(2);
    int image_stride = w * h;
    int batch_stride = image_stride*ch;
    int output_stride = w * h * filter;
    int ofset_input = batch_stride * bi;
    int offset_out = output_stride * bi + yi*w + xi;
    int filetr_size = 9;
    int offset_w = 0;//filetr_size * (filter-1);
    float sum = 0.0;
    float a[9];
    int x = xi - 1;
    int y = yi - 1;
    float backup = 0.0;
    //
    for (int f=0;f<filter;f++){
        sum = 0.0;
        offset_w = filetr_size * f;
        if (f==ni){
            backup = weight[offset_w+ii];
            weight[offset_w+ii] = alt;
        }
        for (int c=0;c<ch;c++){
            int offset = ofset_input+image_stride*c;
            if (yi==0){ // on the top
                if (xi==0){ // top left corner
                    a[0] = 0.0;
                    a[1] = 0.0;
                    a[2] = 0.0;
                    a[3] = 0.0;
                    a[4] = input[offset+w*(y+1)+x+1] * weight[offset_w+4];
                    a[5] = input[offset+w*(y+1)+x+2] * weight[offset_w+5];
                    a[6] = 0;
                    a[7] = input[offset+w*(y+2)+x+1] * weight[offset_w+7];
                    a[8] = input[offset+w*(y+2)+x+2] * weight[offset_w+8];
                }else if (xi==w-1){ // top right corner
                    a[0] = 0;
                    a[1] = 0;
                    a[2] = 0;
                    a[3] = input[offset+w*(y+1)+x] * weight[offset_w+3];
                    a[4] = input[offset+w*(y+1)+x+1] * weight[offset_w+4];
                    a[5] = 0;
                    a[6] = input[offset+w*(y+2)+x] * weight[offset_w+6];
                    a[7] = input[offset+w*(y+2)+x+1] * weight[offset_w+7];
                    a[8] = 0;
                }else{ // top line
                    a[0] = 0;
                    a[1] = 0;
                    a[2] = 0;
                    a[3] = input[offset+w*(y+1)+x] * weight[offset_w+3];
                    a[4] = input[offset+w*(y+1)+x+1] * weight[offset_w+4];
                    a[5] = input[offset+w*(y+1)+x+2] * weight[offset_w+5];
                    a[6] = input[offset+w*(y+2)+x] * weight[offset_w+6];
                    a[7] = input[offset+w*(y+2)+x+1] * weight[offset_w+7];
                    a[8] = input[offset+w*(y+2)+x+2] * weight[offset_w+8];
                }
            }else if (yi==h-1){ //on the bottom
                if (xi==0){ // bottom left corner
                    a[0] = 0;
                    a[1] = input[offset+w*y+x+1] * weight[offset_w+1];
                    a[2] = input[offset+w*y+x+2] * weight[offset_w+2];
                    a[3] = 0;
                    a[4] = input[offset+w*(y+1)+x+1] * weight[offset_w+4];
                    a[5] = input[offset+w*(y+1)+x+2] * weight[offset_w+5];
                    a[6] = 0;
                    a[7] = 0;
                    a[8] = 0;
                }else if (xi==w-1){ // bottom right corner
                    a[0] = input[offset+w*y+x] * weight[offset_w];
                    a[1] = input[offset+w*y+x+1] * weight[offset_w+1];
                    a[2] = 0;
                    a[3] = input[offset+w*(y+1)+x] * weight[offset_w+3];
                    a[4] = input[offset+w*(y+1)+x+1] * weight[offset_w+4];
                    a[5] = 0;
                    a[6] = 0;
                    a[7] = 0;
                    a[8] = 0;
                }else{ // bottom line
                    a[0] = input[offset+w*y+x] * weight[offset_w];
                    a[1] = input[offset+w*y+x+1] * weight[offset_w+1];
                    a[2] = input[offset+w*y+x+2] * weight[offset_w+2];
                    a[3] = input[offset+w*(y+1)+x] * weight[offset_w+3];
                    a[4] = input[offset+w*(y+1)+x+1] * weight[offset_w+4];
                    a[5] = input[offset+w*(y+1)+x+2] * weight[offset_w+5];
                    a[6] = 0;
                    a[7] = 0;
                    a[8] = 0;
                }
            }else{ // in the middle
                a[0] = input[offset+w*y+x] * weight[offset_w];
                a[1] = input[offset+w*y+x+1] * weight[offset_w+1];
                a[2] = input[offset+w*y+x+2] * weight[offset_w+2];
                a[3] = input[offset+w*(y+1)+x] * weight[offset_w+3];
                a[4] = input[offset+w*(y+1)+x+1] * weight[offset_w+4];
                a[5] = input[offset+w*(y+1)+x+2] * weight[offset_w+5];
                a[6] = input[offset+w*(y+2)+x] * weight[offset_w+6];
                a[7] = input[offset+w*(y+2)+x+1] * weight[offset_w+7];
                a[8] = input[offset+w*(y+2)+x+2] * weight[offset_w+8];
            }
            for (int i=0;i<9;i++){
                sum += a[i];
            }
        } // ch loop
        //
        // relu
        //
        if (sum<0.0){
            output[offset_out+image_stride*f] = 0.0;
        }else{
            output[offset_out+image_stride*f] = sum;
        }
        if (f==ni){
            weight[offset_w+ii] = backup;
        }
    } // filter loop
};

__kernel void conv2d_batch(
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
    int image_stride = w * h;
    int batch_stride = image_stride*ch;
    int output_stride = w * h * filter;
    int ofset_input = batch_stride * bi;
    int offset_out = output_stride * bi + yi*w + xi;
    int filetr_size = 9;
    int offset_w = filetr_size * (filter-1);
    float sum = 0.0;
    float a[9];
    int x = xi - 1;
    int y = yi - 1;
    //
    
    for (int f=0;f<filter;f++){
        sum = 0.0;
        offset_w = filetr_size * f;
        for (int c=0;c<ch;c++){
            int offset = ofset_input + image_stride * c;
            if (yi==0){ // on the top
                if (xi==0){ // top left corner
                    a[0] = 0.0;
                    a[1] = 0.0;
                    a[2] = 0.0;
                    a[3] = 0.0;
                    a[4] = input[offset+w*(y+1)+x+1] * weight[offset_w+4];
                    a[5] = input[offset+w*(y+1)+x+2] * weight[offset_w+5];
                    a[6] = 0;
                    a[7] = input[offset+w*(y+2)+x+1] * weight[offset_w+7];
                    a[8] = input[offset+w*(y+2)+x+2] * weight[offset_w+8];
                }else if (xi==w-1){ // top right corner
                    a[0] = 0;
                    a[1] = 0;
                    a[2] = 0;
                    a[3] = input[offset+w*(y+1)+x] * weight[offset_w+3];
                    a[4] = input[offset+w*(y+1)+x+1] * weight[offset_w+4];
                    a[5] = 0;
                    a[6] = input[offset+w*(y+2)+x] * weight[offset_w+6];
                    a[7] = input[offset+w*(y+2)+x+1] * weight[offset_w+7];
                    a[8] = 0;
                }else{ // top line
                    a[0] = 0;
                    a[1] = 0;
                    a[2] = 0;
                    a[3] = input[offset+w*(y+1)+x] * weight[offset_w+3];
                    a[4] = input[offset+w*(y+1)+x+1] * weight[offset_w+4];
                    a[5] = input[offset+w*(y+1)+x+2] * weight[offset_w+5];
                    a[6] = input[offset+w*(y+2)+x] * weight[offset_w+6];
                    a[7] = input[offset+w*(y+2)+x+1] * weight[offset_w+7];
                    a[8] = input[offset+w*(y+2)+x+2] * weight[offset_w+8];
                }
            }else if (yi==h-1){ //on the bottom
                if (xi==0){ // bottom left corner
                    a[0] = 0;
                    a[1] = input[offset+w*y+x+1] * weight[offset_w+1];
                    a[2] = input[offset+w*y+x+2] * weight[offset_w+2];
                    a[3] = 0;
                    a[4] = input[offset+w*(y+1)+x+1] * weight[offset_w+4];
                    a[5] = input[offset+w*(y+1)+x+2] * weight[offset_w+5];
                    a[6] = 0;
                    a[7] = 0;
                    a[8] = 0;
                }else if (xi==w-1){ // bottom right corner
                    a[0] = input[offset+w*y+x] * weight[offset_w+0];
                    a[1] = input[offset+w*y+x+1] * weight[offset_w+1];
                    a[2] = 0;
                    a[3] = input[offset+w*(y+1)+x] * weight[offset_w+3];
                    a[4] = input[offset+w*(y+1)+x+1] * weight[offset_w+4];
                    a[5] = 0;
                    a[6] = 0;
                    a[7] = 0;
                    a[8] = 0;
                }else{ // bottom line
                    a[0] = input[offset+w*y+x] * weight[offset_w+0];
                    a[1] = input[offset+w*y+x+1] * weight[offset_w+1];
                    a[2] = input[offset+w*y+x+2] * weight[offset_w+2];
                    a[3] = input[offset+w*(y+1)+x] * weight[offset_w+3];
                    a[4] = input[offset+w*(y+1)+x+1] * weight[offset_w+4];
                    a[5] = input[offset+w*(y+1)+x+2] * weight[offset_w+5];
                    a[6] = 0;
                    a[7] = 0;
                    a[8] = 0;
                }
            }else{ // in the middle
                a[0] = input[offset+w*y+x] * weight[offset_w+0];
                a[1] = input[offset+w*y+x+1] * weight[offset_w+1];
                a[2] = input[offset+w*y+x+2] * weight[offset_w+2];
                a[3] = input[offset+w*(y+1)+x] * weight[offset_w+3];
                a[4] = input[offset+w*(y+1)+x+1] * weight[offset_w+4];
                a[5] = input[offset+w*(y+1)+x+2] * weight[offset_w+5];
                a[6] = input[offset+w*(y+2)+x] * weight[offset_w+6];
                a[7] = input[offset+w*(y+2)+x+1] * weight[offset_w+7];
                a[8] = input[offset+w*(y+2)+x+2] * weight[offset_w+8];
            }
            for (int i=0;i<9;i++){
                sum += a[i];
            }
        } // ch loop
        //
        // relu
        //
        if (sum<0.0){
            output[offset_out+image_stride*f] = 0.0;
        }else{
            output[offset_out+image_stride*f] = sum;

        }
    } // filter loop
};

__kernel void max_batch(
    __global float* input,
    __global float* output,
    const int ch,
    const int w, // output w
    const int h,
    const int batch_stride)
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
    //int k = 0;
    //int offset_out = w * h * bi;
    //int k = ofset_input + y*w*2 + x*2;
    
    
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

__kernel void conv_batch_alt(
    __global float* input,
    __global int* map,
    __global float* w,
    __global float* output,
    const int ksize,
    const int kx,
    const int ky,
    const int batch_stride,
    const int alt_x,
    const int alt_y,
    const int alt_i,
    const float alt_w)
{
    int bi = get_global_id(0);
    int kix = get_global_id(1);
    int kiy = get_global_id(2);
    
    int ofset_input = batch_stride*bi;
    int offset_map = (kx*kiy + kix)*ksize;
    int offset_out = kx*ky*bi + kx*kiy + kix;
    float sum = 0.0;
    
    if (alt_x==kix && alt_y==kiy){
        for (int i=0;i<ksize;i++){
            int k = map[offset_map+i];
            if (i==alt_i){
                sum += (input[ofset_input+k] * alt_w);
            }else{
                sum += (input[ofset_input+k] * w[offset_map+i]);
            }
        }
    }else{
        for (int i=0;i<ksize;i++){
            int k = map[offset_map+i];
            sum += (input[ofset_input+k] * w[offset_map+i]);
        }
    }
    // relu
    if (sum<0.0){
        output[offset_out] = 0.0;
    }else{
        output[offset_out] = sum;
    }
}

__kernel void conv_batch(
    __global float* input,
    __global int* map,
    __global float* w,
    __global float* output,
    const int ksize,
    const int kx,
    const int ky,
    const int batch_stride)
{
    int bi = get_global_id(0);
    int kix = get_global_id(1);
    int kiy = get_global_id(2);
    int ofset_input = batch_stride*bi;
    int offset_map = (kx*kiy + kix)*ksize;
    int offset_out = kx*ky*bi + kx*kiy + kix;
    float sum = 0.0;
    
    for (int i=0;i<ksize;i++){
        int k = map[offset_map+i];
        sum += (input[ofset_input+k] * w[offset_map+i]);
    }

    // relu
    if (sum<0.0){
        output[offset_out] = 0.0;
    }else{
        output[offset_out] = sum;
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

__kernel void normalize_batch_cnn(__global float* data,
                                  int batch_size, int batch_stride,
                                  int filter_num, int image_size)
{
    int fi = get_global_id(0);

    float sum = 0.0;
    float avg = 0.0;
    float delta = 0.0000001;
    float div2 = 0.0;
    float div = 0.0;

    for (int bi=0; bi<batch_size; bi++){
        for (int i=0; i<image_size; i++){
            sum += data[bi*batch_stride + fi*image_size + i];
        }
    }
    avg = sum / batch_size;
    avg = avg / image_size;
    
    sum = 0.0;
    for (int bi=0; bi<batch_size; bi++){
        for (int i=0; i<image_size; i++){
            float k = data[bi*batch_stride + fi*image_size + i] - avg;
            sum += k * k;
        }
    }
    div2 = sum / batch_size;
    div2 = div2 / image_size + delta;
    div =  sqrt(div2);
    
    for (int bi=0; bi<batch_size; bi++){
        for (int i=0; i<image_size; i++){
            float k = data[bi*batch_stride + fi*image_size + i] - avg;
            data[bi*batch_stride + fi*image_size + i] = k / div;
        }
    }
}

__kernel void normalize_batch_2(__global float* data, int size, int batch_size)
{
    int i = get_global_id(0);
    float sum = 0.0;
    float avg = 0.0;
    float delta = 0.0000001;
    float div2 = 0.0;
    float div = 0.0;

    for (int bi=0;bi<batch_size;bi++){
        sum += data[bi*size+i];
    }
    avg = sum / batch_size;
    
    sum = 0.0;
    for (int bi=0;bi<batch_size;bi++){
        float k = data[bi*size+i] - avg;
        sum += k * k;
    
    }
    div2 = sum / batch_size + delta;
    div =  sqrt(div2);
    
    for (int bi=0;bi<batch_size;bi++){
        float k = data[bi*size+i] - avg;
        data[bi*size+i] = k / div;
    }
}

__kernel void normalize_batch(__global float* in, int bsize, int num_batch)
{
    int i = get_global_id(0);
    float max = 0.0;
    float delta = 0.0000001;
    
    for (int bi=0;bi<num_batch;bi++){
        if (in[bi*bsize+i]>max){
            max = in[bi*bsize+i];
        }
    }
    
    for (int bi=0;bi<num_batch;bi++){
        in[bi*bsize+i] = (in[bi*bsize+i]/(max+delta));
    }
}

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

__kernel void scale_cnn(__global float* data, int batch_stride, int filter_stride)
{
    int bi = get_global_id(0);
    int fi = get_global_id(1);
    float max = 0.0;
    int offset = bi * batch_stride + fi * filter_stride;
    
    for (int i=0;i<filter_stride;i++){
        if (data[offset+i]>max){
            max = data[offset+i];
        }
    }
    
    for (int i=0;i<filter_stride;i++){
        data[offset+i] = (data[offset+i]/max);
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
        data[bi*size+i] = (data[bi*size+i]/max);
    }
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

// stride_1 : num_node * num_input
// stride_2 : num_input

__kernel void multiple_x_by_w_batch_alt(
    __global const float* x,
    __global const float* w,
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
    
//    float temp = w[stride_2*alt_ni + alt_ii];
//    w[stride_2*alt_ni + alt_ii] = alt_w;
    if (j==alt_ni && i==alt_ii){
        y[stride_1*bi + stride_2*j + i] = x[stride_2*bi + i] * alt_w;
    }else{
        y[stride_1*bi + stride_2*j + i] = x[stride_2*bi + i] * w[stride_2*j + i];
    }
//    y[stride_1*bi + stride_2*j + i] = x[stride_2*bi + i] * w[stride_2*j + i];
//    w[stride_2*alt_ni + alt_ii] = temp;
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
        
    def scale_cnn(self, data, batch_size, filter_size, batch_stride, filter_stride):
        event = self.prg.scale_cnn(self._queue, (batch_size, filter_size), None,
                                   data, np.int32(batch_stride), np.int32(filter_stride))
        event.wait()
    
    def normalize_layer(self, data, size, batch_size):
        event = self.prg.normalize_layer(self._queue, (batch_size,), None, data, np.int32(size))
        event.wait()

    def normalize_batch(self, data, size, batch_size):
        event = self.prg.normalize_batch_2(self._queue, (size,), None, data, np.int32(size), np.int32(batch_size))
        event.wait()
  
    def normalize_batch_cnn(self, data, batch_size, batch_stride, filter_num, image_size):
        event = self.prg.normalize_batch_cnn(self._queue, (filter_num,), None, data,
                                           np.int32(batch_size), np.int32(batch_stride),
                                           np.int32(filter_num), np.int32(image_size))
        event.wait()
    
    def softmax(self, data, size, num_batch):
        event = self.prg.p_softmax(self._queue, (num_batch,), None, data, np.int32(size))
        event.wait()
    
    def k_cross_entropy(self, infs, output, labels, size, num_batch):
        event = self.prg.k_cross_entropy(self._queue, (num_batch,), None,
                                        infs, output, labels, np.int32(size))
        event.wait()
        
    def conv_batch(self, input, map, w, output, ksize, kx, ky, num_batch, batch_stride):
        event = self.prg.conv_batch(self._queue, (num_batch, kx, ky), None,
                                    input, map, w, output,
                                    np.int32(ksize), np.int32(kx), np.int32(ky), np.int32(batch_stride))
        event.wait()

    def conv_batch_alt(self, input, map, w, output, ksize, kx, ky, num_batch, batch_stride, alt_x, alt_y, alt_i, alt_w):
        event = self.prg.conv_batch_alt(self._queue, (num_batch, kx, ky), None,
                                    input, map, w, output,
                                    np.int32(ksize), np.int32(kx), np.int32(ky), np.int32(batch_stride),
                                    np.int32(alt_x), np.int32(alt_y), np.int32(alt_i), np.float32(alt_w))
        event.wait()

    def max_batch(self, input, output, ch, w, h, num_batch, batch_stride):
        event = self.prg.max_batch(self._queue, (num_batch, w, h), None,
                                   input, output, np.int32(ch), np.int32(w), np.int32(h), np.int32(batch_stride))
        event.wait()
        
    def conv2d_batch(self, input, weight, output, w, h, ch, filter, batch_size):
        event = self.prg.conv2d_batch(self._queue, (batch_size, w, h), None,
                                      input, weight, output, np.int32(w), np.int32(h),
                                      np.int32(ch), np.int32(filter))
        event.wait()
        
    def conv2d_batch_alt(self, input, weight, output, w, h, ch, filter, batch_size, ni, ii, alt):
        event = self.prg.conv2d_batch_alt(self._queue, (batch_size, w, h), None,
                                          input, weight, output, np.int32(w), np.int32(h),
                                          np.int32(ch), np.int32(filter),
                                          np.int32(ni), np.int32(ii), np.float32(alt))
        event.wait()
        
    def conv3d_batch(self, input, weight, output, w, h, ch, filter, batch_size):
        event = self.prg.conv3d_batch(self._queue, (batch_size, w, h), None,
                                      input, weight, output, np.int32(w), np.int32(h),
                                      np.int32(ch), np.int32(filter))
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

    print(data_x)
    print(data_w)
    print(data_y)
    
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
    
    print("num=%d, stride=%d, left=%d" % (num_input, stride, left))
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
    print(">> start")
    sts = main()
    print(">> end")
    print("\007")
    sys.exit(sts)

