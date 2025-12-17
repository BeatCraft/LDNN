#! /usr/bin/python
# -*- coding: utf-8 -*-
#
import os
import sys
import numpy as np

import Metal

import gpu

MSL = r"""
#include <metal_stdlib>
using namespace metal;

kernel void double_float(device const float* a [[buffer(0)]],
                        device float*       b [[buffer(1)]],
                        uint idx [[thread_position_in_grid]]) {
    b[idx] = a[idx] * (float)2.0;
}

struct Params_mac {
    uint  xsize; // node
    uint  wsize; // input
    uint  act;
};

kernel void calc_mac_relu(
    device const float* x [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* y [[buffer(2)]],
    constant Params_mac& P [[buffer(3)]],    
    uint2 gid [[thread_position_in_grid]])
{
    uint bi = gid.x;
    uint xi = gid.y;
    
    uint x_start = P.wsize * bi;
    uint w_start = P.wsize * xi;
    uint y_start = (P.xsize * bi) + xi;
    float temp = 0.0;

    for (uint i=0;i<P.wsize;i++){
        temp += (x[x_start+i] * w[w_start+i]);
    }
    
    if (P.act==0){ // no
        y[y_start] = (float)temp;
    } else {
        if (temp>=0){
            y[y_start] = (float)temp;
        }else{
            y[y_start] = (float)0.0;
        }
    }
}

struct Params_scale {
    uint  size;
    float  scale;
};

kernel void scale_layer(
    device float* data [[buffer(0)]],
    constant Params_scale& P [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    uint bi = idx;
    uint start = bi * P.size;
    float max = 0.0;
    
    for (uint i=0;i<P.size;i++){
        float k = fabs(data[start+i]);
        if (k>max){
            max = k;
        }
    }
    
    if (max>0.0){
        max = max / P.scale;
        for (uint i=0;i<P.size;i++){
            data[start+i] = (data[start+i]/max);
        }
    }
}

struct Params_softmax {
    uint num;
    float scale;
};

kernel void softmax(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant Params_softmax& P [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    int bi = idx;
    float temp = 0.0;
    float total = 0.0;
    uint start = bi*P.num;

    for (uint i=0;i<P.num;i++){
        temp = (float)(in[start+i] / P.scale);
        //if (temp>11.0){ // fix overflow
        //    temp = 11.0;
        //}
        temp = exp(temp);
        if (isinf(temp)){
            temp = 3.402823e+38;
        }else if (isnan(temp)){
            temp = 0;
        }
        //printf("exp=%f\n", (float)temp);
        out[start+i] = (float)temp;
        total += temp;
    }

    for (uint i=0;i<P.num;i++){
        out[start+i] = out[start+i]/(float)total;
    }
}

struct Params_ce {
    uint  num;
};

kernel void cross_entropy(
    device const float* infs [[buffer(0)]],
    device const float* labels [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant Params_ce& P [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    const float eps = 1e-7f;
    float sum = 0.0f;
    const uint base = idx * P.num;
    
    for (uint i = 0; i < P.num; ++i) {
        float t = (float)labels[base + i];
        float p = (float)infs[base + i];
        p = fmax(p, eps);
        sum += t * log(p);
    }
    
    output[idx] = (float)(-sum);
}

kernel void cross_entropy16(
    device const float* infs [[buffer(0)]],
    device const float* labels [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant Params_ce& P [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    int bi = idx;
    /*
    float delta;
    float k;
    float t;
    float sum;
    
    delta = 0.0000001;
    sum = 0.0;
    
    for (uint i=0;i<P.num;i++){
        t = labels[bi*P.num + i];
        k = infs[bi*P.num + i] + delta;
        sum += t * log(k);
    }
    
    output[bi] = (-1.0)*sum;
    */
    
    const float eps = 1e-7f;
    float sum = 0.0f;

    const uint base = idx * P.num;
    for (uint i = 0; i < P.num; ++i) {
        float t = (float)labels[base + i];
        float p = (float)infs  [base + i];
        p = fmax(p, eps);
        sum += t * log(p);
    }

    output[idx] = (float)(-sum);
}

struct Params_rs {
    uint  n;
};

kernel void reduce_sum_pass(
    device const float*  in  [[buffer(0)]],
    device float*       out [[buffer(1)]],
    constant Params_rs&    P   [[buffer(2)]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  gtid [[thread_position_in_grid]],
    uint  tgsz [[threads_per_threadgroup]],
    uint  gidg [[threadgroup_position_in_grid]]
){
    threadgroup float sdata[1024];
    float v = 0.0f;
    if (gtid < P.n) v = (float)in[gtid];
    sdata[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = tgsz >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) { sdata[tid] += sdata[tid + stride]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) { out[gidg] = sdata[0]; }
}
    
"""

def ceil_div(a, b): return (a + b - 1) // b

class LMetal(gpu.Gpu):
    def __init__(self):
        super(gpu.Gpu, self).__init__()
        self.name = "Metal"
        self.type = 2
        # -1:unknown, 0:OpenCL, 1:CuPy/GDX, 2:Metal

        self.device = Metal.MTLCreateSystemDefaultDevice()
        assert self.device is not None, "No Metal device found"
        
        self.queue = self.device.newCommandQueue()
        
        #opts = Metal.MTLCompileOptions.alloc().init()
        #opts.setLanguageVersion_(Metal.MTLLanguageVersion2_1)
        self.lib, err = self.device.newLibraryWithSource_options_error_(MSL, None, None)
        if err:
            raise RuntimeError(err)
        #
        self.opts = Metal.MTLResourceOptions(Metal.MTLResourceStorageModeShared)
        self.params_buf = self.device.newBufferWithLength_options_(32, self.opts)
    
    def alloc_buf(self, nbytes):
        return self.device.newBufferWithLength_options_(nbytes, self.opts)
        
    def alloc_buf_from_array(self, abuf):
        return self.device.newBufferWithBytes_length_options_(memoryview(abuf).tobytes(), abuf.nbytes, self.opts)
        
    def write_np_to_mbuf(self, arr, mbuf):
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)
        #
        mv = mbuf.contents().as_buffer(mbuf.length())
        mv[:arr.nbytes] = memoryview(arr).tobytes()
    
    def read_mbuf_to_numpy(self, mbuf, dtype, shape=None):
        count = mbuf.length() // np.dtype(dtype).itemsize
        arr = np.frombuffer(mbuf.contents().as_buffer(mbuf.length()), dtype=dtype, count=count)
        if shape is not None:
            arr = arr.reshape(shape, order="C")
        #
        return arr
    
    def copy_mbuf_to_numpy(self, mbuf, dtype, shape=None):
        view = mtlbuffer_to_numpy_view(buf, dtype, shape)
        return view.copy()
        
    def init_test_func(self):
        fn = self.lib.newFunctionWithName_("double_float")
        self.pipe, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if err:
            raise RuntimeError(err)
        #
        
    def test_func(self, mbuf0, mbuf1, n):
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipe)
        enc.setBuffer_offset_atIndex_(mbuf0, 0, 0)
        enc.setBuffer_offset_atIndex_(mbuf1, 0, 1)

        grid = Metal.MTLSize(n, 1, 1)
        tpg = Metal.MTLSize(min(self.pipe.threadExecutionWidth(), n), 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(grid, tpg)
    
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

    def init_calc_mac_relu(self):
        fn = self.lib.newFunctionWithName_("calc_mac_relu")
        self.pipe_calc_mac_relu, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if err:
            raise RuntimeError(err)
        #
        #self.params_buf_mac = self.device.newBufferWithLength_options_(32, self.opts)
        
    def calc_mac_relu(self, batch_size, mbuf0, mbuf1, mbuf2, xsize, wsize, act):
        params = np.zeros(1, dtype=np.dtype([
            ("xsize", np.uint32),
            ("wsize", np.uint32),
            ("act", np.uint32),
            ], align=True))
            
        params["xsize"] = xsize # node
        params["wsize"] = wsize # weight
        params["act"] = act
        
        mv = self.params_buf.contents().as_buffer(self.params_buf.length())
        mv[:params.nbytes] = memoryview(params).tobytes()
        
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipe_calc_mac_relu)
        enc.setBuffer_offset_atIndex_(mbuf0, 0, 0)
        enc.setBuffer_offset_atIndex_(mbuf1, 0, 1)
        enc.setBuffer_offset_atIndex_(mbuf2, 0, 2)
        enc.setBuffer_offset_atIndex_(self.params_buf, 0, 3)
        
        grid = Metal.MTLSize(batch_size, xsize, 1)
        #tpg  = Metal.MTLSize(16, 16, 1)
        tpg  = Metal.MTLSize(min(self.pipe_calc_mac_relu.maxTotalThreadsPerThreadgroup(), 256), 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(grid, tpg)
        
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

    def init_scale_layer(self):
        fn = self.lib.newFunctionWithName_("scale_layer")
        self.pipe_scale_layer, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if err:
            raise RuntimeError(err)
        #
    
    def scale_layer(self, batch_size, size, scale, mbuf0):
        params = np.zeros(1, dtype=np.dtype([
            ("size", np.uint32),
            ("scale", np.float32),
            ], align=True))
        params["size"] = size
        params["scale"] = scale
        
        mv = self.params_buf.contents().as_buffer(self.params_buf.length())
        mv[:params.nbytes] = memoryview(params).tobytes()
        
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipe_scale_layer)
        enc.setBuffer_offset_atIndex_(mbuf0, 0, 0)
        enc.setBuffer_offset_atIndex_(self.params_buf, 0, 1)
        
        grid = Metal.MTLSize(batch_size, 1, 1)
        #tpg  = Metal.MTLSize(4096, 1, 1)
        tpg  = Metal.MTLSize(min(self.pipe_scale_layer.maxTotalThreadsPerThreadgroup(), 256), 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(grid, tpg)
        
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        
    def init_softmax(self):
        fn = self.lib.newFunctionWithName_("softmax")
        self.pipe_softmax, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if err:
            raise RuntimeError(err)
        #
    
    def softmax(self, batch_size, num, scale, mbuf0, mbuf1):
        
        params = np.zeros(1, dtype=np.dtype([
            ("num", np.uint32),
            ("scale", np.float32),
            ], align=True))
        params["num"] = num
        params["scale"] = scale
    
        mv = self.params_buf.contents().as_buffer(self.params_buf.length())
        mv[:params.nbytes] = memoryview(params).tobytes()
        
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipe_softmax)
        enc.setBuffer_offset_atIndex_(mbuf0, 0, 0)
        enc.setBuffer_offset_atIndex_(mbuf1, 0, 1)
        enc.setBuffer_offset_atIndex_(self.params_buf, 0, 2)
        
        grid = Metal.MTLSize(batch_size, 1, 1)
        #tpg  = Metal.MTLSize(4096, 1, 1)
        tpg  = Metal.MTLSize(min(self.pipe_softmax.maxTotalThreadsPerThreadgroup(), 256), 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(grid, tpg)
        
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        
    def init_cross_entropy(self):
        fn = self.lib.newFunctionWithName_("cross_entropy")
        self.pipe_cross_entropy, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if err:
            raise RuntimeError(err)
        #

    def cross_entropy(self, batch_size, num, mbuf0, mbuf1, mbuf2):
        params = np.zeros(1, dtype=np.dtype([
            ("num", np.uint32),
            ], align=True))
        params["num"] = num
    
        mv = self.params_buf.contents().as_buffer(self.params_buf.length())
        mv[:params.nbytes] = memoryview(params).tobytes()
        
        cmd = self.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self.pipe_cross_entropy)
        enc.setBuffer_offset_atIndex_(mbuf0, 0, 0)
        enc.setBuffer_offset_atIndex_(mbuf1, 0, 1)
        enc.setBuffer_offset_atIndex_(mbuf2, 0, 2)
        enc.setBuffer_offset_atIndex_(self.params_buf, 0, 3)

        grid = Metal.MTLSize(batch_size, 1, 1)
        #tpg  = Metal.MTLSize(4096, 1, 1)
        tpg  = Metal.MTLSize(min(self.pipe_cross_entropy.maxTotalThreadsPerThreadgroup(), 256), 1, 1)
        enc.dispatchThreads_threadsPerThreadgroup_(grid, tpg)
        
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()


    def init_reduce_sum_float(self):
        fn = self.lib.newFunctionWithName_("reduce_sum_float")
        self.pipe_reduce_sum_float, err = self.device.newComputePipelineStateWithFunction_error_(fn, None)
        if err:
            raise RuntimeError(err)
        #

    def reduce_sum_float(self, arr_fp16):# np.ndarray):# -> float:
        params = np.zeros(1, dtype=np.dtype([
            ("n", np.uint32),
            ], align=True))
        params["n"] = num
    
        arr = np.ascontiguousarray(arr_fp16.astype(np.float32, copy=False))
        N = arr.size
        
        in_buf  = device.newBufferWithBytes_length_options_(memoryview(arr).tobytes(), arr.nbytes, opts)
        tpg_x = min(pipe.maxTotalThreadsPerThreadgroup(), 256)  # 例: 256
        tpg   = Metal.MTLSize(tpg_x, 1, 1)
        
        blocks = ceil_div(N, tpg_x)
        partial_bytes = blocks * 4  # float32
        out_buf = self.device.newBufferWithLength_options_(partial_bytes, opts)

        #Params_dtype = np.dtype([("n", np.uint32)], align=True)
        #params = np.zeros(1, dtype=Params_dtype); params["n"] = N
        #pbuf = device.newBufferWithBytes_length_options_(memoryview(params).tobytes(), params.nbytes, opts)
        #mv = self.params_buf.contents().as_buffer(self.params_buf.length())
        #mv[:params.nbytes] = memoryview(params).tobytes()

        cur_in_is_float = True
        cur_in_buf = in_buf
        cur_len = N

        while True:
            blocks = ceil_div(cur_len, tpg_x)
            out_buf = self.device.newBufferWithLength_options_(blocks * 4, opts)

            params["n"] = cur_len
            mv = self.params_buf.contents().as_buffer(self.params_buf.length())
            memoryview(mv).cast('B')[:params.nbytes] = memoryview(params).tobytes()

            grid = Metal.MTLSize(blocks * tpg_x, 1, 1)
            cmd = queue.commandBuffer()
            enc = cmd.computeCommandEncoder()
            enc.setComputePipelineState_(self.pipe_reduce_sum_float)

            if cur_in_is_float:
                enc.setBuffer_offset_atIndex_(cur_in_buf, 0, 0)  # float* in
                enc.setBuffer_offset_atIndex_(out_buf,   0, 1)  # float* out
                enc.setBuffer_offset_atIndex_(pbuf,      0, 2)
                enc.dispatchThreads_threadsPerThreadgroup_(grid, tpg)
                enc.endEncoding()
                cmd.commit(); cmd.waitUntilCompleted()
            else:
                pass
            #
            if blocks == 1:
                res = np.frombuffer(out_buf.contents().as_buffer(out_buf.length()), dtype=np.float32, count=1)[0]
                return float(res)
            #
        #
        partial = np.frombuffer(out_buf.contents().as_buffer(out_buf.length()), dtype=np.float32, count=blocks)
        return float(partial.sum(dtype=np.float64))
        
def main():
    m = LMetal()
    m.init_test_func()
    m.init_calc_mac_relu()
    m.init_scale_layer()
    m.init_softmax()
    m.init_cross_entropy()
    
    n = 4096 # 64 * 64
    a = (np.random.uniform(size=n).astype(np.float32))
    
    a_buf = m.alloc_buf_from_array(a)
    b_buf = m.alloc_buf(a.nbytes)
    
    m.test_func(a_buf, b_buf, n)
    out = np.frombuffer(b_buf.contents().as_buffer(b_buf.length()), dtype=np.float32)
    #ok = np.allclose(out, a * 2, rtol=1e-3, atol=1e-3)
    #print("OK:", ok)
    print("in[:8] :", a[:8])
    print("out[:8]:", out[:8])
    
    
    print(type(a.nbytes))
    
    c = (np.random.uniform(size=n).astype(np.float32))
    x = m.alloc_buf_from_array(c)
    d = (np.random.uniform(size=n).astype(np.float32))
    w = m.alloc_buf_from_array(d)
    y = m.alloc_buf(n)
    #m.calc_mac_relu(x, w, y, 16, 16, 1)
    
    #out = np.frombuffer(y.contents().as_buffer(y.length()), dtype=np.float32)
    #print("y:", out[:8])
    
    print(m.device.supportsFeatureSet_(Metal.MTLFeatureSet_macOS_GPUFamily1_v3))
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

