from __future__ import division

KERNEL_CODE = """

// Thread block size
#define BLOCK_SIZE %(block_size)d

// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WA %(w_a)d // Matrix A width
#define HA %(h_a)d // Matrix A height
#define WB %(w_b)d // Matrix B width
#define HB WA      // Matrix B height
#define WC WB      // Matrix C width
#define HC HA      // Matrix C height


/* Matrix multiplication: C = A * B.
 * Device code.
 */

#define AS(j, i) As[i + j * BLOCK_SIZE]
#define BS(j, i) Bs[i + j * BLOCK_SIZE]

////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! WA is A's width and WB is B's width
////////////////////////////////////////////////////////////////////////////////
//__kernel __attribute__((reqd_work_group_size(16,16,1))) 
__kernel __attribute__((reqd_work_group_size(1,1,1))) 
void
matrixMul( __global float* C, __global float* A, __global float* B)
{
    __local float As[BLOCK_SIZE*BLOCK_SIZE];
    __local float Bs[BLOCK_SIZE*BLOCK_SIZE];

    // Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);

    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);

    // Index of the first sub-matrix of A processed by the block
    int aBegin = WA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + WA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * WB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0.0f;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        AS(ty, tx) = A[a + WA * ty + tx];
        BS(ty, tx) = B[b + WB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += AS(ty, k) * BS(k, tx);

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    C[get_global_id(1) * get_global_size(0) + get_global_id(0)] = Csub;
    
}

"""
#
#
#
import pyopencl as cl
import os, sys, time
from time import time
import numpy
## warmup ----------------------------------------------------------------------
#for i in range(5):
#    event = kernel(queue, h_c.shape, (block_size, block_size), d_c_buf, d_a_buf, d_b_buf)
#    event.wait()
#
#queue.finish()


block_size = 1

#
#
#
class Gpu:
    def __init__(self):
        self._ctx = cl.create_some_context()

        for dev in self._ctx.devices:
            assert dev.local_mem_size > 0

        self._queue = cl.CommandQueue(
                self._ctx,
                properties=cl.command_queue_properties.PROFILING_ENABLE)
                
        #block_size = 1
        a_width = block_size * 196
        a_height = block_size * 1
        b_width = block_size * 196
        kernel_params = {
                "block_size": block_size,
                "w_a":a_width, "h_a":a_height, "w_b":b_width}
        prg = cl.Program(
                self._ctx,
                KERNEL_CODE % kernel_params,).build(options="-cl-mad-enable -cl-fast-relaxed-math")
        self._kernel = prg.matrixMul

    
    # transfer host -> device
    def write(self, h_a, h_b, h_c):
        mf = cl.mem_flags

        d_a_buf = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_a)
        d_b_buf = cl.Buffer(self._ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=h_b)
        d_c_buf = cl.Buffer(self._ctx, mf.WRITE_ONLY, size=h_c.nbytes)

        return d_a_buf, d_b_buf, d_c_buf
    
    def write(self, mem_flags, buf, size):
        mf = cl.mem_flags
        d_buf = cl.Buffer(self._ctx, mem_flags, hostbuf=buf)
        return d_buf
    
    # transfer device -> host
    def read(self, d_c_buf, h_c):
        cl.enqueue_copy(self._queue, d_c_buf, h_c)
        return d_c_buf, h_c
    
    def execute(self, h_c, d_c_buf, d_a_buf, d_b_buf):
        event = self._kernel(self._queue, h_c.shape, (block_size, block_size), d_c_buf, d_a_buf, d_b_buf)
        event.wait()
#
#
#
def main():
    a_width = block_size * 196
    a_height = block_size * 1
    b_width = block_size * 196
    b_height = block_size * 2
    c_width = b_width
    c_height = a_height

    assert a_width % block_size == 0
    assert a_height % block_size == 0
    assert b_width % block_size == 0
    
    h_a = numpy.random.rand(a_height, a_width).astype(numpy.float32)
    h_b = numpy.random.rand(b_height, b_width).astype(numpy.float32)
    h_c = numpy.empty((c_height, c_width)).astype(numpy.float32)
    
    g = Gpu()
    d_a_buf, d_b_buf, d_c_buf = g.write(h_a, h_b, h_c)
    g.execute(h_c, d_c_buf, d_a_buf, d_b_buf)
    
    #ret1, ret2 =
    d_c_buf, h_c = g.read(d_c_buf, h_c)
    print d_c_buf
    print h_c
    
    return 0

if __name__=='__main__':
    print ">> start"
    sts = main()
    print ">> end"
    print("\007")
    sys.exit(sts)

