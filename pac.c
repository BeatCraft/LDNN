

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
}



int main(){
    float* x;
    float* w;
    float* y;
    const int stride_1;
    const int stride_2;

    for (int bi=0;bi<bsize;bi++){
        for (int j=0;j<num_node;j++){
            for (int i=0;i<num_input;i++){
                y[stride_1*bi + stride_2*j+i] = x[stride_2*bi+i] * w[stride_2*j+i];
            }
        }
    }
}
