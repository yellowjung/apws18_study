__kernel void img_rotate(__global float *dest_data, __global float *src_data,
                        int W, int H,
                        float sinTheta, float cosTheta)
{
    int dest_x = get_global_id(0);
    int dest_y = get_global_id(1);

    float x0 = W / 2.0f;
    float y0 = H / 2.0f;
    if(dest_x >= W || dest_y >= H){
        return;
    }
    float xOff = dest_x - x0;
    float yOff = dest_y - y0;
    int src_x = (int)(xOff * cosTheta + yOff * sinTheta + x0);
    int src_y = (int)(yOff * cosTheta - xOff * sinTheta + y0);
    if((src_x >= 0) && (src_x < W) && (src_y >= 0) && (src_y < H)){
        dest_data[dest_y * W + dest_x] = src_data[src_y * W + src_x];
    }else{
        dest_data[dest_y * W + dest_x] = 0.0f;
    }


}
