double f(double x){
    return (3*x*x + 2*x + 1);
}
__kernel void integral(int N, __global double *g_sum, __local double *l_sum)
{
  int i = get_global_id(0);
  int l_i = get_local_id(0);

  double dx = (1000.0 / (double)N);

  l_sum[l_i] = (i < N) ? f(i * dx) * dx : 0;
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int p = get_local_size(0) / 2; p >= 1; p = p >> 1) {
    if (l_i < p) l_sum[l_i] += l_sum[l_i + p];
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (l_i == 0) {
    g_sum[get_group_id(0)] = l_sum[0];
  }
}


