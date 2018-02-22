#include <stdio.h>
#include <stdlib.h>

extern int N;

__global__ void gpuReduction(int *g_num, 
                             int *g_sum, 
                             int TotalNum) {
  // TODO: implement kernel code here
}

double reduction_cuda(int *array, int N) {
  // TODO: implement host code here
  return 0;
}
