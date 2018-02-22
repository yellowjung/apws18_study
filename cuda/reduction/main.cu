#include <stdio.h>
#include <stdlib.h>

int N = 8388608;

double reduction_seq(int *array, int N);
double reduction_opencl(int *array, int N);
double reduction_cuda(int *array, int N);

int main() {
  int *array = (int*)malloc(sizeof(int) * N);
  int i;
  double ans_seq, ans_opencl, ans_cuda;

  for (i = 0; i < N; i++) {
    array[i] = rand() % 100;
  }

  printf("Sequential version...\n");
  ans_seq = reduction_seq(array, N);
  printf("Average: %f\n", ans_seq);

  printf("OpenCL version...\n");
  ans_opencl = reduction_opencl(array, N);
  printf("Average: %f\n", ans_opencl);

  printf("CUDA version...\n");
  ans_cuda = reduction_cuda(array, N);
  printf("Average: %f\n", ans_cuda);

  free(array);
  return 0;
}

double reduction_seq(int *array, int N) {
  int sum = 0;
  int i;
  for (i = 0; i < N; i++) {
    sum += array[i];
  }
  return (double)sum / N;
}

