#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.h>
#include <mpi.h>
#include <sys/time.h>
#include <unistd.h>

static int ROW_A = 10240;
static int COL_A = 10240;
static int COL_B = 10240;

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)1e-6 * tv.tv_usec;
}

void mat_mul(float *A, float *B, float *C,
             int ROW_A, int COL_A, int COL_B);
void verify(float *A, float *B, float *C,
            int ROW_A, int COL_A, int COL_B);

int main(int argc, char *argv[]) {
  int num_nodes;
  int rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  float *A, *B, *C;
  int i, j;
  if (rank == 0) {
    A = (float*)malloc(sizeof(float) * ROW_A * COL_A);
    B = (float*)malloc(sizeof(float) * COL_A * COL_B);
    C = (float*)malloc(sizeof(float) * ROW_A * COL_B);

    for (i = 0; i < ROW_A; i++) {
      for (j = 0; j < COL_A; j++) {
        A[i * COL_A + j] = (float)(rand() % 1000) / 100.0f;
      }
    }
    for (i = 0; i < COL_A; i++) {
      for (j = 0; j < COL_B; j++) {
        B[i * COL_B + j] = (float)(rand() % 1000) / 100.0f;
      }
    }
  } else {
    A = NULL;
    B = NULL;
    C = NULL;
  }

  if (rank == 0) {
    printf("Matrix Multiplication\n");
    printf("C[%d X %d] = A[%d X %d] X B[%d X %d]\n",
           ROW_A, COL_B, ROW_A, COL_A, COL_A, COL_B);
  }

  mat_mul(A, B, C, ROW_A, COL_A, COL_B);

  if (rank == 0) {
    //verify(A, B, C, ROW_A, COL_A, COL_B);

    free(A);
    free(B);
    free(C);
  }

  MPI_Finalize();
  return 0;
}

void verify(float *A, float *B, float *C,
            int ROW_A, int COL_A, int COL_B) {
  int i, j, k;
  float sum;

  for (i = 0; i < ROW_A; i++) {
    for (j = 0; j < COL_B; j++) {
      sum = 0.0f;
      for (k = 0; k < COL_A; k++) {
        sum += A[i * COL_A + k] * B[k * COL_B + j];
      }
      if (fabsf(C[i * COL_B + j] - sum) > 0.1) {
        printf("Verification failed! C[%d][%d]: %f vs. %f\n",
               i, j, C[i * COL_B + j], sum);
        return;
      }
    }
  }
  printf("Verification success!\n");
}

#define CHECK_ERROR(err) \
  if (err != CL_SUCCESS) { \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
  }

char *get_source_code(const char *file_name, size_t *len) {
  char *source_code;
  size_t length;
  FILE *file = fopen(file_name, "r");
  if (file == NULL) {
    printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
    exit(EXIT_FAILURE);
  }

  fseek(file, 0, SEEK_END);
  length = (size_t)ftell(file);
  rewind(file);

  source_code = (char *)malloc(length + 1);
  fread(source_code, length, 1, file);
  source_code[length] = '\0';

  fclose(file);

  *len = length;
  return source_code;
}

void mat_mul(float *A, float *B, float *C,
             int ROW_A, int COL_A, int COL_B) {
  int num_nodes;
  int rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  cl_platform_id platform;
  cl_uint num_devices;
  cl_device_id *device;
  cl_context context;
  cl_command_queue *queue;
  cl_program program;
  char *kernel_source;
  size_t kernel_source_size;
  cl_kernel *kernel;
  int i;
  cl_int err;

  err = clGetPlatformIDs(1, &platform, NULL);
  CHECK_ERROR(err);

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
  CHECK_ERROR(err);

  if (rank == 0) {
    printf("%d nodes, %u devices\n", num_nodes, num_devices);
  }

  device = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
  queue = (cl_command_queue*)malloc(sizeof(cl_command_queue) * num_devices);
  kernel = (cl_kernel*)malloc(sizeof(cl_kernel) * num_devices);

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, device, NULL);
  CHECK_ERROR(err);

  context = clCreateContext(NULL, num_devices, device, NULL, NULL, &err);
  CHECK_ERROR(err);

  for (i = 0; i < num_devices; i++) {
    queue[i] = clCreateCommandQueue(context, device[i], 0, &err);
    CHECK_ERROR(err);
  }

  kernel_source = get_source_code("kernel.cl", &kernel_source_size);
  program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source,
                                      &kernel_source_size, &err);
  CHECK_ERROR(err);

  err = clBuildProgram(program, num_devices, device, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    char *log;

    err = clGetProgramBuildInfo(program, device[0], CL_PROGRAM_BUILD_LOG,
                                0, NULL, &log_size);
    CHECK_ERROR(err);

    log = (char*)malloc(log_size + 1);
    err = clGetProgramBuildInfo(program, device[0], CL_PROGRAM_BUILD_LOG,
                                log_size, log, NULL);
    CHECK_ERROR(err);

    log[log_size] = '\0';
    printf("Compiler error:\n%s\n", log);
    free(log);
  }
  CHECK_ERROR(err);

  for (i = 0; i < num_devices; i++) {
    kernel[i] = clCreateKernel(program, "mat_mul", &err);
    CHECK_ERROR(err);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  double start_time = get_time();

  int ROW_A_PER_NODE = ROW_A / num_nodes;
  int ROW_A_PER_DEVICE = ROW_A_PER_NODE / num_devices;

  if (rank != 0) {
    A = (float*)malloc(sizeof(float) * ROW_A_PER_NODE * COL_A);
    B = (float*)malloc(sizeof(float) * COL_A * COL_B);
    C = (float*)malloc(sizeof(float) * ROW_A_PER_NODE * COL_B);
  }

  cl_mem *bufA, *bufB, *bufC;
  bufA = (cl_mem*)malloc(sizeof(cl_mem) * num_devices);
  bufB = (cl_mem*)malloc(sizeof(cl_mem) * num_devices);
  bufC = (cl_mem*)malloc(sizeof(cl_mem) * num_devices);

  for (i = 0; i < num_devices; i++) {
    bufA[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*ROW_A_PER_DEVICE*COL_A,
                             NULL, &err);
    CHECK_ERROR(err);
    bufB[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*COL_A*COL_B,
                             NULL, &err);
    CHECK_ERROR(err);
    bufC[i] = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*ROW_A_PER_DEVICE*COL_B,
                             NULL, &err);
    CHECK_ERROR(err);
  }

  if (rank == 0) {
    MPI_Request *requests;
    requests = (MPI_Request*)malloc(sizeof(MPI_Request) * (num_nodes - 1) * 2);
    for (i = 1; i < num_nodes; i++) {
      MPI_Isend(A + (ROW_A_PER_NODE * COL_A * i), ROW_A_PER_NODE * COL_A, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &requests[(i - 1) * 2]);
      MPI_Isend(B, COL_A * COL_B, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &requests[(i - 1) * 2 + 1]);
    }
    MPI_Waitall((num_nodes - 1) * 2, requests, NULL);
    free(requests);
  } else {
    MPI_Request requests[2];
    MPI_Irecv(A, ROW_A_PER_NODE * COL_A, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(B, COL_A * COL_B, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &requests[1]);
    MPI_Waitall(2, requests, NULL);
  }

  for (i = 0; i < num_devices; i++) {
    err = clEnqueueWriteBuffer(queue[i], bufA[i], CL_FALSE, 0,
                               sizeof(float)*ROW_A_PER_DEVICE*COL_A, A + (ROW_A_PER_DEVICE*COL_A*i),
                               0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue[i], bufB[i], CL_FALSE, 0,
                               sizeof(float)*COL_A*COL_B, B,
                               0, NULL, NULL);
    CHECK_ERROR(err);
  }

  for (i = 0; i < num_devices; i++) {
    err = clSetKernelArg(kernel[i], 0, sizeof(cl_mem), &bufA[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[i], 1, sizeof(cl_mem), &bufB[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[i], 2, sizeof(cl_mem), &bufC[i]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[i], 3, sizeof(cl_int), &ROW_A_PER_DEVICE);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[i], 4, sizeof(cl_int), &COL_A);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel[i], 5, sizeof(cl_int), &COL_B);
    CHECK_ERROR(err);
  }

  size_t global_size[2] = {COL_B, ROW_A_PER_DEVICE};
  size_t local_size[2] = {16, 16};
  global_size[0] = (global_size[0] + local_size[0] - 1) / local_size[0] * local_size[0];
  global_size[1] = (global_size[1] + local_size[1] - 1) / local_size[1] * local_size[1];

  for (i = 0; i < num_devices; i++) {
    err = clEnqueueNDRangeKernel(queue[i], kernel[i], 2, NULL, global_size, local_size,
                                 0, NULL, NULL);
    CHECK_ERROR(err);
  }

  for (i = 0; i < num_devices; i++) {
    err = clEnqueueReadBuffer(queue[i], bufC[i], CL_FALSE, 0,
                              sizeof(float)*ROW_A_PER_DEVICE*COL_B, C + (ROW_A_PER_DEVICE*COL_A*i),
                              0, NULL, NULL);
    CHECK_ERROR(err);
  }

  for (i = 0; i < num_devices; i++) {
    clFinish(queue[i]);
  }

  if (rank == 0) {
    MPI_Request *requests;
    requests = (MPI_Request*)malloc(sizeof(MPI_Request) * (num_nodes - 1));
    for (i = 1; i < num_nodes; i++) {
      MPI_Irecv(C + (ROW_A_PER_NODE * COL_B * i), ROW_A_PER_NODE * COL_B, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &requests[i - 1]);
    }
    MPI_Waitall(num_nodes - 1, requests, NULL);
    free(requests);
  } else {
    MPI_Request request;
    MPI_Isend(C, ROW_A_PER_NODE * COL_B, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, NULL);
  }

  double end_time = get_time();
  if (rank == 0) {
    printf("Elapsed time: %f sec\n", end_time - start_time);
  }

  for (i = 0; i < num_devices; i++) {
    clReleaseMemObject(bufA[i]);
    clReleaseMemObject(bufB[i]);
    clReleaseMemObject(bufC[i]);
  }
  free(bufA);
  free(bufB);
  free(bufC);
  if (rank != 0) {
    free(A);
    free(B);
    free(C);
  }
  for (i = 0; i < num_devices; i++) {
    clReleaseKernel(kernel[i]);
  }
  free(kernel);
  clReleaseProgram(program);
  for (i = 0; i < num_devices; i++) {
    clReleaseCommandQueue(queue[i]);
  }
  free(queue);
  clReleaseContext(context);
  free(device);
}
