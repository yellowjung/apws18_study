#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <sys/time.h>
#include <unistd.h>

static int N = 16777216;

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)1e-6 * tv.tv_usec;
}

double reduction_seq(int *array, int N);
double reduction_opencl(int *array, int N);

int main() {
  int *array = (int*)malloc(sizeof(int) * N);
  int i;
  double ans_seq, ans_opencl;

  for (i = 0; i < N; i++) {
    array[i] = rand() % 100;
  }

  printf("Sequential version...\n");
  ans_seq = reduction_seq(array, N);
  printf("Average: %f\n", ans_seq);

  printf("OpenCL version...\n");
  ans_opencl = reduction_opencl(array, N);
  printf("Average: %f\n", ans_opencl);

  free(array);
  return 0;
}

double reduction_seq(int *array, int N) {
  int sum = 0;
  int i;
  double start_time, end_time;
  start_time = get_time();
  for (i = 0; i < N; i++) {
    sum += array[i];
  }
  end_time = get_time();
  printf("Elapsed time: %f sec\n", end_time - start_time);
  return (double)sum / N;
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

double reduction_opencl(int *array, int N) {
  // TODO
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;

    int *result;
    int total = 0;
    size_t source_size;
    char *source_code;
    int loop = 0;
    //get platform id
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err);

    //get Device id
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err);

    //Create context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);

    //Create commandqueue
    queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_ERROR(err);

    //Create object
    source_code = get_source_code("kernel.cl", &source_size);
    program = clCreateProgramWithSource(context, 1,
            (const char**)&source_code, &source_size, &err);
    CHECK_ERROR(err);

    //program build
    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    if(err == CL_BUILD_PROGRAM_FAILURE){
        size_t log_size;
        char *log;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        log = (char *)malloc(log_size + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = '\0';
        printf("%s", log);
        free(log);
    }

    //create kernel
    kernel = clCreateKernel(program, "reduction", &err);
    CHECK_ERROR(err);

    //set global, local size
    size_t global_size = N;
    size_t local_size = 256;
    size_t work_group_num = global_size / local_size;

    result = (int *)malloc(sizeof(int) * work_group_num);
    //create buffer
    cl_mem g_num, g_sum;

    g_num = clCreateBuffer(context, CL_MEM_READ_ONLY,
            sizeof(int) * N, NULL, &err);
    CHECK_ERROR(err);
    g_sum = clCreateBuffer(context, CL_MEM_READ_WRITE,
            sizeof(int) * work_group_num, NULL, &err);
    CHECK_ERROR(err);

    //Write buffer
    err = clEnqueueWriteBuffer(queue, g_num, CL_TRUE, 0,
            sizeof(int) * N, array, 0, NULL, NULL);

    //Check time
    double start_time, end_time;
    start_time = get_time(); 
    //input kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &g_num);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &g_sum);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(int) * local_size, NULL);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), &N);

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    CHECK_ERROR(err);

    err = clEnqueueReadBuffer(queue, g_sum, CL_TRUE, 0,
            sizeof(int) * work_group_num, result, 0, NULL, NULL);
    CHECK_ERROR(err);
    for(loop = 0; loop < work_group_num; loop++){
        total += result[loop];
    }

    end_time = get_time();

    clReleaseMemObject(g_num);
    clReleaseMemObject(g_sum);
    free(result);
    clReleaseContext(context);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    printf("Elapsed time: %f sec\n", end_time - start_time);
    return (double)total / N;
}
