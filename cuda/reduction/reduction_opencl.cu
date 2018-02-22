#include <stdio.h>
#include <CL/cl.h>

extern int N;

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
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  char *kernel_source;
  size_t kernel_source_size;
  cl_kernel kernel;
  cl_int err;

  err = clGetPlatformIDs(1, &platform, NULL);
  CHECK_ERROR(err);

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  CHECK_ERROR(err);

  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERROR(err);

  queue = clCreateCommandQueue(context, device, 0, &err);
  CHECK_ERROR(err);

  kernel_source = get_source_code("kernel.cl", &kernel_source_size);
  program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source,
                                      &kernel_source_size, &err);
  CHECK_ERROR(err);

  err = clBuildProgram(program, 1, &device, "", NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE) {
    size_t log_size;
    char *log;

    err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                0, NULL, &log_size);
    CHECK_ERROR(err);

    log = (char*)malloc(log_size + 1);
    err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                log_size, log, NULL);
    CHECK_ERROR(err);

    log[log_size] = '\0';
    printf("Compiler error:\n%s\n", log);
    free(log);
    exit(0);
  }
  CHECK_ERROR(err);

  kernel = clCreateKernel(program, "reduction", &err);
  CHECK_ERROR(err);

  size_t global_size = N;
  size_t local_size = 256;
  size_t num_work_groups = global_size / local_size;

  cl_mem buf_array, buf_partial_sum;
  buf_array = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * N, NULL, &err);
  CHECK_ERROR(err);
  buf_partial_sum = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                   sizeof(int) * num_work_groups, NULL, &err);
  CHECK_ERROR(err);

  err = clEnqueueWriteBuffer(queue, buf_array, CL_FALSE, 0, sizeof(int) * N, array,
                             0, NULL, NULL);
  CHECK_ERROR(err);

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_array);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_partial_sum);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel, 2, sizeof(int) * local_size, NULL);
  CHECK_ERROR(err);
  err = clSetKernelArg(kernel, 3, sizeof(int), &N);
  CHECK_ERROR(err);

  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size,
                               0, NULL, NULL);
  CHECK_ERROR(err);

  int *partial_sum = (int*)malloc(sizeof(int) * num_work_groups);

  err = clEnqueueReadBuffer(queue, buf_partial_sum, CL_TRUE, 0,
                            sizeof(int) * num_work_groups, partial_sum, 0, NULL, NULL);
  CHECK_ERROR(err);

  int sum = 0;
  int i;
  for (i = 0; i < num_work_groups; i++) {
    sum += partial_sum[i];
  }


  clReleaseMemObject(buf_array);
  clReleaseMemObject(buf_partial_sum);
  free(partial_sum);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return (double)sum / N;
}

