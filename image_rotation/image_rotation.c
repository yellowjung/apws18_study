#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.h>
#include <sys/time.h>
#include "bmpfuncs.h"

static float theta = 3.14159/6;

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)1e-6 * tv.tv_usec;
}

void rotate(float *input_image, float *output_image, int image_width, int image_height,
            float sin_theta, float cos_theta);

int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("Usage: %s <src file> <dest file>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  float sin_theta = sinf(theta);
  float cos_theta = cosf(theta);

  int image_width, image_height;
  float *input_image = readImage(argv[1], &image_width, &image_height);
  float *output_image = (float*)malloc(sizeof(float) * image_width * image_height);
  rotate(input_image, output_image, image_width, image_height, sin_theta, cos_theta);
  storeImage(output_image, argv[2], image_height, image_width, argv[1]);
  return 0;
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

void rotate(float *input_image, float *output_image, int image_width, int image_height,
            float sin_theta, float cos_theta) {
  // TODO
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_int err;

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
    kernel = clCreateKernel(program, "img_rotate", &err);
    CHECK_ERROR(err);

    //create buffer
    cl_mem b_input;
    cl_mem b_output;

    b_input = clCreateBuffer(context, CL_MEM_READ_ONLY,
            sizeof(float) * image_width * image_height, NULL, &err);
    CHECK_ERROR(err);
    b_output = clCreateBuffer(context, CL_MEM_READ_WRITE,
            sizeof(float) * image_width * image_height, NULL, &err);
    CHECK_ERROR(err);

    //Write buffer
    err =  clEnqueueWriteBuffer(queue, b_input, CL_TRUE, 0,
            sizeof(float) * image_width * image_height, input_image, 0, NULL, NULL);
    CHECK_ERROR(err);

    //input kernel argumanets
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &b_output);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_input);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_int), &image_width);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_int), &image_height);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 4, sizeof(cl_float), &sin_theta);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 5, sizeof(cl_float), &cos_theta);
    CHECK_ERROR(err);

    //set global, local size
    size_t global_size[2] = {image_width, image_height};
    size_t local_size[2] = {32, 32};
    global_size[0] = (global_size[0] + local_size[0] - 1) / local_size[0] * local_size[0];
    global_size[1] = (global_size[1] + local_size[1] - 1) / local_size[1] * local_size[1];

    double start_time = get_time();
    //run kernel
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);

    double end_time = get_time();

    err = clEnqueueReadBuffer(queue, b_output, CL_TRUE, 0,
            sizeof(float) * image_width * image_height, output_image, 0, NULL, NULL);
    CHECK_ERROR(err);
    printf("Elapsed time: %f sec\n", end_time - start_time);
    clReleaseContext(context);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseKernel(kernel);

}
