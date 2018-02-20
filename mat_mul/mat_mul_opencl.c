#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

static int padding = 23;

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

double get_time(); // use the get_time() function in mat_mul.c

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

void mat_mul_opencl(float *A, float *B, float *C,
        int ROW_A, int COL_A, int COL_B) {
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
        free(log);
    }

    //create kernel
    kernel = clCreateKernel(program, "mat_mul", &err);
    CHECK_ERROR(err);

    cl_mem bufa;
    cl_mem bufb;
    cl_mem bufc;

    bufa = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * ROW_A *COL_A, NULL, &err);
    CHECK_ERROR(err);
    bufb = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * COL_A * COL_B, NULL, &err);
    CHECK_ERROR(err);
    bufc = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * ROW_A * COL_B, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, bufa, CL_TRUE, 0, sizeof(float) * ROW_A * COL_A, A, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue, bufb, CL_TRUE, 0, sizeof(float) * COL_A * COL_B ,B, 0, NULL, NULL);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufa);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufb);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufc);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(int), &ROW_A);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 4, sizeof(int), &COL_A);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 5, sizeof(int), &COL_B);
    CHECK_ERROR(err);

    size_t global_size[2] = {COL_B, ROW_A};
    size_t local_size[2] = {16, 16};
    global_size[0] = (global_size[0] + local_size[0] - 1) / local_size[0] * local_size[0];
    global_size[1] = (global_size[1] + local_size[1] - 1) / local_size[1] * local_size[1];
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL);

    err = clEnqueueReadBuffer(queue, bufc, CL_TRUE, 0, sizeof(float) * ROW_A * COL_B,
            C, 0, NULL, NULL);
    CHECK_ERROR(err);

    clReleaseContext(context);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseKernel(kernel);

}
