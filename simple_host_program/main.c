#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

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

int main() {
    // TODO
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel my_kernel;
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
    my_kernel = clCreateKernel(program, "my_kernel", &err);
    CHECK_ERROR(err);
    int *A = (int*)malloc(sizeof(int) * 16384);
    int *B = (int*)malloc(sizeof(int) * 16384);
    int *C = (int*)malloc(sizeof(int) * 16384);

    cl_mem bufa;
    cl_mem bufb;
    cl_mem bufc;

     for(loop = 0; loop < 16384; loop ++){
        A[loop] = 1;
    }

     for(loop = 0; loop < 16384; loop ++){
        B[loop] = 2;
    }


//    for(loop = 0; loop < 16384; loop ++){
//        printf("%d : %d\n", loop, C[loop]);
//    }

    bufa = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * 16384, NULL, &err);
    CHECK_ERROR(err);
    bufb = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * 16384, NULL, &err);
    CHECK_ERROR(err);
    bufc = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * 16384, NULL, &err);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queue, bufa, CL_TRUE, 0, sizeof(int) * 16384, A, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue, bufb, CL_TRUE, 0, sizeof(int) * 16384, B, 0, NULL, NULL);
    CHECK_ERROR(err);

    err = clSetKernelArg(my_kernel, 0, sizeof(cl_mem), &bufa);
    CHECK_ERROR(err);
    err = clSetKernelArg(my_kernel, 1, sizeof(cl_mem), &bufb);
    CHECK_ERROR(err);
    err = clSetKernelArg(my_kernel, 2, sizeof(cl_mem), &bufc);

    size_t global_size = 16384;
    size_t local_size = 256;

    clEnqueueNDRangeKernel(queue, my_kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

    err = clEnqueueReadBuffer(queue, bufc, CL_TRUE, 0, sizeof(int) * 16384, C, 0, NULL, NULL);

//    for(loop = 0; loop < 16384; loop++){
//        printf("C: %d : %d\n", loop, C[loop]);
//    }

    printf("\n");
    clReleaseContext(context);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseKernel(my_kernel);
    printf("Finished!\n");

    return 0;
}
