#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <sys/time.h>
#include <unistd.h>

static int N = 536870912;

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)1e-6 * tv.tv_usec;
}

double f(double x){
    return (3 * x * x + 2 * x + 1);
}

double integral_seq(int N);
double integral_opencl(int N);

int main() {
    int i;
    double ans_seq, ans_opencl;

    printf("Sequential version...\n");
    ans_seq = integral_seq(N);
    printf("int_0^1000 f(x) dx = %f\n", ans_seq);

    printf("OpenCL version...\n");
    ans_opencl = integral_opencl(N);
    printf("int_0^1000 f(x) dx = %f\n", ans_opencl);

    return 0;
}

double integral_seq(int N) {
    double dx = (1000.0 / (double)N);
    double sum = 0;
    int i;
    double start_time, end_time;
    start_time = get_time();
    for( i = 0; i < N; i++){
        sum += f(i * dx) * dx;
    }
    end_time = get_time();
    printf("Elapsed time: %f sec\n", end_time - start_time);
    return sum;
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

double integral_opencl(int N) {
    // TODO
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue[2];
    cl_program program;
    cl_kernel kernel;
    cl_int err;

    int *result;
    size_t source_size;
    char *source_code;
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
    queue[0] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_ERROR(err);
    queue[1] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    CHECK_ERROR(err);

    //Create object
    source_code = get_source_code("c_kernel.cl", &source_size);
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
    kernel = clCreateKernel(program, "integral", &err);
    CHECK_ERROR(err);

    //Check time
    double start_time, end_time;
    start_time = get_time();

    //set global, local size
    size_t global_size = N / 2;
    size_t local_size = 256;
    size_t num_work_groups = global_size / local_size;

    //create buffer
    cl_mem buf_partial_sum[2];
    buf_partial_sum[0] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(double) * num_work_groups, NULL, &err);
    CHECK_ERROR(err);

    buf_partial_sum[1] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                        sizeof(double) * num_work_groups, NULL, &err);
    CHECK_ERROR(err);

    cl_event kernel_event[2], read_event[2];
    int base, offset;
    cl_ulong time_start, time_end;

    base = 0;
    offset = N /2;
    //input kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(int), &N);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(int), &base);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(int), &offset);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_partial_sum[0]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 4, sizeof(double) * local_size, NULL);
    CHECK_ERROR(err);

    err = clEnqueueNDRangeKernel(queue[0], kernel, 1, NULL, &global_size, &local_size, 0, NULL, &kernel_event[0]);
    CHECK_ERROR(err);
    err = clFlush(queue[0]);
    CHECK_ERROR(err);

    base = N / 2;
    offset = N /2;
    //input kernel arguments
    err = clSetKernelArg(kernel, 0, sizeof(int), &N);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(int), &base);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(int), &offset);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_partial_sum[1]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 4, sizeof(double) * local_size, NULL);
    CHECK_ERROR(err);

    err = clEnqueueNDRangeKernel(queue[0], kernel, 1, NULL, &global_size, &local_size, 0, NULL, &kernel_event[1]);
    CHECK_ERROR(err);
    err = clFlush(queue[0]);
    CHECK_ERROR(err);

    double *partial_sum = (double*)malloc(sizeof(double) * num_work_groups * 2);

    err = clEnqueueReadBuffer(queue[1], buf_partial_sum[0], CL_TRUE, 0,
            sizeof(double) * num_work_groups, partial_sum, 1, &kernel_event[0], &read_event[0]);
    CHECK_ERROR(err);
    err = clFlush(queue[1]);

    err = clEnqueueReadBuffer(queue[1], buf_partial_sum[1], CL_TRUE, 0,
            sizeof(double) * num_work_groups, partial_sum + num_work_groups, 1, &kernel_event[1], &read_event[1]);
    CHECK_ERROR(err);
    err = clFlush(queue[1]);
    CHECK_ERROR(err);

    err = clFinish(queue[0]);
    CHECK_ERROR(err);
    err = clFinish(queue[1]);
    CHECK_ERROR(err);

    double sum = 0;
    int i;
    for(i = 0; i < num_work_groups; i++){
        sum += partial_sum[i];
    }
    end_time = get_time();
    printf("Elapsed time: %f sec\n", end_time - start_time);
    clGetEventProfilingInfo(kernel_event[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
    clGetEventProfilingInfo(kernel_event[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
    printf("kernel_event 0 Elaped time : %lu ns\n", time_end - time_start);

    clGetEventProfilingInfo(kernel_event[1], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
    clGetEventProfilingInfo(kernel_event[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
    printf("kernel_event 1 Elaped time : %lu ns\n", time_end - time_start);

    clGetEventProfilingInfo(read_event[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
    clGetEventProfilingInfo(read_event[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
    printf("read_event 0 Elaped time : %lu ns\n", time_end - time_start);

    clGetEventProfilingInfo(read_event[1], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
    clGetEventProfilingInfo(read_event[1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
    printf("read_event 1 Elaped time : %lu ns\n", time_end - time_start);

    clReleaseMemObject(buf_partial_sum[0]);
    clReleaseMemObject(buf_partial_sum[1]);
    free(partial_sum);
    clReleaseContext(context);
    clReleaseCommandQueue(queue[0]);
    clReleaseCommandQueue(queue[1]);
    clReleaseProgram(program);
    clReleaseKernel(kernel);

    return sum;
}
