#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

int main() {
    // TODO
    //Check platforms count and IDs
    int loop = 0;
    int loop_device = 0;
    cl_int err;
    cl_uint num_platforms;
    cl_platform_id *platforms;

    cl_uint num_devices;
    cl_device_id *devices;
    cl_device_type device_type;

    cl_ulong mem;
    size_t max_size;
    size_t name_size;
    char* name;
    char* vendor;

    err = clGetPlatformIDs(0, NULL, &num_platforms);
    CHECK_ERROR(err);
    printf("Number of platforms: %d\n\n",num_platforms);

    platforms = (cl_platform_id *)malloc(sizeof(cl_platform_id) * num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    CHECK_ERROR(err);
    for(loop = 0; loop < num_platforms; loop++){
        printf("platform : %d\n", loop);
        //Get platform name
        err = clGetPlatformInfo(platforms[loop], CL_PLATFORM_NAME, 0, NULL, &name_size);
        CHECK_ERROR(err);
        name = (char *) malloc(name_size);
        err = clGetPlatformInfo(platforms[loop], CL_PLATFORM_NAME, name_size, name, NULL);
        printf("-  CL_PLATFORM_NAME   : %s\n",name);
        free(name);
        //Get platform vendor
        err = clGetPlatformInfo(platforms[loop], CL_PLATFORM_VENDOR, 0, NULL, &name_size);
        CHECK_ERROR(err);
        vendor = (char *)malloc(name_size);
        err = clGetPlatformInfo(platforms[loop], CL_PLATFORM_VENDOR, name_size, vendor, NULL);
        printf("-  CL_PLATFORM_VENDOR : %s\n",vendor);
        free(vendor);
        //Check device count and Get device ID
        err = clGetDeviceIDs(platforms[loop], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        CHECK_ERROR(err);
        devices = (cl_device_id *)malloc(sizeof(cl_device_id) * num_devices);
        err = clGetDeviceIDs(platforms[loop], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
        CHECK_ERROR(err);
        printf("\n\nNumber of devices: %d\n\n",num_devices);

        for(loop_device = 0; loop_device < num_devices; loop_device++){
            printf("device: %d\n", loop_device);
            //Get Device type
            err = clGetDeviceInfo(devices[loop_device], CL_DEVICE_TYPE, sizeof(cl_device_type),
                    &device_type, NULL);
            CHECK_ERROR(err);
            printf("-  CL_DEVICE_TYPE  :");
            if(device_type & CL_DEVICE_TYPE_CPU){
                printf(" CLDEVICE_TYPE_CPU");
            }
            if(device_type & CL_DEVICE_TYPE_GPU){
                printf(" CLDEVICE_TYPE_GPU");
            }
            if(device_type & CL_DEVICE_TYPE_ACCELERATOR){
                printf(" CLDEVICE_TYPE_ACCELERATOR");
            }
            if(device_type & CL_DEVICE_TYPE_DEFAULT){
                printf(" CLDEVICE_TYPE_DEFAULT");
            }
            if(device_type & CL_DEVICE_TYPE_CUSTOM){
                printf(" CLDEVICE_TYPE_CUSTOM");
            }
            printf("\n");

            //Get Device name
            err = clGetDeviceInfo(devices[loop_device], CL_DEVICE_NAME, 0, NULL, &name_size);
            CHECK_ERROR(err);
            name = (char*)malloc(name_size);
            err = clGetDeviceInfo(devices[loop_device], CL_DEVICE_NAME, name_size, name, NULL);
            CHECK_ERROR(err);
            printf("-  CL_DEVICE_NAME :  %s\n",name);
            free(name);

            //Get Device Max work group size
            err = clGetDeviceInfo(devices[loop_device], CL_DEVICE_MAX_WORK_GROUP_SIZE,
                    sizeof(size_t), &max_size, NULL);
            CHECK_ERROR(err);
            printf("-  CL_DEVICE_MAX_WORK_GROUP_SIZE: %d\n",max_size);

            //Get Device Global mem size
            err = clGetDeviceInfo(devices[loop_device], CL_DEVICE_GLOBAL_MEM_SIZE,
                    sizeof(cl_ulong), &mem, NULL);
            CHECK_ERROR(err);
            printf("-  CL_DEVICE_GLOBAL_MEM_SIZE: %ld\n",mem);

            //Get Device local mem size
            err = clGetDeviceInfo(devices[loop_device], CL_DEVICE_LOCAL_MEM_SIZE,
                    sizeof(cl_ulong), &mem, NULL);
            CHECK_ERROR(err);
            printf("-  CL_DEVICE_LOCAL_MEM_SIZE: %ld\n",mem);

            //Get Device max mem alloc size
            err = clGetDeviceInfo(devices[loop_device], CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                    sizeof(cl_ulong), &mem, NULL);
            CHECK_ERROR(err);
            printf("-  CL_DEVICE_MAX_MEM_ALLOC_SIZE: %ld\n",mem);
            printf("\n\n");
        }
    }

    return 0;
}
