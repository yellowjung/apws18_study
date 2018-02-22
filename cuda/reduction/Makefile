mat_mul: main.cu reduction.cu reduction_opencl.cu
	nvcc -o reduction main.cu reduction.cu reduction_opencl.cu -lOpenCL -I/usr/local/cuda/include

clean:
	rm reduction
