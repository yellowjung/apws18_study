mat_mul: mat_mul.cu mat_mul_seq.cu mat_mul_opencl.cu mat_mul_cuda.cu mat_mul_cuda_multi.cu mat_mul_cublas.cu
	nvcc -o mat_mul mat_mul.cu mat_mul_seq.cu mat_mul_opencl.cu mat_mul_cuda.cu mat_mul_cuda_multi.cu mat_mul_cublas.cu -lOpenCL -I/usr/local/cuda/include -lcublas

clean:
	rm mat_mul
