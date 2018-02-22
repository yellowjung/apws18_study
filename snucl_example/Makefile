all: mat_mul_single mat_mul_multi mat_mul_snucl mat_mul_snucl_opt mat_mul_mpi

mat_mul_single: mat_mul_single.c
	gcc -o mat_mul_single mat_mul_single.c -lOpenCL

mat_mul_multi: mat_mul_multi.c
	gcc -o mat_mul_multi mat_mul_multi.c -lOpenCL

mat_mul_snucl: mat_mul_multi.c
	gcc -o mat_mul_snucl mat_mul_multi.c -L$(SNUCLROOT)/lib -lsnucl_cluster

mat_mul_snucl_opt: mat_mul_snucl_opt.c
	gcc -o mat_mul_snucl_opt mat_mul_snucl_opt.c -L$(SNUCLROOT)/lib -lsnucl_cluster

mat_mul_mpi: mat_mul_mpi.c
	mpicc -o mat_mul_mpi mat_mul_mpi.c -lOpenCL

clean:
	rm mat_mul_single mat_mul_multi mat_mul_snucl mat_mul_snucl_opt mat_mul_mpi
