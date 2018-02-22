#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>

#define CHECK_CUBLAS_ERROR(err) \
if (err != CUBLAS_STATUS_SUCCESS) \
    {\
        printf("[%s:%d] CUBLAS error %d\n", __FILE__, __LINE__, err);\
        exit(EXIT_FAILURE);\
    } 

void mat_mul_cublas(float *A, float *B, float *C,
                    int ROW_A, int COL_A, int COL_B) {
  int N = ROW_A;
  if(ROW_A != COL_A || ROW_A != COL_B)
  {
    printf("Support Square Matrix Only!\n");
   exit(EXIT_FAILURE);
  }
  /******************** TODO *********************/


}

