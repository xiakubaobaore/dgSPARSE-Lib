#ifndef CONVERT_CUDA
#define CONVERT_CUDA

#include "stdio.h"
#include <torch/extension.h>

#define THREADS 256

__global__ void ind2ptr_kernel(const int *ind_data, int *out_data,
                               int64_t M, int numel) {

  int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;

  if (thread_idx == 0) {
    for (int i = 0; i <= ind_data[0]; i++)
      out_data[i] = 0;
  } else if (thread_idx < numel) {
    for (int i = ind_data[thread_idx - 1]; i < ind_data[thread_idx]; i++)
      out_data[i + 1] = thread_idx;
  } else if (thread_idx == numel) {
    for (int i = ind_data[numel - 1] + 1; i < M + 1; i++)
      out_data[i] = numel;
  }
}


__global__ void ptr2ind_kernel(const int *ptr_data, int *out_data,
                               int64_t E, int numel) {
                          
  int thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
  printf("enter kernel and thread_idx is %d, E is %ld, numel is %d, thread_idx < numel is %d \n", thread_idx, E, numel, thread_idx < numel);

  if (thread_idx < numel) {
    // printf("enter thread_idx < numel \n");
    // int idx = ptr_data[thread_idx], next_idx = ptr_data[thread_idx + 1];
    // printf("idx is: %d, next_idx is: %d \n", idx, next_idx);
    // for (int i = idx; i < next_idx; i++) {
    //   printf("num %d will write %d \n", i, thread_idx);
    //   out_data[i] = thread_idx;
    // }
    out_data[thread_idx] = 1;
  }
}


#endif
