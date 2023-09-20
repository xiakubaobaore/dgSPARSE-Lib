


#include "../../include/cuda/convert_cuda.h"
#include "../../include/cuda/convert_cuda.cuh"
#include <torch/extension.h>

torch::Tensor ind2ptr_cuda(torch::Tensor ind, int64_t M) {

  auto out = torch::empty({M + 1}, ind.options());

  if (ind.numel() == 0)
    return out.zero_();

  auto ind_data = ind.data_ptr<int>();
  auto out_data = out.data_ptr<int>();
  ind2ptr_kernel<<<(ind.numel() + 2 + THREADS - 1) / THREADS, THREADS, 0>>>(ind_data, out_data, M, ind.numel());
  return out;
}


torch::Tensor ptr2ind_cuda(torch::Tensor ptr, int64_t E) {

  printf("enter cu file and ptr.numel() is %d, E is %d, ptr.options \n", ptr.numel(), E);
  // cudaError_t err = cudaGetLastError();
  // if (err != cudaSuccess) {
  //   printf("CUDA Error: %s\n", cudaGetErrorString(err));
  // }
  auto out = torch::zeros({E}, ptr.options());
  // auto ptr_data = ptr.data_ptr<int>();
  // auto out_data = out.data_ptr<int>();
  // printf("ptr_data is %p \n", ptr_data);
  // printf("out_data is %p \n", out_data);
  // printf("ptr_data[0] is: %d \n", *ptr_data);
  // printf("and ptr_data is: \n");
  // for (int i = 0; i < 5; i++) {
  //   printf("%d ", ptr_data[i]);
  // }
  // printf("\n");
  ptr2ind_kernel<<<(ptr.numel() - 1 + THREADS - 1) / THREADS, THREADS>>>(ptr.data_ptr<int>(), out.data_ptr<int>(), E, ptr.numel() - 1);
  printf("out kernel, ready return, THREADS is %d \n", THREADS);
  return out;
}


