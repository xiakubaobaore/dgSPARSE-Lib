

#include "../include/cuda/convert_cuda.h"
#include <torch/all.h>
#include <torch/extension.h>
#include <torch/python.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>


torch::Tensor ind2ptr(torch::Tensor ind, int64_t M) {
    if (ind.device().is_cuda()) {
#ifdef WITH_CUDA
    return  ind2ptr_cuda(ind, M);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
    } else {
        printf("no CUDA \n");
        return torch::Tensor();
    }
}



torch::Tensor ptr2ind(torch::Tensor ptr, int64_t E) {
    if (ptr.device().is_cuda()) {
#ifdef WITH_CUDA
    printf("enter cpp and E is %d, ptr_data is %p \n", E, ptr.data_ptr<int>());
    return  ptr2ind_cuda(ptr, E);
#else
    AT_ERROR("Not compiled with CUDA support");
#endif
    } else {
        printf("no CUDA \n");
        return torch::Tensor();
    }
}


TORCH_LIBRARY(dgsparse_convert, m) {
  m.def("ind2ptr", &ind2ptr);
  m.def("ptr2ind", &ptr2ind);
}




