#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/detail/common.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include <CL/sycl.hpp>
#include <cstdio>
#include <iostream>
#include <vector>

#include <sycl/ext/intel/fpga_extensions.hpp>

#include "store_queue.hpp"
#include "memory_utils.hpp"

using namespace sycl;
using namespace fpga_tools;

double spmv_kernel(queue &q, 
                   std::vector<float> &h_matrix,       
                   const std::vector<int> &h_row,
                   const std::vector<int> &h_col,
                   const std::vector<float> &h_a,             
                   const int M) {

  std::cout << "Static HLS\n";

  float *matrix = toDevice(h_matrix, q);
  int *row = toDevice(h_row, q);
  int *col = toDevice(h_col, q);
  float *a = toDevice(h_a, q);

  auto event = q.single_task<class spmv_static>([=]() [[intel::kernel_args_restrict]] {
    for (int k = 1; k < M; k++) {
      for (int p = 0; p < M; p++) {
        matrix[k*M + row[p]] += a[p] * matrix[(k - 1) * M + col[p]];
      }
    }
  });

  event.wait();
  q.memcpy(h_matrix.data(), matrix, sizeof(h_matrix[0])*h_matrix.size()).wait();

  sycl::free(matrix, q);  
  sycl::free(row, q);  
  sycl::free(col, q);  
  sycl::free(a, q);  

  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}
