#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/detail/common.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include <CL/sycl.hpp>
#include <cstdio>
#include <iostream>
#include <vector>

#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

using PipelinedLSU = ext::intel::lsu<>;

double spmv_kernel(queue &q, 
                   std::vector<float> &matrix,       
                   const std::vector<int> &row,
                   const std::vector<int> &col,
                   std::vector<float> &a,             
                   const int M) {

  buffer matrix_buf(matrix);
  buffer row_buf(row);
  buffer col_buf(col);
  buffer a_buf(a);

  auto event = q.submit([&](handler &hnd) {
    accessor matrix(matrix_buf, hnd, read_write);
    accessor a(a_buf, hnd, read_only);
    accessor row(row_buf, hnd, read_only);
    accessor col(col_buf, hnd, read_only);

    hnd.single_task<class spmv_static>([=]() [[intel::kernel_args_restrict]] {
      for (int k = 1; k < M; k++) {
        for (int p = 0; p < M; p++) {
          auto new_val = matrix[k*M + row[p]] + a[p] * matrix[(k - 1) * M + col[p]];
          PipelinedLSU::store(matrix.get_pointer() + (k*M + row[p]), new_val);
        }
      }
    });
  });

  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}
