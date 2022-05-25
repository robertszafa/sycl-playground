#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#if FPGA || FPGA_EMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif

using namespace sycl;

class compute;

template <typename T>
double data_hazard_kernel(queue &q, const std::vector<int> &addr_in,
                          const std::vector<int> &addr_out, std::vector<T> &A) {
  std::cout << "Static HLS\n";

  const int array_size = A.size();

  buffer addr_in_buf(addr_in);
  buffer addr_out_buf(addr_out);
  buffer A_buf(A);

  event e = q.submit([&](handler &hnd) {
    accessor addr_in(addr_in_buf, hnd, read_only);
    accessor addr_out(addr_out_buf, hnd, read_only);
    accessor A(A_buf, hnd, read_write);

    // From Benchmarks for High-Level Synthesis (Jianyi Cheng)
    hnd.single_task<compute>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) {
        auto beta = A[addr_in[i]];

        T result;
        if (beta >= 1) {
          result = 1.0;
        } else {
          result = ((beta * beta + 19.52381) * beta * beta + 3.704762) * beta;
        }

        A[addr_out[i]] = result;
      }
    });
  });

  auto start = e.get_profiling_info<info::event_profiling::command_start>();
  auto end = e.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}
