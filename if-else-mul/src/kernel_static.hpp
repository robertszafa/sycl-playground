#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

// Forward declare kernel names to avoid name mangling.
class Compute;

template <typename T>
double if_else_mul_kernel(queue &q, const std::vector<int> &wet, std::vector<T> &B,
                          const int array_size) {
  std::cout << "Static HLS\n";

  buffer wet_buf(wet);
  buffer B_buf(B);

  event e = q.submit([&](handler &hnd) {
    accessor wet(wet_buf, hnd, read_only);
    accessor B(B_buf, hnd, write_only);

    hnd.single_task<Compute>([=]() [[intel::kernel_args_restrict]]  {
      T etan, t = 0.0;
      // II=78
      for (int i = 0; i < array_size; ++i) {
        if (wet[i] > 0) {
          // 78 cycles of stall for 32-bit (more for 64-bit since can't use DSPs)
          t = 0.25 + etan * T(wet[i]) / 2.0 + exp(etan);
          etan = etan + t + exp(t+etan);
        } else {
          // 3 cycles of stall
          etan -= 0.01;
        }
      }

      B[0] = etan - 2.0;
    });
  });

  auto start = e.get_profiling_info<info::event_profiling::command_start>();
  auto end = e.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}
