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

double if_mul_kernel(queue &q, const std::vector<int> &wet, std::vector<float> &B, const int array_size) {
  std::cout << "Static HLS\n";
  
  buffer wet_buf(wet);
  buffer B_buf(B);

  auto event = q.submit([&](handler &hnd) {
    accessor wet(wet_buf, hnd, read_only);
    accessor B(B_buf, hnd, write_only);

    hnd.single_task<class read>([=]() [[intel::kernel_args_restrict]] {
      float etan, t = 0.0;
      // II=35
      for (int i=0; i < array_size; ++i) {
          if (wet[i] > 0) {
              // 35 cycles of stall
              t = 0.25 + etan * float(wet[i]) / 2.0;
              etan += t;
          }
      } 

      B[0] = etan;
    });
  });

  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}
