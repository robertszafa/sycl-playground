#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#include <sycl/ext/intel/fpga_extensions.hpp>

#include "store_queue.hpp"
#include "memory_utils.hpp"

using namespace sycl;

class HistogramKernel;


double histogram_kernel(queue &q, const std::vector<uint> &h_feature, const std::vector<uint> &h_weight,
                        std::vector<uint> &h_hist) {
  std::cout << "Static HLS\n";

  const uint array_size = h_feature.size();

  const auto feature = toDevice(h_feature, q);
  const auto weight = toDevice(h_weight, q);
  uint* hist = toDevice(h_hist, q);

  auto event = q.submit([&](handler &hnd) {
    hnd.single_task<HistogramKernel>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) {
        uint wt = weight[i];
        uint idx = feature[i];
        uint x = hist[idx];
        
        // #pragma unroll
        // for (int j = 0; j < 256; ++j)
        //   x = x*x + j;

        
        hist[idx] = x;
      }
    });
  });

  q.wait();
  q.memcpy(h_hist.data(), hist, sizeof(h_hist[0]) * h_hist.size()).wait();
  sycl::free(hist, q);
  sycl::free(feature, q);
  sycl::free(weight, q);

  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}
