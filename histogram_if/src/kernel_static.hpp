#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#include <sycl/ext/intel/fpga_extensions.hpp>

#include "memory_utils.hpp"

using namespace sycl;
using namespace fpga_tools;

class HistogramKernel;

double histogram_if_kernel(queue &q, const std::vector<int> &h_feature, const std::vector<int> &h_weight,
                           std::vector<int> &h_hist) {
  std::cout << "Static HLS\n";

  const uint array_size = h_feature.size();

  int* feature = toDevice(h_feature, q);
  int* weight = toDevice(h_weight, q);
  int* hist = toDevice(h_hist, q);

  auto event = q.submit([&](handler &hnd) {
    hnd.single_task<HistogramKernel>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) {
        auto wt = weight[i];
        auto idx = feature[i];
        auto x = hist[idx];

        if (wt > 0)
          hist[idx] = x + wt;
      }
    });
  });

  event.wait();
  q.copy(hist, h_hist.data(), h_hist.size()).wait();

  sycl::free(hist, q);
  sycl::free(feature, q);
  sycl::free(weight, q);

  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}
