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

#ifdef __SYCL_DEVICE_ONLY__
  #define CL_CONSTANT __attribute__((opencl_constant))
#else
  #define CL_CONSTANT
#endif

#define PRINTF(format, ...) { \
            static const CL_CONSTANT char _format[] = format; \
            sycl::ext::oneapi::experimental::printf(_format, ## __VA_ARGS__); }


double maximal_matching_kernel(queue &q, const std::vector<int> &h_edges, std::vector<int> &h_vertices,
                               int *h_out, const int num_edges) {
  const int* edges = toDevice(h_edges, q);
  int* vertices = toDevice(h_vertices, q);
  int* out = toDevice(h_out, 1, q);

  auto event = q.submit([&](handler &hnd) {
    hnd.single_task<class StaticKernel>([=]() [[intel::kernel_args_restrict]] {
      int i = 0;
      int out_scalar = 0;

      while (i < num_edges) {
        int j = i * 2;

        int u = edges[j];
        int v = edges[j + 1];

        if ((vertices[u] < 0) && (vertices[v] < 0)) {
          vertices[u] = v;
          vertices[v] = u;

          out_scalar = out_scalar + 1;
        }

        i = i + 1;
      }

      *out = out_scalar;
    });
  });

  event.wait();
  q.memcpy(h_vertices.data(), vertices, sizeof(h_vertices[0]) * h_vertices.size()).wait();
  q.memcpy(h_out, out, sizeof(h_out[0])).wait();

  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}
