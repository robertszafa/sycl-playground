#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

#ifdef __SYCL_DEVICE_ONLY__
  #define CL_CONSTANT __attribute__((opencl_constant))
#else
  #define CL_CONSTANT
#endif

#define PRINTF(format, ...) { \
            static const CL_CONSTANT char _format[] = format; \
            sycl::ext::oneapi::experimental::printf(_format, ## __VA_ARGS__); }


double maximal_matching_kernel(queue &q, const std::vector<int> &edges, std::vector<int> &vertices,
                               int *out, const int num_edges) {
  std::cout << "Static HLS\n";

  buffer edges_buf(edges);
  buffer vertices_buf(vertices);
  buffer out_buf(out, range{1});

  event e = q.submit([&](handler &hnd) {
    accessor edges(edges_buf, hnd, read_only);
    accessor vertices(vertices_buf, hnd, read_write);
    accessor out_pointer(out_buf, hnd, write_only);

    hnd.single_task<class StaticKernel>([=]() [[intel::kernel_args_restrict]] {
      int i = 0;
      int out = 0;

      while (i < num_edges) {
        int j = i * 2;

        int u = edges[j];
        int v = edges[j + 1];

        if ((vertices[u] < 0) && (vertices[v] < 0)) {
          vertices[u] = v;
          vertices[v] = u;

          out = out + 1;
        }

        i = i + 1;
      }

      out_pointer[0] = out;
    });
  });

  auto start = e.get_profiling_info<info::event_profiling::command_start>();
  auto end = e.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}
