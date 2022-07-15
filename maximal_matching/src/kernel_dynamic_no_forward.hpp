#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

// The default PipelinedLSU will start a load/store immediately, which the memory disambiguation 
// logic relies upon.
// A BurstCoalescedLSU would instead of waiting for more requests to arrive for a coalesced access.
using PipelinedLSU = ext::intel::lsu<>;

#ifndef Q_SIZE
  #define Q_SIZE 8
#endif

constexpr uint STORE_Q_SIZE = Q_SIZE;
constexpr uint STORE_LATENCY = 12; // This should be gotten from static analysis.


struct store_entry {
  int idx; // This should be the full address in a real impl.
  uint val;
  bool executed;
  int countdown;
  int tag;
};

constexpr store_entry INVALID_ENTRY = {-1, 0, false, -1, -1};

struct pair {
  int queue_idx; 
  uint store_idx; 
};

struct candidate {
  int tag; 
  bool forward;
};



double maximal_matching_kernel(queue &q, const std::vector<int> &edges, std::vector<int> &vertices,
                               int *out, const int num_edges) {
  std::cout << "Dynamic HLS\n";

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
