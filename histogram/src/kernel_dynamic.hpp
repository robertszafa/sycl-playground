#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

using PipelinedLSU = ext::intel::lsu<>;
using BurstCoalescedLSU = ext::intel::lsu<ext::intel::burst_coalesce<false>, 
                                          ext::intel::statically_coalesce<false>>;

// This should be STORE_Q_SIZE >= STORE_LATENCY
constexpr uint STORE_Q_SIZE = 72;

struct store_entry {
  uint idx; // This should be the full address in a real impl.
  uint val;
};


double histogram_kernel(queue &q, const std::vector<uint> &feature, const std::vector<uint> &weight,
                        std::vector<uint> &hist) {
  std::cout << "Dynamic HLS\n";

  const uint array_size = feature.size();

  buffer feature_buf(feature);
  buffer weight_buf(weight);
  buffer hist_buf(hist);

  using weight_load_pipe = pipe<class weight_load_pipe_class, uint>;
  using feature_load_pipe = pipe<class feature_load_pipe_class, uint>;
  using hist_load_pipe = pipe<class hist_load_pipe_class, uint>;

  using hist_store_pipe = pipe<class hist_store_pipe_class, uint>;

  auto event = q.submit([&](handler &hnd) {
    accessor weight(weight_buf, hnd, read_only);

    hnd.single_task<class LoadWeight>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) {
        uint wt = weight[i];
        weight_load_pipe::write(wt);
      }
    });
  });
 
  q.submit([&](handler &hnd) {
    accessor hist(hist_buf, hnd, read_write);
    accessor feature(feature_buf, hnd, read_only);

    hnd.single_task<class LoadStoreHist>([=]() [[intel::kernel_args_restrict]] {
      [[intel::fpga_register]] store_entry store_q[STORE_Q_SIZE];

      // All entries are invalid at the start.
      #pragma unroll
      for (uint i=0; i<STORE_Q_SIZE; ++i) {
        store_q[i].idx = (uint) -1;
      }

      [[intel::ivdep]]
      for (int i = 0; i < array_size; ++i) {
        // Move the store queue forward by one every cycle.
        #pragma unroll
        for (uint i=0; i<STORE_Q_SIZE-1; ++i) {
          store_q[i] = store_q[i+1];
        }

        uint m = feature[i];

        bool forward = false;
        uint hist_val;

        // TODO: Do we need to explicitly pick the latest matching entry:
        //       the one with the hisghest idx? 
        #pragma unroll
        for (uint i=0; i<STORE_Q_SIZE; ++i) {
          if (store_q[i].idx == ext::intel::fpga_reg(m)) {
            forward = true;
            hist_val = store_q[i].val;
          }
        }

        if (!forward) hist_val = PipelinedLSU::load(hist.get_pointer() + m);
        hist_load_pipe::write(hist_val);

        auto new_hist = hist_store_pipe::read();
        hist[m] = new_hist;
        store_q[STORE_Q_SIZE-1] = store_entry{m, new_hist};
      }
    });
  });

  q.submit([&](handler &hnd) {
    hnd.single_task<class Compute>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) {
        uint wt = weight_load_pipe::read();
        uint hist = hist_load_pipe::read();

        auto new_hist = hist + wt;

        hist_store_pipe::write(new_hist);
      }
    });
  });



  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}

