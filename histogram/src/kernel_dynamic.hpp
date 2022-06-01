#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

class HistogramKernel;

constexpr uint STORE_LATENCY = 72;

struct idxValPair {
  uint idx;
  uint val;
};

double histogram_kernel(queue &q, const std::vector<uint> &feature, const std::vector<uint> &weight,
                        std::vector<uint> &hist) {
  std::cout << "Dynamic HLS\n";

  const uint array_size = feature.size();

  buffer feature_buf(feature);
  buffer weight_buf(weight);
  buffer hist_buf(hist);

  // using p_wt = pipe<class p_wt_class, uint>;
  // using p_m_store = pipe<class p_m_store_class, uint>;
  using p_m_load = pipe<class p_m_load_class, uint, 128>;

  // using p_hist_load = pipe<class p_hist_load_class, uint>;
  using p_m_hist_store = pipe<class p_hist_store_class, idxValPair, 128>;

  using p_hist_out = pipe<class p_hist_out_class, uint, 128>;
  
  using p_predicate = pipe<class p_predicate_class, bool, 128>;



  auto event = q.submit([&](handler &hnd) {
    accessor feature(feature_buf, hnd, read_only);
    accessor weight(weight_buf, hnd, read_only);

    hnd.single_task<HistogramKernel>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) {
        auto m = feature[i];
        auto wt = weight[i];

        p_predicate::write(true);
        p_m_load::write(m);

        auto x = p_hist_out::read();
        auto x_new = x + wt;

        // hist[m] = x + wt;
        p_m_hist_store::write(idxValPair{m, x_new});
      }

      p_predicate::write(false);
    });
  });

  q.submit([&](handler &hnd) {
    accessor hist(hist_buf, hnd, write_only);

    hnd.single_task<class HistLSQ>([=]() [[intel::kernel_args_restrict]] {
      idxValPair store_q[STORE_LATENCY];
      #pragma unroll
      for (uint i = 0; i < STORE_LATENCY; ++i) 
        store_q[i].idx = (uint) -1;

      bool in_fligh_st_logic = false, in_flight_ld_logic = false;
      uint in_flight_st_idx;

      idxValPair m_hist_store;
      uint m_load, m_store;
      uint hist_store, hist_load, hist_forward;

      bool is_store, is_load;
      while (p_predicate::read()) {
        // Shift the storeQ by one entry on every cycle.
        #pragma unroll
        for (uint i = 1; i < STORE_LATENCY; ++i) {
          store_q[i] = store_q[i - 1];
        }

        is_store = false;
        is_load = false;
        in_fligh_st_logic = false;
        in_flight_ld_logic = false;

        m_load = p_m_load::read(is_load);
        m_hist_store = p_m_hist_store::read(is_store);

        // Load logic
        if (is_load) {
          // Check if there is an in-flight store to the requested address.
          #pragma unroll
          for (uint i = 0; i < STORE_LATENCY; ++i) {
            in_flight_ld_logic |= (m_load == store_q[i].idx);
            if (in_flight_ld_logic) {
              hist_forward = store_q[i].val;
            }
          }

          // If found, forward the value from the storeQ to the load.
          // Otherwise, issue load.
          if (in_flight_ld_logic)
            p_hist_out::write(hist_forward);
          else
            p_hist_out::write(hist[m_load]);
        }

        // Store logic
        if (is_store) {
          // Check if a different store to the same address is in flight.
          #pragma unroll
          for (uint i = 0; i < STORE_LATENCY; ++i) {
            in_fligh_st_logic |= (m_hist_store.idx == store_q[i].idx);
            if (in_fligh_st_logic) {
              store_q[i].idx = (uint)-1;
            }
          }

          // Enqueue store.
          store_q[0] = m_hist_store;
        }
      }
    });
  });

  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}
