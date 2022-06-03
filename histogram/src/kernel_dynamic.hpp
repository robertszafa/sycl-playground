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
constexpr uint STOREQ_ENTRIES = Q_SIZE;

struct idx_val_pair {
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

  using p_m_load = pipe<class p_m_load_class, uint, STORE_LATENCY>;

  using p_store_ack = pipe<class p_store_ack_class, uint, STORE_LATENCY>;
  using p_m_hist_store = pipe<class p_hist_store_class, idx_val_pair, STORE_LATENCY>;
  using p_m_hist_store2 = pipe<class p_hist_store2_class, idx_val_pair, STORE_LATENCY>;

  using p_to_store = pipe<class p_to_store_class, idx_val_pair, STORE_LATENCY>;

  using p_hist_out = pipe<class p_hist_out_class, uint, STORE_LATENCY>;

  using p_predicate = pipe<class p_predicate_class, bool, STORE_LATENCY>;
  using p_predicate_port = pipe<class p_predicate_port_class, bool, STORE_LATENCY>;


  auto event = q.submit([&](handler &hnd) {
    accessor feature(feature_buf, hnd, read_only);
    accessor weight(weight_buf, hnd, read_only);

    hnd.single_task<HistogramKernel>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) {

        auto m = feature[i];
        auto wt = weight[i];

        p_m_load::write(m);

        // sycl::ext::oneapi::experimental::printf("Issued load \n");
        auto x = p_hist_out::read();
        auto x_new = x + wt;
        // sycl::ext::oneapi::experimental::printf("Received value \n");

        // hist[m] = x + wt;
        p_m_hist_store::write(idx_val_pair{m, x_new});
        // sycl::ext::oneapi::experimental::printf("Issued store \n");
      }

      // sycl::ext::oneapi::experimental::printf("Finished \n");
      p_predicate::write(false);
    });
  });

  q.submit([&](handler &hnd) {
    accessor hist(hist_buf, hnd, read_write);

    struct store_entry {
      uint idx;
      uint val;
      int count_down;
    };


    hnd.single_task<class HistLSQ>([=]() [[intel::kernel_args_restrict]] {
      store_entry store_q[STOREQ_ENTRIES];

      bool terminate_signal = false;
      bool any_in_flight = false;

      [[intel::ivdep]] 
      do {

        // Count free slots in the queue. Get any free slot (we're ok with non-determinism here).
        int next_slot = -1;
        #pragma unroll
        for (uint i = 0; i < STOREQ_ENTRIES; ++i) {
          store_q[i].count_down -= 1;

          if (store_q[i].count_down <= 0) {
            next_slot = i;
            store_q[i].count_down = 0;
          }
        }

        bool is_load = false;
        auto m_load = p_m_load::read(is_load);

        bool is_store = false;
        idx_val_pair m_hist_store;

        // Only accept new store entry if there's space for it. If not, it will wait in the FIFO.
        bool stall_store = next_slot == -1;
        if (!stall_store)
          m_hist_store = p_m_hist_store::read(is_store);

        /* Load logic */
        if (is_load) {
          // Check if there already is an in-flight store to the requested index.
          // TODO: Select younger store based on count_down.
          bool in_flight_ld_logic = false;
          uint hist_forward;
          #pragma unroll
          for (uint i = 0; i < STOREQ_ENTRIES; ++i) {
            in_flight_ld_logic |= (m_load == store_q[i].idx);
            if (in_flight_ld_logic) 
              hist_forward = store_q[i].val;
          }

          // If yes, use its value.
          if (in_flight_ld_logic)
            p_hist_out::write(hist_forward);
          else
            p_hist_out::write(hist[m_load]);
        }
        /* End Load logic */

        /* Store logic */
        if (is_store) {
          hist[m_hist_store.idx] = m_hist_store.val;
          store_q[next_slot] = store_entry{m_hist_store.idx, m_hist_store.val, STORE_LATENCY};
          // atomic_fence(memory_order_seq_cst, memory_scope_device);
          // p_store_ack::write(m_hist_store.idx);
        }
        /* End Store logic */

        // bool ack_success = false;
        // auto i_ack = p_store_ack::read(ack_success);
        // if (ack_success) {
        //   store_q[i_ack].in_flight = false;
        // }

        any_in_flight = false;
        #pragma unroll
        for (uint i = 0; i < STOREQ_ENTRIES; ++i)
          any_in_flight |= (store_q[i].count_down > 0);

        if (!terminate_signal)
          auto _dc = p_predicate::read(terminate_signal);

      } while (!terminate_signal || any_in_flight);

    });
  });
 


  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}
