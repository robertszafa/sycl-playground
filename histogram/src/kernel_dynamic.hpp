#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#include "pipe_utils.hpp"
#include "unrolled_loop.hpp"
#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

class HistogramKernel;

constexpr uint STORE_LATENCY = 72;
// constexpr uint STOREQ_ENTRIES = 8;
constexpr uint STOREQ_ENTRIES = 1;

struct idx_val_pair {
  uint idx;
  uint val;
};

double histogram_kernel(queue &q, queue &q2, const std::vector<uint> &feature, const std::vector<uint> &weight,
                        std::vector<uint> &hist) {
  std::cout << "Dynamic HLS\n";

  const uint array_size = feature.size();

  buffer feature_buf(feature);
  buffer weight_buf(weight);
  buffer hist_buf(hist);

  // using p_wt = pipe<class p_wt_class, uint>;
  // using p_m_store = pipe<class p_m_store_class, uint>;
  using p_m_load = pipe<class p_m_load_class, uint, 4>;

  using p_store_ack = pipe<class p_store_ack_class, uint, 4>;
  using p_m_hist_store = pipe<class p_hist_store_class, idx_val_pair, 4>;
  using p_m_hist_store2 = pipe<class p_hist_store2_class, idx_val_pair, 4>;

  using p_to_store =
      pipe<class p_to_store_class, idx_val_pair, 4>;

  using p_hist_out = pipe<class p_hist_out_class, uint, 4>;

  using p_predicate = pipe<class p_predicate_class, bool, 4>;
  using p_predicate_port = pipe<class p_predicate_port_class, bool, 4>;


  q2.submit([&](handler &hnd) {
    accessor hist(hist_buf, hnd, write_only);

    hnd.single_task<class StorePort>([=]() [[intel::kernel_args_restrict]] {
      [[intel::ivdep]]
      while (p_predicate_port::read()) {
        auto hist_store = p_m_hist_store2::read();

        hist[hist_store.idx] = hist_store.val;
        atomic_fence(memory_order_seq_cst, memory_scope_work_item);
        p_store_ack::write(hist_store.idx);
      }
    });
  });

  auto event = q.submit([&](handler &hnd) {
    accessor feature(feature_buf, hnd, read_only);
    accessor weight(weight_buf, hnd, read_only);

    hnd.single_task<HistogramKernel>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) {

        auto m = feature[i];
        auto wt = weight[i];

        // p_predicate::write(true);
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
    // accessor hist(hist_buf, hnd, read_only);

    struct store_entry {
      uint idx;
      uint val;
      bool in_flight;
    };

    // using store_ack_pipe_array = fpga_tools::PipeArray< // Defined in "pipe_utils.hpp".
    //     class class_store_ack_pipe_array,               // An identifier for the pipe.
    //     bool,                                           // The type of data in the pipe.
    //     0,                                              // The capacity of each pipe.
    //     STOREQ_ENTRIES                                  // array dimension.
    //     >;

    hnd.single_task<class HistLSQ>([=]() [[intel::kernel_args_restrict]] {
      store_entry store_q[STOREQ_ENTRIES];
#pragma unroll
      for (uint i = 0; i < STORE_LATENCY; ++i)
        store_q[i] = {(uint)-1, (uint)-1, false};

      bool predicate_read_succ = false;
      bool any_in_flight = false;
      uint timestamp = 0;

      // while (p_predicate::read()) {
      // while (1) {
      [[intel::ivdep]] 
      do {

        if (!predicate_read_succ)
          auto _dc = p_predicate::read(predicate_read_succ);

        int next_slot = -1;

        // Count free slots in the queue.
        uint num_in_flight = 0;
        #pragma unroll
        for (uint i = 0; i < STOREQ_ENTRIES; ++i) {
          if (store_q[i].in_flight) {
            num_in_flight += 1;
          }
          else {
            next_slot = i;
          }
        }

        bool stall_stores = num_in_flight >= STOREQ_ENTRIES;
        bool is_store = false;
        bool is_load = false;
        uint m_load;
        idx_val_pair m_hist_store;

        m_load = p_m_load::read(is_load);
        if (!stall_stores)
          m_hist_store = p_m_hist_store::read(is_store);


        ////// Load logic
        if (is_load) {
          bool in_flight_ld_logic = false;
          uint hist_forward;

          // Check if there laready is an in-flight store to the requested index.
          #pragma unroll
          for (uint i = 0; i < STOREQ_ENTRIES; ++i) {
            in_flight_ld_logic |= (m_load == store_q[i].idx);
            if (in_flight_ld_logic) {
              hist_forward = store_q[i].val;
            }
          }

          // If found, forward the value from the storeQ to the load. Otherwise, issue load.
          if (in_flight_ld_logic)
            p_hist_out::write(hist_forward);
          else
            p_hist_out::write(0); // Dummy value for now, issuing actual ld results in deadlock.
            // p_hist_out::write(hist[m_load]);
        }

        ////// Store logic
        if (is_store) {
          // Check if a different store to the same address is in flight.
          #pragma unroll
          for (uint i = 0; i < STOREQ_ENTRIES; ++i) {
            if (m_hist_store.idx == store_q[i].idx)
              store_q[i] = store_entry{(uint) -1, (uint) -1, false};
          }

          // Enqueue store.
          p_predicate_port::write(true);
          p_m_hist_store2::write(m_hist_store);
          store_q[next_slot] = store_entry{m_hist_store.idx, m_hist_store.val, true};
        }

        bool ack_success = false;
        auto i_ack = p_store_ack::read(ack_success);
        if (ack_success) {
          store_q[i_ack].in_flight = false;
        }

        any_in_flight = false;
        #pragma unroll
        for (uint i = 0; i < STOREQ_ENTRIES; ++i)
          any_in_flight |= store_q[i].in_flight;

      } while (!predicate_read_succ || any_in_flight);

      p_predicate_port::write(false);
      // sycl::ext::oneapi::experimental::printf("Finished LSQ\n");
    });
  });
 


  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}
