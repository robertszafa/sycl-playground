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

double histogram_kernel(queue &q, const std::vector<uint> &feature, const std::vector<uint> &weight,
                        std::vector<uint> &hist) {
  std::cout << "Dynamic HLS\n";

  const uint array_size = feature.size();

  buffer feature_buf(feature);
  buffer weight_buf(weight);
  buffer hist_buf(hist);

  // using p_wt = pipe<class p_wt_class, uint>;
  // using p_m_store = pipe<class p_m_store_class, uint>;
  using p_m_load = sycl::ext::intel::experimental::pipe<class p_m_load_class, uint, 256>;

  using p_store_ack = sycl::ext::intel::experimental::pipe<class p_store_ack_class, uint>;
  using p_m_hist_store =
      sycl::ext::intel::experimental::pipe<class p_hist_store_class, idx_val_pair, 256>;

  using p_to_store =
      sycl::ext::intel::experimental::pipe<class p_to_store_class, idx_val_pair, 256>;

  using p_hist_out = sycl::ext::intel::experimental::pipe<class p_hist_out_class, uint, 256>;

  using p_predicate = sycl::ext::intel::experimental::pipe<class p_predicate_class, bool, 256>;

  auto event = q.submit([&](handler &hnd) {
    accessor feature(feature_buf, hnd, read_only);
    accessor weight(weight_buf, hnd, read_only);

    hnd.single_task<HistogramKernel>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) {
        auto m = feature[i];
        auto wt = weight[i];

        // p_predicate::write(true);
        p_m_load::write(m);

        auto x = p_hist_out::read();
        auto x_new = x + wt;

        // hist[m] = x + wt;
        p_m_hist_store::write(idx_val_pair{m, x_new});
      }

      p_predicate::write(false);
    });
  });

  q.submit([&](handler &hnd) {
    accessor hist(hist_buf, hnd, read_write);

    struct store_entry {
      uint idx;
      uint val;
      bool in_flight;
    };

    using store_ack_pipe_array = fpga_tools::PipeArray< // Defined in "pipe_utils.hpp".
        class class_store_ack_pipe_array,               // An identifier for the pipe.
        bool,                                           // The type of data in the pipe.
        0,                                              // The capacity of each pipe.
        STOREQ_ENTRIES                                  // array dimension.
        >;

    using LSU = ext::intel::experimental::lsu<ext::intel::experimental::burst_coalesce<false>,
                                              ext::intel::experimental::statically_coalesce<false>>;

    hnd.single_task<class HistLSQ>([=]() [[intel::kernel_args_restrict]] {
      store_entry store_q[STOREQ_ENTRIES];
#pragma unroll
      for (uint i = 0; i < STORE_LATENCY; ++i)
        store_q[i].idx = (uint)-1;
      uint timestamp = 0;

      bool predicate_read_succ = false;
      bool any_in_flight = false;
      // while (p_predicate::read()) {
      // [[intel::ivdep]] while (1) {
      do {
        auto _dc = p_predicate::read(predicate_read_succ);

        int next_slot = -1;

        // Count free slots in the queue.
        uint num_free = 0;
#pragma unroll
        for (uint i = 0; i < STOREQ_ENTRIES; ++i) {
          if (!store_q[i].in_flight) {
            num_free += 1;
            next_slot = i;
          }
        }

        bool stall_stores = num_free == 0;
        bool is_store = false;
        bool is_load = false;
        uint m_load;
        idx_val_pair m_hist_store;

        // do {
        // Non blocking read/write requests associated with the same array.
        m_load = p_m_load::read(is_load);
        if (!stall_stores)
          m_hist_store = p_m_hist_store::read(is_store);
        // } while (!is_load && !is_store);

        // Load logic
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

          // If found, forward the value from the storeQ to the load.
          // Otherwise, issue load.
          if (in_flight_ld_logic)
            p_hist_out::write(hist_forward);
          else
            p_hist_out::write(hist[m_load]);
        }

        // Store logic
        if (is_store) {
          // Enqueue store.
          // store_q[0] = m_hist_store;
          LSU::store(hist.get_pointer() + m_hist_store.idx, m_hist_store.val,
                     ext::oneapi::experimental::properties(
                         ext::intel::experimental::latency_anchor_id<0>));
          p_store_ack::write(
              next_slot,
              ext::oneapi::experimental::properties(
                  ext::intel::experimental::latency_constraint<
                      0, ext::intel::experimental::latency_control_type::exact, STORE_LATENCY>));

          // fpga_tools::UnrolledLoop<STOREQ_ENTRIES>([&next_slot](auto i) {
          //   if (i == next_slot) {
          //     store_ack_pipe_array::PipeAt<i>::write(
          //         true, ext::oneapi::experimental::properties(
          //                   ext::intel::experimental::latency_constraint<
          //                       0, ext::intel::experimental::latency_control_type::exact,
          //                       STORE_LATENCY>));
          //   }
          // });

// Check if a different store to the same address is in flight.
// for (uint i = 0; i < STORE_LATENCY; ++i) {
#pragma unroll
          for (uint i = 0; i < STOREQ_ENTRIES; ++i) {
            if (m_hist_store.idx == store_q[i].idx) // invalidate value
              store_q[i].in_flight = false;

            if (i == next_slot)
              store_q[i].in_flight = true;
            //   store_ack_pipe_array::PipeAt<i>::write<
            //       ext::intel::experimental::latency_anchor_id<0>>(true);
          }

          // fpga_tools::UnrolledLoop<STOREQ_ENTRIES>([&next_slot](auto i) {
          //   if (i == next_slot) {
          //     store_ack_pipe_array::PipeAt<i>::write<
          //         sycl::ext::intel::experimental::latency_anchor_id<i>>(true);
          //   }
          // });
          // // atomic_fence(memory_order::)
        }

        // #pragma unroll
        //         for (uint i = 0; i < STOREQ_ENTRIES; ++i) {
        //           auto is_valid_signal = false;
        //           auto ack_signal =
        //           store_ack_pipe_array::read<ext::experimental::latency_constraint<
        //               0, ext::intel::experimental::type::exact, STORE_LATENCY>>(is_valid_signal);
        //           if (is_valid_signal) {
        //             store_q[i].in_flight = false;
        //           }
        //         }

        bool ack_success = false;
        auto i_ack = p_store_ack::read(ack_success);
        if (ack_success) {
          store_q[i_ack].in_flight = false;
        }

        // fpga_tools::UnrolledLoop<STOREQ_ENTRIES>([&store_q](auto i) {
        //   bool ack_success = false;
        //   // auto ack_signal =
        //   // store_ack_pipe_array::PipeAt<i>::read<ext::intel::experimental::latency_constraint<
        //   //     i, sycl::ext::intel::experimental::type::exact, STORE_LATENCY>>(ack_success);
        //   auto ack_signal = store_ack_pipe_array::PipeAt<i>::read(ack_success);
        //   if (ack_success) {
        //     store_q[i].in_flight = false;
        //   }
        // });

        any_in_flight = false;
#pragma unroll
        for (uint i = 0; i < STOREQ_ENTRIES; ++i)
          any_in_flight |= store_q[i].in_flight;

      } while (!predicate_read_succ || any_in_flight);
    });
  });


  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}
