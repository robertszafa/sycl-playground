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

struct store_entry {
  uint idx;
  uint val;
  int count_down;
};

double histogram_kernel(queue &q, const std::vector<uint> &feature, const std::vector<uint> &weight,
                        std::vector<uint> &hist) {
  std::cout << "Dynamic HLS with queue_size = " << STOREQ_ENTRIES << "\n";

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
  using p_to_load = pipe<class p_to_load_class, uint, STORE_LATENCY>;
  
  using p_to_bypass = pipe<class p_to_bypass_class, uint, STORE_LATENCY>;


  auto event = q.submit([&](handler &hnd) {
    accessor feature(feature_buf, hnd, read_only);
    accessor weight(weight_buf, hnd, read_only);

    hnd.single_task<HistogramKernel>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) {
        auto wt = weight[i];
        auto m = feature[i];

        p_m_load::write(m);

        // sycl::ext::oneapi::experimental::printf("Issued load \n");
        auto x = p_hist_out::read();
        auto x_new = x + wt;
        // sycl::ext::oneapi::experimental::printf("Received value \n");
        p_m_hist_store::write(idx_val_pair{m, x_new});
        // sycl::ext::oneapi::experimental::printf("Issued store \n");
      }

      p_predicate::write(false);
      // sycl::ext::oneapi::experimental::printf("Finished \n");
    });
  });

  q.submit([&](handler &hnd) {
    hnd.single_task<class HistLSQ>([=]() [[intel::kernel_args_restrict]] {
      [[intel::fpga_register]] store_entry store_q[STOREQ_ENTRIES];

      bool terminate_signal = false;
      bool any_in_flight = false;
      uint m_load;
      uint hist_loaded;

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

        /* Load logic */
        bool is_load = false;
        m_load = p_m_load::read(is_load);

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

          if (in_flight_ld_logic) {  // If yes, use its value.
            hist_loaded = hist_forward;
            p_to_bypass::write(hist_loaded);
          }
          else { // Otherwise, issue load
            p_to_load::write(m_load);
          }
        }
        /* End Load logic */

        /* Store logic */
        bool is_store = false;
        idx_val_pair m_hist_store;

        // Only accept new store entry if there's space for it. If not, it will wait in the FIFO.
        bool stall_store = next_slot == -1;
        if (!stall_store)
          m_hist_store = p_m_hist_store::read(is_store);

        // Issue load and start countdown (todo: use latency control API here, once available).
        if (is_store) { 
          p_to_store::write(m_hist_store);
          store_q[next_slot] = store_entry{m_hist_store.idx, m_hist_store.val, STORE_LATENCY};
        }
        /* End Store logic */

        any_in_flight = false;
        #pragma unroll
        for (uint i = 0; i < STOREQ_ENTRIES; ++i)
          any_in_flight |= (store_q[i].count_down > 0);

        if (!terminate_signal)
          auto _dc = p_predicate::read(terminate_signal);

      } while (!terminate_signal || any_in_flight);

      p_predicate_port::write(false);
      // sycl::ext::oneapi::experimental::printf("Finished 2\n");
    });
  });
 

  q.submit([&](handler &hnd) {
    accessor hist(hist_buf, hnd, read_write);

    hnd.single_task<class ActualLoad>([=]() [[intel::kernel_args_restrict]] {
      using PipelinedLSU = cl::sycl::ext::intel::lsu<>;
      using PipelinedLSU2 = cl::sycl::ext::intel::lsu<>;
      bool lsu_flag = false;

      bool is_finish = false;
      [[intel::ivdep]] 
      do {
        bool suc_bypass_pipe = false;
        auto bypass_val = p_to_bypass::read(suc_bypass_pipe);
        if (suc_bypass_pipe) {
          p_hist_out::write(bypass_val);
        }

        bool suc_load_pipe = false;
        auto m_to_load = p_to_load::read(suc_load_pipe);
        if (suc_load_pipe) {
          uint load_val;
          if (lsu_flag)
            load_val = PipelinedLSU::load(hist.get_pointer() + m_to_load);
          else
            load_val = PipelinedLSU2::load(hist.get_pointer() + m_to_load);

          lsu_flag = !lsu_flag;
          p_hist_out::write(load_val);
        }
        
        bool suc_store_pipe = false;
        auto m_hist_to_store = p_to_store::read(suc_store_pipe);
        if (suc_store_pipe) {
          hist[m_hist_to_store.idx] = m_hist_to_store.val;
        }

        p_predicate_port::read(is_finish);
      } while (!is_finish);

    });
  });


  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}
