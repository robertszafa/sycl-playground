
#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <utility>

#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

using PipelinedLSU = ext::intel::lsu<>;
using BurstCoalescedLSU = ext::intel::lsu<ext::intel::burst_coalesce<false>, 
                                          ext::intel::statically_coalesce<false>>;

// This should be STORE_Q_SIZE >= STORE_LATENCY
constexpr uint STORE_Q_SIZE = Q_SIZE;
constexpr uint STORE_LATENCY = 72;

struct store_entry {
  uint idx; // This should be the full address in a real impl.
  uint val;
  bool executed;
  int countdown;
};

struct m_store_entry {
  uint idx_hist;
  uint idx_store_q;
};



double histogram_kernel(queue &q, const std::vector<uint> &feature, const std::vector<uint> &weight,
                        std::vector<uint> &hist) {
  std::cout << "Dynamic HLS\n";

  const uint array_size = feature.size();

  buffer feature_buf(feature);
  buffer weight_buf(weight);
  buffer hist_buf(hist);

  using weight_load_pipe = pipe<class weight_load_pipe_class, uint, 16>;
  using m_load_pipe = pipe<class m_load_pipe_class, uint, 16>;
  using m_store_pipe = pipe<class m_store_pipe_class, uint, 16>;
  using hist_load_pipe = pipe<class hist_load_pipe_class, uint, 16>;

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
    accessor feature(feature_buf, hnd, read_only);

    hnd.single_task<class LoadFeature>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) {
        uint m = feature[i];
        m_load_pipe::write(m);
      }
    });
  });

  q.submit([&](handler &hnd) {
    accessor feature(feature_buf, hnd, read_only);

    hnd.single_task<class LoadFeature2>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) {
        uint m = feature[i];
        m_store_pipe::write(m);
      }
    });
  });
 
  q.submit([&](handler &hnd) {
    accessor hist(hist_buf, hnd, read_write);
    // accessor feature(feature_buf, hnd, read_only);

    hnd.single_task<class LoadStoreHist>([=]() [[intel::kernel_args_restrict]] {
      [[intel::fpga_register]] store_entry store_q[STORE_Q_SIZE];

      // constexpr uint M_STORE_Q_SIZE = 16;
      // Keeps track of m_store into hist, and free_slot into store_q
      // [[intel::fpga_register]] m_store_entry m_store_q[M_STORE_Q_SIZE];

      // All entries are invalid at the start.
      #pragma unroll
      for (uint i=0; i<STORE_Q_SIZE; ++i) {
        store_q[i].idx = (uint) -1;
        store_q[i].countdown = -1;
        store_q[i].executed = false;
      }

      uint i_load = 0;
      uint i_store = 0;
      uint enqued_idx_counter = 0;
      uint enqued_val_counter = 0;

      bool is_load_waiting_for_executed = false;
      uint m_load;

      [[intel::ivdep]]
      // for (int i_top = 0; i_top < array_size; ++i_top) {
      while (i_store < array_size) {
        sycl::ext::oneapi::experimental::printf("New iter i_load %d   i_store %d\n", i_load, i_store);

        if (i_load < array_size) {
          if (!is_load_waiting_for_executed)
            m_load = m_load_pipe::read();

          bool forward = false;
          uint hist_val;

          // TODO: Do we need to explicitly pick the entry with the hisghest idx (latest store)? 
          #pragma unroll
          for (uint i=0; i<STORE_Q_SIZE; ++i) {
            if (store_q[i].executed && store_q[i].idx == ext::intel::fpga_reg(m_load)) {
              forward = true;
              hist_val = store_q[i].val;
              is_load_waiting_for_executed = false;
            }
            else if (!store_q[i].executed && store_q[i].idx == ext::intel::fpga_reg(m_load)) {
              is_load_waiting_for_executed |= true;
            }
            else {
              is_load_waiting_for_executed |= false;
            }
          }

          if (!forward || !is_load_waiting_for_executed) {
            is_load_waiting_for_executed = false;

            if (!forward) {
              hist_val = PipelinedLSU::load(hist.get_pointer() + m_load);
              // sycl::ext::oneapi::experimental::printf("Load %d\n", m_load);
            }
            
            hist_load_pipe::write(hist_val);

            i_load++;
          }
        }
        
        bool to_shift = false;
        // Countdown store latency for executed entries
        #pragma unroll
        for (uint i=0; i<STORE_Q_SIZE; ++i) {
          if (store_q[i].executed) {
            store_q[i].countdown -= 1;
            if (store_q[i].countdown <= 0) {
              to_shift |= true;
            }
          }
        }

        sycl::ext::oneapi::experimental::printf("Countdown q[0]: %d\n", store_q[0].countdown);
        sycl::ext::oneapi::experimental::printf("Executed q[0]: %d\n", store_q[0].executed);
        if (to_shift) {
          // Shift store queue.
          #pragma unroll
          for (uint i = 1; i < STORE_Q_SIZE; ++i) {
            store_q[i - 1] = store_q[i];
          }

          enqued_idx_counter -= 1;
          enqued_val_counter -= 1;
          store_q[enqued_idx_counter] = store_entry{(uint) -1, 0, false, -1};
        }

        bool m_store_pipe_succ = false;
        uint m_store;
        if (enqued_idx_counter < STORE_Q_SIZE) {
          m_store = m_store_pipe::read(m_store_pipe_succ);
        }
        if (m_store_pipe_succ) {
          // Enque the index in the store queue.
          store_q[enqued_idx_counter].idx = m_store;
          store_q[enqued_idx_counter].countdown = STORE_LATENCY;
          store_q[enqued_idx_counter].executed = false;
          
          sycl::ext::oneapi::experimental::printf("Store 1: %d\n", m_store);

          // Index FIFO to store the value in the correct slot once it's is ready.
          // store_q[num_enqued_stores].idx_hist = m_store;
          // store_q[num_enqued_stores].idx_store_q = free_slot;
          enqued_idx_counter++;
        }

        if (enqued_idx_counter > 0) {
          bool store_pipe_succ = false;
          auto new_hist = hist_store_pipe::read(store_pipe_succ);

          sycl::ext::oneapi::experimental::printf("read store_pipe_succ: %d\n", store_pipe_succ);
          if (store_pipe_succ) {
            auto next_m_to_store = store_q[enqued_val_counter];

            hist[next_m_to_store.idx] = new_hist;
            store_q[enqued_val_counter].val = new_hist; 
            store_q[enqued_val_counter].executed = true;
            i_store++;
            enqued_val_counter++;

            sycl::ext::oneapi::experimental::printf("Store 2: %d\n", next_m_to_store.idx);
            // // Shift FIFO.
            // #pragma unroll
            // for (uint i=0; i<M_STORE_Q_SIZE-1; ++i) {
            //   m_store_q[0] = m_store_q[i + 1];
            // }
            // num_enqued_stores -= 1;
          }
        }
      
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

