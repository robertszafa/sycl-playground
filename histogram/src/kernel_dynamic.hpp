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

constexpr uint STORE_Q_SIZE = Q_SIZE;
constexpr uint STORE_LATENCY = 128;

struct store_entry {
  int idx; // This should be the full address in a real impl.
  uint val;
  bool executed;
  int countdown;
};

struct pair {
  int idx_q; 
  uint idx_hist; 
};


double histogram_kernel(queue &q, const std::vector<uint> &feature, const std::vector<uint> &weight,
                        std::vector<uint> &hist) {
  std::cout << "Dynamic HLS\n";

  const uint array_size = feature.size();

  buffer feature_buf(feature);
  buffer weight_buf(weight);
  buffer hist_buf(hist);

  using weight_load_pipe = pipe<class weight_load_pipe_class, uint, 64>;
  using feature_load_pipe = pipe<class feature_load_pipe_class, uint, 64>;
  using feature_store_pipe = pipe<class feature_store_pipe_class, uint, 64>;
  using hist_load_pipe = pipe<class hist_load_pipe_class, uint, 4>;
  using hist_store_pipe = pipe<class hist_store_pipe_class, uint, 4>;

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
      for (int i = 0; i < array_size; ++i) 
        feature_load_pipe::write(feature[i]);
    });
  });
  q.submit([&](handler &hnd) {
    accessor feature(feature_buf, hnd, read_only);
    hnd.single_task<class LoadFeature2>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) 
        feature_store_pipe::write(feature[i]);
    });
  });
 

  q.submit([&](handler &hnd) {
    accessor hist(hist_buf, hnd, read_write);

    hnd.single_task<class LoadStoreHist>([=]() [[intel::kernel_args_restrict]] {
      [[intel::fpga_register]] store_entry store_entries[STORE_Q_SIZE];
      // A FIFO of {idx_into_store_q, idx_into_hist} pairs.
      [[intel::fpga_register]] pair store_idx_fifo[STORE_Q_SIZE];

      // All entries are invalid at the start.
      #pragma unroll
      for (uint i=0; i<STORE_Q_SIZE; ++i) {
        store_entries[i] = {-1, 0, false, -1};
      }

      uint i_store_vals = 0;
      uint i_store_idxs = 0;
      uint i_load = 0;
      uint store_idx_fifo_head = 0;

      // Is load written in the client pipe?
      bool load_returned = true;
      bool load_waiting_for_val = false;
      uint load_val;

      [[intel::ivdep]]
      while (i_store_vals < array_size) {
        
        // sycl::ext::oneapi::experimental::printf("\n -- Iter --\n");

        /* Load logic */
        if (i_load < array_size && i_store_idxs > i_load && load_returned) {// && store_idx_fifo_head < STORE_Q_SIZE) {
          bool load_idx_pipe_succ = false;
          uint load_idx;
          
          // TODO: tag each load and store idx with the iteration/priority.
          if (!load_waiting_for_val)
            load_idx = feature_load_pipe::read(load_idx_pipe_succ);

          if (load_idx_pipe_succ || load_waiting_for_val) {
            bool forward = false;

            // Maybe forward and load_waiting_for_val are both true sometimes, leading to deadlock?
            #pragma unroll
            for (uint i=0; i<STORE_Q_SIZE; ++i) {
              if (store_entries[i].idx == load_idx && store_entries[i].executed) {
                forward |= true;
                load_val = store_entries[i].val;
              }
              else if (store_entries[i].idx == load_idx && !store_entries[i].executed) {
                load_waiting_for_val |= true;
              }
            }

            if (load_waiting_for_val) goto end_logic_label;

            if (!forward) {
              load_val = PipelinedLSU::load(hist.get_pointer() + load_idx);
            }
            load_returned = false;
            load_waiting_for_val = false;
          }
        }
        if (i_load < array_size && !load_returned) {
          // Keep trying to return load once on every iteration, until successful.
          hist_load_pipe::write(load_val, load_returned);
          if (load_returned) {
            i_load += 1;
          }
        }
        end_logic_label:
        /* End Load logic */

        /* Store logic */
        // Countdown stores in progress.
        int next_q_slot = -1;
        #pragma unroll
        for (uint i=0; i<STORE_Q_SIZE; ++i) {
          if (store_entries[i].executed) {
            store_entries[i].countdown -= 1;
          }

          if (store_entries[i].countdown <= 0) {
            next_q_slot = i;
            store_entries[i] = {-1, 0, false, -1};
          }
        }

        // Non blocking store_idx read.
        uint store_idx;
        bool store_idx_pipe_succ = false;
        if (next_q_slot != -1 && store_idx_fifo_head < STORE_Q_SIZE) {
          store_idx = feature_store_pipe::read(store_idx_pipe_succ);
        }

        if (store_idx_pipe_succ) {
          store_entries[next_q_slot].idx = store_idx;
          store_entries[next_q_slot].countdown = STORE_LATENCY;

          store_idx_fifo[store_idx_fifo_head] = {next_q_slot, store_idx};

          store_idx_fifo_head += 1;
          i_store_idxs += 1;
        }

        // Non blocking store_val read.
        bool store_val_pipe_succ = false;
        uint store_val;
        if (i_store_idxs > i_store_vals) {
          store_val = hist_store_pipe::read(store_val_pipe_succ);
        }
        
        if (store_val_pipe_succ) {
          auto q_idx_hist_idx_pair = store_idx_fifo[0];

          // sycl::ext::oneapi::experimental::printf("Stored hist[ %d ]\n", q_idx_hist_idx_pair.idx_hist);
          // sycl::ext::oneapi::experimental::printf("Q slot %d\n", q_idx_hist_idx_pair.idx_q);
          hist[q_idx_hist_idx_pair.idx_hist] = store_val;
          store_entries[q_idx_hist_idx_pair.idx_q].executed = true;

          #pragma unroll
          for (uint i=1; i<STORE_Q_SIZE; ++i) {
            store_idx_fifo[i-1] = store_idx_fifo[i];
          }

          store_idx_fifo_head -= 1;
          i_store_vals += 1;
        }
        /* End Store logic */
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

