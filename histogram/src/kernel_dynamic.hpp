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
constexpr uint STORE_LATENCY = 256;

struct store_entry {
  int idx; // This should be the full address in a real impl.
  uint val;
  bool executed;
  int countdown;
};

struct pair {
  int queue_idx; 
  uint store_idx; 
};


double histogram_kernel(queue &q, const std::vector<uint> &feature, const std::vector<uint> &weight,
                        std::vector<uint> &hist) {
  std::cout << "Dynamic HLS\n";

  const uint array_size = feature.size();

  buffer feature_buf(feature);
  buffer weight_buf(weight);
  buffer hist_buf(hist);

  using weight_load_pipe = pipe<class weight_load_pipe_class, uint, 64>;
  using idx_load_pipe = pipe<class feature_load_pipe_class, uint, 64>;
  using idx_store_pipe = pipe<class feature_store_pipe_class, uint, 64>;
  using val_load_pipe = pipe<class hist_load_pipe_class, uint, 4>;
  using val_store_pipe = pipe<class hist_store_pipe_class, uint, 4>;

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
        idx_load_pipe::write(feature[i]);
    });
  });
  q.submit([&](handler &hnd) {
    accessor feature(feature_buf, hnd, read_only);
    hnd.single_task<class LoadFeature2>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) 
        idx_store_pipe::write(feature[i]);
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

      int i_store_val = 0;
      int i_store_idx = 0;
      int i_load = 0;

      uint idx_load, idx_store, val_load, val_store;
      int store_idx_fifo_head = 0;

      bool val_load_pipe_write_succ = true;
      bool is_load_waiting_for_val = false;

      [[intel::ivdep]]
      while (i_store_val < array_size) {
        /* Start Load Logic */
        if (i_load < array_size && i_load <= i_store_idx && val_load_pipe_write_succ) {
          bool idx_load_pipe_succ = false;

          if (!is_load_waiting_for_val) {
            idx_load = idx_load_pipe::read(idx_load_pipe_succ);
          }

          if (idx_load_pipe_succ || is_load_waiting_for_val) {
            bool is_in_flight = false;

            #pragma unroll
            for (uint i=0; i<STORE_Q_SIZE; ++i) {
              if (store_entries[i].idx == idx_load && store_entries[i].executed) {
                is_in_flight |= true;
                val_load = store_entries[i].val;
              }
              else if (store_entries[i].idx == idx_load && !store_entries[i].executed) {
                is_load_waiting_for_val |= true;
              }
            }

            if (!is_in_flight && !is_load_waiting_for_val) {
              val_load = hist[idx_load];
            }

            if (!is_load_waiting_for_val) {
              val_load_pipe::write(val_load, val_load_pipe_write_succ);
              if (val_load_pipe_write_succ) {
                i_load++;
              }
            }
          }
        }
        else if (i_load < array_size && i_load <= i_store_idx && !val_load_pipe_write_succ) {
          if (!val_load_pipe_write_succ && !is_load_waiting_for_val) {
            val_load_pipe::write(val_load, val_load_pipe_write_succ);
            if (val_load_pipe_write_succ) {
              i_load++;
            }
          }
        }
        /* End Load Logic */
      

        /* Start Store Logic */
        int next_entry_slot = -1;
        #pragma unroll
        for (uint i=0; i<STORE_Q_SIZE; ++i) {
          if (store_entries[i].executed) {
            store_entries[i].countdown--;
          }

          if (store_entries[i].countdown <= 0) {
            next_entry_slot = i;
          }
        }

        bool idx_store_pipe_succ = false;
        if (next_entry_slot != -1 && store_idx_fifo_head < STORE_Q_SIZE) {
          idx_store = idx_store_pipe::read(idx_store_pipe_succ);
        }

        if (idx_store_pipe_succ) {
          store_entries[next_entry_slot].idx = idx_store;
          store_entries[next_entry_slot].countdown = STORE_LATENCY;

          store_idx_fifo[store_idx_fifo_head] = {next_entry_slot, idx_store};

          i_store_idx++;
          store_idx_fifo_head++;
        }
        
        bool val_store_pipe_succ = false;
        if (i_store_idx > i_store_val) {
          val_store = val_store_pipe::read(val_store_pipe_succ);
        }

        if (val_store_pipe_succ) {
          auto entry_store_idx_pair = store_idx_fifo[0];           
          hist[entry_store_idx_pair.store_idx] = val_store;
          store_entries[entry_store_idx_pair.queue_idx].executed = true;

          #pragma unroll
          for (uint i=1; i<STORE_Q_SIZE; ++i) {
            store_idx_fifo[i-1] = store_idx_fifo[i];
          }
          store_idx_fifo_head--;

          i_store_val++;
        }
        /* End Store Logic */
      }

    });
  });

  q.submit([&](handler &hnd) {
    hnd.single_task<class Compute>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) {
        uint wt = weight_load_pipe::read();
        uint hist = val_load_pipe::read();

        auto new_hist = hist + wt;

        val_store_pipe::write(new_hist);
      }
    });
  });



  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}

