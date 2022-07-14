#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

#ifdef __SYCL_DEVICE_ONLY__
  #define CL_CONSTANT __attribute__((opencl_constant))
#else
  #define CL_CONSTANT
#endif

#define PRINTF(format, ...) { \
            static const CL_CONSTANT char _format[] = format; \
            sycl::ext::oneapi::experimental::printf(_format, ## __VA_ARGS__); }
using PipelinedLSU = ext::intel::lsu<>;

#ifndef Q_SIZE
  #define Q_SIZE 8
#endif

constexpr uint STORE_Q_SIZE = Q_SIZE;
constexpr uint STORE_LATENCY = 12; // This should be gotten from static analysis.


struct store_entry {
  int idx; // This should be the full address in a real impl.
  int tag;
  int countdown;
  uint val;
};

struct pair {
  int fst; 
  int snd; 
};


double histogram_if_kernel(queue &q, const std::vector<uint> &feature, 
                           const std::vector<uint> &weight, std::vector<uint> &hist) {
  std::cout << "Dynamic HLS\n";

  const uint array_size = feature.size();

  buffer feature_buf(feature);
  buffer weight_buf(weight);
  buffer hist_buf(hist);

  using weight_load_pipe = pipe<class weight_load_pipe_class, uint, 64>;
  using weight_load_2_pipe = pipe<class weight_load_2_pipe_class, uint, 64>;
  using weight_load_3_pipe = pipe<class weight_load_3_pipe_class, uint, 64>;
  using idx_load_pipe = pipe<class feature_load_pipe_class, pair, 64>;
  using idx_store_pipe = pipe<class feature_store_pipe_class, pair, 64>;
  using val_load_pipe = pipe<class hist_load_pipe_class, uint, 64>;
  using val_store_pipe = pipe<class hist_store_pipe_class, uint, 64>;

  using calc_predicate_pipe = pipe<class calc_predicate_pipe_class, bool, 64>;
  using end_signal_pipe = pipe<class end_signal_pipe_class, bool>;

  q.submit([&](handler &hnd) {
    accessor weight(weight_buf, hnd, read_only);
    hnd.single_task<class LoadWeight>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) {
        uint wt = weight[i];
        weight_load_2_pipe::write(wt);
        weight_load_3_pipe::write(wt);

        if (wt > 0) {
          calc_predicate_pipe::write(1);
          weight_load_pipe::write(wt);
        }
      }
      calc_predicate_pipe::write(0);
    });
  });
  
  q.submit([&](handler &hnd) {
    accessor feature(feature_buf, hnd, read_only);
    hnd.single_task<class LoadFeature>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) {
        if (weight_load_2_pipe::read() > 0)
          idx_load_pipe::write({int(feature[i]), i});
      }
    });
  });
  q.submit([&](handler &hnd) {
    accessor feature(feature_buf, hnd, read_only);
    hnd.single_task<class LoadFeature2>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) {
        if (weight_load_3_pipe::read() > 0)
          idx_store_pipe::write({int(feature[i]), i});
      }
    });
  });
 

    q.submit([&](handler &hnd) {
    accessor hist(hist_buf, hnd, read_write);

    hnd.single_task<class LSQ>([=]() [[intel::kernel_args_restrict]] {
      [[intel::fpga_register]] store_entry store_entries[STORE_Q_SIZE];
      // A FIFO of {idx_into_store_q, idx_into_x} pairs.
      [[intel::fpga_register]] pair store_idx_fifo[STORE_Q_SIZE];

      #pragma unroll
      for (uint i=0; i<STORE_Q_SIZE; ++i) {
        store_entries[i] = {-1, -1, -1, 0};
      }

      int i_store_val = 0;
      int i_store_idx = 0;
      int store_idx_fifo_head = 0;

      int tag_store = 0;
      int idx_store;
      uint val_store;
      pair idx_tag_pair_store;

      uint val_load_1;
      int idx_load_1, tag_load_1 = 0; 
      bool try_val_load_1_pipe_write = true;
      bool is_load_1_waiting = false;
      pair idx_tag_pair_load_1;

      bool end_signal = false;

      [[intel::ivdep]] 
      while (!end_signal || i_store_idx > i_store_val) {
        /* Start Load 1 Logic */
        if (tag_load_1 <= tag_store && try_val_load_1_pipe_write) {

          bool idx_load_pipe_succ = false;
          if (!is_load_1_waiting) {
            idx_tag_pair_load_1 = idx_load_pipe::read(idx_load_pipe_succ);
          }

          if (idx_load_pipe_succ || is_load_1_waiting) {
            idx_load_1 = idx_tag_pair_load_1.fst;
            tag_load_1 = idx_tag_pair_load_1.snd;

            int max_tag = -1;
            bool _is_load_1_waiting = false;
            #pragma unroll
            for (uint i=0; i<STORE_Q_SIZE; ++i) {
              auto st_entry = store_entries[i];
              if (st_entry.idx == idx_load_1 && st_entry.tag < tag_load_1 && st_entry.tag > max_tag ) {
                _is_load_1_waiting = (st_entry.countdown == -1);
                max_tag = st_entry.tag;
                val_load_1 = st_entry.val;
              }
            }
            is_load_1_waiting = _is_load_1_waiting;

            if (!is_load_1_waiting && max_tag == -1) {
              val_load_1 = PipelinedLSU::load(hist.get_pointer() + idx_load_1);
            }

            try_val_load_1_pipe_write = is_load_1_waiting;
          }
        }
        if (!try_val_load_1_pipe_write) {
          val_load_pipe::write(val_load_1, try_val_load_1_pipe_write);
        }
        /* End Load 1 Logic */

        /* Start Store Logic */
        #pragma unroll
        for (uint i=0; i<STORE_Q_SIZE; ++i) {
          const int count = store_entries[i].countdown;
          if (count == 1) store_entries[i].idx = -1;
          if (count > 1) store_entries[i].countdown--;
        }
        int next_entry_slot = -1;
        #pragma unroll
        for (uint i=0; i<STORE_Q_SIZE; ++i) {
          if (store_entries[i].idx == -1)  {
            next_entry_slot = i;
          }
        }

        bool idx_store_pipe_succ = false;
        if (next_entry_slot != -1 && store_idx_fifo_head < STORE_Q_SIZE) {
          idx_tag_pair_store = idx_store_pipe::read(idx_store_pipe_succ);
        }
        if (idx_store_pipe_succ) {
          idx_store = idx_tag_pair_store.fst;
          tag_store = idx_tag_pair_store.snd;
          store_entries[next_entry_slot] = {idx_store, tag_store, -1};
          store_idx_fifo[store_idx_fifo_head] = {next_entry_slot, idx_store};

          i_store_idx++;
          store_idx_fifo_head++;
        }
        
        bool val_store_pipe_succ = false;
        if (i_store_idx > i_store_val) {
          val_store = val_store_pipe::read(val_store_pipe_succ);
        }
        if (val_store_pipe_succ) {
          auto entry_slot_and_idx_pair = store_idx_fifo[0];           
          PipelinedLSU::store(hist.get_pointer() + entry_slot_and_idx_pair.snd, val_store);
          store_entries[entry_slot_and_idx_pair.fst].val = val_store;
          store_entries[entry_slot_and_idx_pair.fst].countdown = STORE_LATENCY;

          #pragma unroll
          for (uint i=1; i<STORE_Q_SIZE; ++i) {
            store_idx_fifo[i-1] = store_idx_fifo[i];
          }
          store_idx_fifo_head--;

          i_store_val++;
        }
        /* End Store Logic */
      
        if (!end_signal) end_signal_pipe::read(end_signal);
      }

    });
  });

  auto event = q.submit([&](handler &hnd) {
    hnd.single_task<class Compute>([=]() [[intel::kernel_args_restrict]] {
      while (calc_predicate_pipe::read()) {
        uint wt = weight_load_pipe::read();
        uint hist = val_load_pipe::read();

        auto new_hist = hist + wt;
        val_store_pipe::write(new_hist);
      }

      end_signal_pipe::write(1);
    });
  });



  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}

