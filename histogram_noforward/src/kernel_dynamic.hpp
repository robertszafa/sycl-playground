#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

// The default PipelinedLSU will start a load/store immediately, which the memory disambiguation 
// logic relies upon.
// A BurstCoalescedLSU would instead of waiting for more requests to arrive for a coalesced access.
using PipelinedLSU = ext::intel::lsu<>;


#ifndef Q_SIZE
  #define Q_SIZE 8
#endif

constexpr uint STORE_Q_SIZE = Q_SIZE;
constexpr uint STORE_LATENCY = 400; // This should be gotten from static analysis.


struct store_entry {
  int idx; // This should be the full address in a real impl.
  uint val;
  bool executed;
  int countdown;
  int tag;
};

constexpr store_entry INVALID_ENTRY = {-1, 0, false, -1, -1};

template<typename T1, typename T2>
struct pair {
  T1 fst; 
  T2 snd; 
};

struct triple {
  uint fst; 
  uint snd; 
  uint thrd; 
};

struct candidate {
  int tag; 
  bool forward;
};


double histogram_kernel(queue &q, const std::vector<uint> &feature, const std::vector<uint> &weight,
                        std::vector<uint> &hist) {
  std::cout << "Dynamic HLS\n";

  const uint array_size = feature.size();

  buffer feature_buf(feature);
  buffer weight_buf(weight);
  buffer hist_buf(hist);

  using load_pair = pair<uint, uint>;

  constexpr uint PIPE_D = 64;
  using val_load_pipe = pipe<class hist_load_pipe_class, uint, PIPE_D>;
  using idx_load_pipe = pipe<class feature_load_pipe_class, load_pair, PIPE_D>;
  using store_pipe = pipe<class hist_store_pipe_class, triple, PIPE_D>;


  q.submit([&](handler &hnd) {
    accessor hist(hist_buf, hnd, read_write);

    hnd.single_task<class LoadStoreHist>([=]() [[intel::kernel_args_restrict]] {
      using store_entry = pair<int, int>;
      [[intel::fpga_register]] store_entry store_entries[STORE_Q_SIZE];

      #pragma unroll
      for (int i = 0; i < STORE_Q_SIZE; ++i) {
        store_entries[i] = {-1, 0};
      }

      int i_store_val = 0;
      int i_store_idx = 0;
      int i_load = 0;
      
      uint load_tag = 0;
      uint store_tag = 0;

      int idx_load, idx_store;
      uint prime_store, prime_load, val_load, val_store;
      int store_idx_fifo_head = 0;

      bool val_load_pipe_write_succ = true;
      bool is_load_waiting = false;

      [[intel::ivdep]] 
      while (i_store_val < array_size) {
        if (val_load_pipe_write_succ && load_tag <= store_tag) {
          bool load_pipe_succ = false;
          load_pair load_pair;
          if (!is_load_waiting) {
            load_pair = idx_load_pipe::read(load_pipe_succ);  
          }

          if (load_pipe_succ || is_load_waiting) {
            idx_load = load_pair.fst;
            load_tag = load_pair.snd;
            is_load_waiting = false;

            #pragma unroll
            for (int i = 0; i < STORE_Q_SIZE; ++i) {
              is_load_waiting |= (store_entries[i].fst == idx_load);
            }

            // sycl_print("idx_load %d  -  is_load_waiting %d\n", idx_load, is_load_waiting);

            if (!is_load_waiting) {
              val_load = PipelinedLSU::load(hist.get_pointer() + idx_load);
              val_load_pipe_write_succ = false;
            }
          }
        }
        if (!val_load_pipe_write_succ) {
          val_load_pipe::write(val_load, val_load_pipe_write_succ);
        }

        int next_slot = -1;
        #pragma unroll
        for (int i = 0; i < STORE_Q_SIZE; ++i) {
          auto st_entry = store_entries[i];

          if (st_entry.snd > 0) {
            store_entries[i].snd--;
          }

          if (st_entry.snd <= 1) {
            store_entries[i].fst = -1;
            next_slot = i;
          }
        }

        bool store_pipe_succ = false;
        if (next_slot != -1) {
          auto store_triple = store_pipe::read(store_pipe_succ);

          if (store_pipe_succ) {
            idx_store = store_triple.fst;
            val_store = store_triple.snd;
            store_tag = store_triple.thrd;
            store_entries[next_slot] = {(int) idx_store, (int) STORE_LATENCY};

            PipelinedLSU::store(hist.get_pointer() + idx_store, val_store);

            i_store_val++;
          }
        }
      }

      // ext::oneapi::experimental::printf("Done LSQ\n");
    });
  });

  event e = q.submit([&](handler &hnd) {
    accessor feature(feature_buf, hnd, read_only);
    accessor weight(weight_buf, hnd, read_only);

    hnd.single_task<class Compute>([=]() [[intel::kernel_args_restrict]] {
      uint tag = 0;
      for (int i = 0; i < array_size; ++i) {
        // ext::oneapi::experimental::printf("Iter %d\n", i);

        uint wt = weight[i];
        uint idx = feature[i];

        idx_load_pipe::write({idx, tag});
        uint hist = val_load_pipe::read();

        auto new_hist = hist + wt;
        store_pipe::write({idx, new_hist, tag});

        tag++;
      }
        // ext::oneapi::experimental::printf("Done calc\n");
    });
  });



  auto start = e.get_profiling_info<info::event_profiling::command_start>();
  auto end = e.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}

