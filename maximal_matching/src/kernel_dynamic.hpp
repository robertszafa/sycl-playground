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

#define PRINTF(format, ...) //{ \
            static const CL_CONSTANT char _format[] = format; \
            sycl::ext::oneapi::experimental::printf(_format, ## __VA_ARGS__); }

// The default PipelinedLSU will start a load/store immediately, which the memory disambiguation 
// logic relies upon.
// A BurstCoalescedLSU would instead of waiting for more requests to arrive for a coalesced access.
using PipelinedLSU = ext::intel::lsu<>;

#ifndef Q_SIZE
  #define Q_SIZE 8
#endif

constexpr uint STORE_Q_SIZE = Q_SIZE;
constexpr uint STORE_LATENCY = 12; // This should be gotten from static analysis.

struct pair {
  int fst; 
  int snd;
};

struct triple {
  int fst; 
  int snd;
  int thrd;
};

struct store_entry {
  int idx; // This should be the full address in a real impl.
  int tag0;
  int tag1;
  bool waiting_for_val;
  int countdown;
  int val;
};



double maximal_matching_kernel(queue &q, const std::vector<int> &edges, std::vector<int> &vertices,
                               int *out, const int num_edges) {
  std::cout << "Dynamic HLS\n";

  buffer edges_buf(edges);
  buffer vertices_buf(vertices);
  buffer out_buf(out, range(1));
  
  using u_pipe = pipe<class u_load_forked_pipe_class, int, 64>;
  using v_pipe = pipe<class v_load_forked_pipe_class, int, 64>;
  using u_load_pipe = pipe<class u_load_pipe_class, triple, 64>;
  using v_load_pipe = pipe<class v_load_pipe_class, triple, 64>;
  using vertex_u_pipe = pipe<class vertex_u_pipe_class, int, 64>;
  using vertex_v_pipe = pipe<class vertex_v_pipe_class, int, 64>;

  using u_store_idx_pipe = pipe<class u_store_idx_pipe_class, triple, 64>;
  using v_store_idx_pipe = pipe<class v_store_idx_pipe_class, triple, 64>;
  using u_store_val_pipe = pipe<class u_store_val_pipe_class, int, 64>;
  using v_store_val_pipe = pipe<class v_store_val_pipe_class, int, 64>;

  using store_idx_pipe = pipe<class store_idx_pipe_class, triple, 64>;
  using store_val_pipe = pipe<class store_val_pipe_class, int, 64>;

  using end_lsq_signal_pipe = pipe<class end_lsq_signal_pipe_class, bool>;
  
  using val_merge_pred_pipe = pipe<class val_merge_pred_class, bool, 64>;

  q.submit([&](handler &hnd) {
    accessor edges(edges_buf, hnd, read_only);

    hnd.single_task<class LoadEdges>([=]() [[intel::kernel_args_restrict]] {
      int i = 0;

      while (i < num_edges) {
        int j = i * 2;

        int u = edges[j];
        int v = edges[j + 1];

        u_pipe::write(u);
        v_pipe::write(v);

        i = i + 1;
      }
    });
  });

  q.submit([&](handler &hnd) {
    accessor edges(edges_buf, hnd, read_only);

    hnd.single_task<class LoadEdges2>([=]() [[intel::kernel_args_restrict]] {
      int i = 0;

      while (i < num_edges) {
        int j = i * 2;

        int u = edges[j];
        int v = edges[j + 1];

        u_load_pipe::write({u, i, 0});
        v_load_pipe::write({v, i, 1});

        i = i + 1;
      }
    });
  });
  
  q.submit([&](handler &hnd) {
    accessor vertices(vertices_buf, hnd, read_write);

    hnd.single_task<class LSQ>([=]() [[intel::kernel_args_restrict]] {
      [[intel::fpga_register]] store_entry store_entries[STORE_Q_SIZE];

      #pragma unroll
      for (uint i=0; i<STORE_Q_SIZE; ++i) {
        store_entries[i] = {-1, -2, -2};
      }

      int i_store_val = 0;
      int i_store_idx = 0;
      // Points to store_q entry where the next store value should be read in.
      int stq_tail = 0;
      // Points to store_q entry where the next store index should be read in.
      int stq_head = 0;
      // Two tags because there could be two stores in the same iteration.
      // In general (nested loops, multiple stores in same scope, etc.) the tag is an n-tuple.
      int tag0_store = -1;
      int tag1_store = -1;
      int idx_store;
      int val_store;
      triple idx_tag_pair_store;

      int val_load_1;
      int idx_load_1; 
      int tag_load_1 = -1; 
      bool consumer_load_1_succ = true;
      bool is_load_1_waiting = false;
      triple idx_tag_pair_load_1;

      int val_load_2;
      int idx_load_2;
      int tag_load_2 = -1; 
      bool consumer_load_2_succ = true;
      bool is_load_2_waiting = false;
      triple idx_tag_pair_load_2;

      bool end_signal = false;

      int max_tag_store = tag0_store;

      [[intel::ivdep]] 
      while (!end_signal) {
        // Load tags should not overtake the most recent store_idx tag.
        // Only advance max_store_tag once *all* stores in that iter finish.
        if (tag1_store == 1) max_tag_store = tag0_store;

        /* Start Load 1 Logic */
        if ((tag_load_1 <= max_tag_store && consumer_load_1_succ) || is_load_1_waiting) {
          // Check for new ld requests.
          bool idx_load_pipe_succ = false;
          if (!is_load_1_waiting) {
            idx_tag_pair_load_1 = u_load_pipe::read(idx_load_pipe_succ);
          }

          // If new ld request, or if we are still waiting for a previous one, then check store_q.
          if (idx_load_pipe_succ || is_load_1_waiting) {
            idx_load_1 = idx_tag_pair_load_1.fst;
            tag_load_1 = idx_tag_pair_load_1.snd;

            is_load_1_waiting = false;
            pair max_tag = {-1, -1}; 
            #pragma unroll
            for (uint i=0; i<STORE_Q_SIZE; ++i) {
              auto st_entry = store_entries[i];
              // If found, make sure it's the youngest store occuring before this ld. 
              if (st_entry.idx == idx_load_1 && st_entry.tag0 < tag_load_1 && 
                  st_entry.tag0 >= max_tag.fst && st_entry.tag1 > max_tag.snd) {
                is_load_1_waiting = st_entry.waiting_for_val;
                val_load_1 = st_entry.val;
                max_tag = {st_entry.tag0, st_entry.tag1};
              }
            }

            if (!is_load_1_waiting && max_tag.fst == -1) {
              // Not found in store_q, so issue ld.
              val_load_1 = PipelinedLSU::load(vertices.get_pointer() + idx_load_1);
            }

            // Setting 'consumer_load_succ' to false forces a write to consumer pipe.
            consumer_load_1_succ = is_load_1_waiting;
          }
        }
        if (!consumer_load_1_succ) {
          vertex_u_pipe::write(val_load_1, consumer_load_1_succ);
        }
        /* End Load 1 Logic */

        /* Start Load 2 Logic */
        if ((tag_load_2 <= max_tag_store && consumer_load_2_succ) || is_load_2_waiting) {
          // Check for new ld requests.
          bool idx_load_pipe_succ = false;
          if (!is_load_2_waiting) {
            idx_tag_pair_load_2 = v_load_pipe::read(idx_load_pipe_succ);
          }

          // If new ld request, or if we are still waiting for a previous one, then check store_q.
          if (idx_load_pipe_succ || is_load_2_waiting) {
            idx_load_2 = idx_tag_pair_load_2.fst;
            tag_load_2 = idx_tag_pair_load_2.snd;

            is_load_2_waiting = false;
            pair max_tag = {-1, -1}; 
            #pragma unroll
            for (uint i=0; i<STORE_Q_SIZE; ++i) {
              auto st_entry = store_entries[i];
              // If found, make sure it's the youngest store occuring before this ld. 
              if (st_entry.idx == idx_load_2 && st_entry.tag0 < tag_load_2 && 
                  st_entry.tag0 >= max_tag.fst && st_entry.tag1 > max_tag.snd) {
                is_load_2_waiting = st_entry.waiting_for_val;
                val_load_2 = st_entry.val;
                max_tag = {st_entry.tag0, st_entry.tag1};
              }
            }

            if (!is_load_2_waiting && max_tag.fst == -1) {
              // Not found in store_q, so issue ld.
              val_load_2 = PipelinedLSU::load(vertices.get_pointer() + idx_load_2);
            }

            // Setting 'consumer_load_succ' to false forces a write to consumer pipe.
            consumer_load_2_succ = is_load_2_waiting;
          }
        }
        if (!consumer_load_2_succ) {
          vertex_v_pipe::write(val_load_2, consumer_load_2_succ);
        }
        /* End Load 2 Logic */
      
        /* Start Store 1 Logic */
        #pragma unroll
        for (uint i=0; i<STORE_Q_SIZE; ++i) {
          // Decrement count, and invalidate idexes if count below 0.
          const int count = store_entries[i].countdown;
          if (count > 0) store_entries[i].countdown--;
          if (count <= 1 && !store_entries[i].waiting_for_val) store_entries[i].idx = -1;
        }

        // If store_q not full, check for new store_idx requests.
        bool is_stq_full = (stq_head + 1) == stq_tail;
        if (!is_stq_full) {
          bool idx_store_pipe_succ = false;
          idx_tag_pair_store = store_idx_pipe::read(idx_store_pipe_succ);

          if (idx_store_pipe_succ) {
            idx_store = idx_tag_pair_store.fst;
            tag0_store = idx_tag_pair_store.snd;
            tag1_store = idx_tag_pair_store.thrd;

            // Requests with idx_store=-1 are only sent to update the store tag 
            // (lets loads know that this iteration doesn't store anything (e.g. conditional store)).
            if (idx_store != -1) {
              store_entries[stq_head] = {idx_store, tag0_store, tag1_store, true};

              i_store_idx++;
              stq_head = (stq_head+1) % STORE_Q_SIZE;
            }
          }
        }
        
        // If we have more store indexes read than store values, then check for new store_vals.
        if (i_store_idx > i_store_val) {
          bool val_store_pipe_succ = false;
          val_store = store_val_pipe::read(val_store_pipe_succ);

          if (val_store_pipe_succ) {
            // Add value to corresponding store entry, store to mem, and start counter.
            PipelinedLSU::store(vertices.get_pointer() + store_entries[stq_tail].idx, val_store);
            store_entries[stq_tail].val = val_store;
            store_entries[stq_tail].countdown = STORE_LATENCY;
            store_entries[stq_tail].waiting_for_val = false;

            i_store_val++;
            stq_tail = (stq_tail+1) % STORE_Q_SIZE;
          }
        }
        /* End Store 1 Logic */
      
        if (!end_signal) end_lsq_signal_pipe::read(end_signal);
      }

      PRINTF("Finished Store Q\n")

    });
  });

  q.submit([&](handler &hnd) {
    hnd.single_task<class StoreIdxMerge>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < num_edges*2; ++i) {
        if (i % 2 == 0)
          store_idx_pipe::write(u_store_idx_pipe::read());
        else
          store_idx_pipe::write(v_store_idx_pipe::read());
      }

      PRINTF("Finished Merge Idx\n")
    });
  });
  
  q.submit([&](handler &hnd) {
    hnd.single_task<class StoreValMerge>([=]() [[intel::kernel_args_restrict]] {
      while (val_merge_pred_pipe::read()) {
        for (int i=0; i<2; i++) {
          if (i % 2 == 0)
            store_val_pipe::write(u_store_val_pipe::read());
          else
            store_val_pipe::write(v_store_val_pipe::read());
        }
      }

      end_lsq_signal_pipe::write(1);
      PRINTF("Finished Merge Vals\n")
    });
  });

  auto event = q.submit([&](handler &hnd) {
    accessor out_pointer(out_buf, hnd, write_only);

    hnd.single_task<class Calculation>([=]() [[intel::kernel_args_restrict]] {
      int i = 0;
      int out_res = 0;
      while (i < num_edges) {

        auto u = u_pipe::read();
        auto v = v_pipe::read();
        PRINTF("-- Iter %d: u %d   v %d \n", i, u, v);
        
        auto vertex_u = vertex_u_pipe::read();
        auto vertex_v = vertex_v_pipe::read();

        if ((vertex_u < 0) && (vertex_v < 0)) {
          PRINTF("TO SQ u %d v %d\n", u, v);
          u_store_idx_pipe::write({u, i, 0});
          v_store_idx_pipe::write({v, i, 1});

          val_merge_pred_pipe::write(1);
          u_store_val_pipe::write(v);
          v_store_val_pipe::write(u);

          out_res += 1;
        }
        else {
          // Always need to provide tag.
          u_store_idx_pipe::write({-1, i, 0});
          v_store_idx_pipe::write({-1, i, 1});
        }

        i = i + 1;
      }

      val_merge_pred_pipe::write(0);
      out_pointer[0] = out_res;

      PRINTF("Finished Calculation\n")
    });
  });



  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}