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

// The default PipelinedLSU will start a load/store immediately, which the memory disambiguation 
// logic relies upon.
// A BurstCoalescedLSU would instead of waiting for more requests to arrive for a coalesced access.
using PipelinedLSU = ext::intel::lsu<>;

#ifndef Q_SIZE
  #define Q_SIZE 8
#endif

constexpr uint STORE_Q_SIZE = Q_SIZE;
constexpr uint STORE_LATENCY = 78; // This should be gotten from static analysis.

struct store_entry {
  int idx; // This should be the full address in a real impl.
  int tag;
  int countdown;
  int val;
};

struct pair {
  int fst; 
  int snd;
};


double maximal_matching_kernel(queue &q, const std::vector<int> &edges, std::vector<int> &vertices,
                               int *out, const int num_edges) {
  std::cout << "Dynamic HLS\n";

  buffer edges_buf(edges);
  buffer vertices_buf(vertices);
  buffer out_buf(out, range{1});
  
  using u_pipe = pipe<class u_load_forked_pipe_class, int, 64>;
  using v_pipe = pipe<class v_load_forked_pipe_class, int, 64>;
  using u_load_pipe = pipe<class u_load_pipe_class, pair, 64>;
  using v_load_pipe = pipe<class v_load_pipe_class, pair, 64>;
  using vertex_u_pipe = pipe<class vertex_u_pipe_class, int, 64>;
  using vertex_v_pipe = pipe<class vertex_v_pipe_class, int, 64>;

  using u_store_idx_pipe = pipe<class u_store_idx_pipe_class, pair, 64>;
  using v_store_idx_pipe = pipe<class v_store_idx_pipe_class, pair, 64>;
  using u_store_val_pipe = pipe<class u_store_val_pipe_class, int, 64>;
  using v_store_val_pipe = pipe<class v_store_val_pipe_class, int, 64>;

  using end_lsq_signal_pipe = pipe <class end_lsq_signal_pipe_class, bool>;

  q.submit([&](handler &hnd) {
    accessor edges(edges_buf, hnd, read_only);

    hnd.single_task<class LoadEdges>([=]() [[intel::kernel_args_restrict]] {
      int i = 0;

      while (i < num_edges) {
        int j = i * 2;

        int u = edges[j];
        int v = edges[j + 1];

        u_load_pipe::write({u, i});
        v_load_pipe::write({v, i});
        
        PRINTF("i_%d: load u %d   v %d\n", i, u, v);

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

        u_pipe::write(u);
        v_pipe::write(v);
      }
    });
  });

  auto event = q.submit([&](handler &hnd) {
    accessor out_pointer(out_buf, hnd, write_only);

    hnd.single_task<class IfCondition>([=]() [[intel::kernel_args_restrict]] {
      int i = 0;
      int out = 0;
      while (i < num_edges) {
        auto vertex_u = vertex_u_pipe::read();
        auto vertex_v = vertex_v_pipe::read();

        PRINTF("i_%d: vertex[u] %d   vertex[v]%d\n", i, vertex_u, vertex_v);

        auto u = u_pipe::read();
        auto v = v_pipe::read();

        if ((vertex_u < 0) && (vertex_v < 0)) {
          u_store_idx_pipe::write({u, i});
          v_store_idx_pipe::write({v, i});

          u_store_val_pipe::write(v);
          v_store_val_pipe::write(u);

          out += 1;
        }

        i = i + 1;
      }

      end_lsq_signal_pipe::write(1);
      out_pointer[0] = out;
      PRINTF("Finished IfCondition\n")
    });
  });


  q.submit([&](handler &hnd) {
    accessor vertices(vertices_buf, hnd, read_write);

    hnd.single_task<class LSQ>([=]() [[intel::kernel_args_restrict]] {
      [[intel::fpga_register]] store_entry store_entries[STORE_Q_SIZE];
      // A FIFO of {idx_into_store_q, idx_into_x} pairs.
      [[intel::fpga_register]] pair store_idx_1_fifo[STORE_Q_SIZE];
      [[intel::fpga_register]] pair store_idx_2_fifo[STORE_Q_SIZE];

      #pragma unroll
      for (uint i=0; i<STORE_Q_SIZE; ++i) {
        store_entries[i] = {-1, -1, -1, -1};
      }

      int i_store_1_val = 0;
      int i_store_2_val = 0;
      int i_store_1_idx = 0;
      int i_store_2_idx = 0;
      int store_idx_1_fifo_head = 0;
      int store_idx_2_fifo_head = 0;

      int tag_store_1 = 0;
      int idx_store_1;
      int val_store_1;
      pair idx_tag_pair_store_1;

      int tag_store_2 = 0;
      int idx_store_2;
      int val_store_2;
      pair idx_tag_pair_store_2;

      int val_load_1;
      int idx_load_1, tag_load_1 = 0; 
      bool try_val_load_1_pipe_write = true;
      bool is_load_1_waiting = false;
      pair idx_tag_pair_load_1;

      int val_load_2;
      int  idx_load_2, tag_load_2 = 0; 
      bool try_val_load_2_pipe_write = true;
      bool is_load_2_waiting = false;
      pair idx_tag_pair_load_2;

      bool end_signal = false;

      [[intel::ivdep]] 
      while (!end_signal || i_store_1_idx > i_store_1_val || i_store_2_idx > i_store_2_val) {
        /* Start Load 1 Logic */
        auto max_tag_store = max(tag_store_1, tag_store_2);
        // PRINTF("max_tag_store: %d\n", max_tag_store);
        // PRINTF("tag_load_1: %d\n", tag_load_1);
        // PRINTF("tag_load_2: %d\n", tag_load_2);

        if (tag_load_1 <= max_tag_store && try_val_load_1_pipe_write) {
          bool idx_load_pipe_succ = false;

          if (!is_load_1_waiting) {
            idx_tag_pair_load_1 = u_load_pipe::read(idx_load_pipe_succ);
          }

          if (idx_load_pipe_succ || is_load_1_waiting) {
            idx_load_1 = idx_tag_pair_load_1.fst;
            tag_load_1 = idx_tag_pair_load_1.snd;

            bool _is_load_1_waiting = false;
            int max_tag = -1;
            #pragma unroll
            for (uint i=0; i<STORE_Q_SIZE; ++i) {
              auto st_entry = store_entries[i];
              if (st_entry.idx == idx_load_1 && st_entry.tag < tag_load_1 && st_entry.tag > max_tag) {
                _is_load_1_waiting = (st_entry.countdown == -1);
                val_load_1 = st_entry.val;
                max_tag = st_entry.tag;
              }
            }

            if (!_is_load_1_waiting && max_tag == -1) {
              val_load_1 = PipelinedLSU::load(vertices.get_pointer() + idx_load_1);
            }

            is_load_1_waiting = _is_load_1_waiting;
            try_val_load_1_pipe_write = _is_load_1_waiting;
          }
        }
        if (!try_val_load_1_pipe_write) {
          vertex_u_pipe::write(val_load_1, try_val_load_1_pipe_write);
        }
        /* End Load 1 Logic */

        /* Start Load 2 Logic */
        if (tag_load_2 <= max_tag_store && try_val_load_2_pipe_write) {
          bool idx_load_pipe_succ = false;

          if (!is_load_2_waiting) {
            idx_tag_pair_load_2 = v_load_pipe::read(idx_load_pipe_succ);
          }

          if (idx_load_pipe_succ || is_load_2_waiting) {
            idx_load_2 = idx_tag_pair_load_2.fst;
            tag_load_2 = idx_tag_pair_load_2.snd;

            bool _is_load_2_waiting = false;
            int max_tag = -1;
            #pragma unroll
            for (uint i=0; i<STORE_Q_SIZE; ++i) {
              auto st_entry = store_entries[i];
              if (st_entry.idx == idx_load_2 && st_entry.tag < tag_load_2 && st_entry.tag > max_tag ) {
                _is_load_2_waiting = (st_entry.countdown == -1);
                val_load_2 = st_entry.val;
                max_tag = st_entry.tag;
              }
            }

            if (!_is_load_2_waiting && max_tag == -1) {
              val_load_2 = PipelinedLSU::load(vertices.get_pointer() + idx_load_2);
            }

            is_load_2_waiting = _is_load_2_waiting;
            try_val_load_2_pipe_write = _is_load_2_waiting;
          }
        }
        if (!try_val_load_2_pipe_write) {
          vertex_v_pipe::write(val_load_2, try_val_load_2_pipe_write);
        }
        /* End Load 2 Logic */
      
        /* Start Store 1 Logic */
        #pragma unroll
        for (uint i=0; i<STORE_Q_SIZE; ++i) {
          const int count = store_entries[i].countdown;
          if (count == 1) store_entries[i].idx = -1;
          if (count > 1) store_entries[i].countdown--;
        }
        int next_entry_slot_1 = -1;
        #pragma unroll
        for (uint i=0; i<STORE_Q_SIZE; ++i) {
          if (store_entries[i].idx == -1)  {
            next_entry_slot_1 = i;
          }
        }

        bool idx_store_1_pipe_succ = false;
        if (next_entry_slot_1 != -1 && store_idx_1_fifo_head < STORE_Q_SIZE) {
          idx_tag_pair_store_1 = u_store_idx_pipe::read(idx_store_1_pipe_succ);
        }

        if (idx_store_1_pipe_succ) {
          idx_store_1 = idx_tag_pair_store_1.fst;
          tag_store_1 = idx_tag_pair_store_1.snd;
          store_entries[next_entry_slot_1] = {idx_store_1, tag_store_1, -1};
          store_idx_1_fifo[store_idx_1_fifo_head] = {next_entry_slot_1, idx_store_1};

          i_store_1_idx++;
          store_idx_1_fifo_head++;
        }
        
        bool val_store_1_pipe_succ = false;
        if (i_store_1_idx > i_store_1_val) {
          val_store_1 = u_store_val_pipe::read(val_store_1_pipe_succ);
        }

        if (val_store_1_pipe_succ) {
          auto entry_slot_and_idx_pair = store_idx_1_fifo[0];           
          PipelinedLSU::store(vertices.get_pointer() + entry_slot_and_idx_pair.snd, val_store_1);
          store_entries[entry_slot_and_idx_pair.fst].val = val_store_1;
          store_entries[entry_slot_and_idx_pair.fst].countdown = STORE_LATENCY;

          #pragma unroll
          for (uint i=1; i<STORE_Q_SIZE; ++i) {
            store_idx_1_fifo[i-1] = store_idx_1_fifo[i];
          }
          store_idx_1_fifo_head--;

          i_store_1_val++;
        }
        /* End Store 1 Logic */

        /* Start Store 2 Logic */
        int next_entry_slot_2 = -1;
        #pragma unroll
        for (uint i=0; i<STORE_Q_SIZE; ++i) {
          if (store_entries[i].idx == -1)  {
            next_entry_slot_2 = i;
          }
        }

        bool idx_store_2_pipe_succ = false;
        if (next_entry_slot_2 != -1 && store_idx_2_fifo_head < STORE_Q_SIZE) {
          idx_tag_pair_store_2 = v_store_idx_pipe::read(idx_store_2_pipe_succ);
        }

        if (idx_store_2_pipe_succ) {
          idx_store_2 = idx_tag_pair_store_2.fst;
          tag_store_2 = idx_tag_pair_store_2.snd;
          store_entries[next_entry_slot_2] = {idx_store_2, tag_store_2, -1};
          store_idx_2_fifo[store_idx_2_fifo_head] = {next_entry_slot_2, idx_store_2};

          i_store_2_idx++;
          store_idx_2_fifo_head++;
        }
        
        bool val_store_2_pipe_succ = false;
        if (i_store_2_idx > i_store_2_val) {
          val_store_2 = v_store_val_pipe::read(val_store_2_pipe_succ);
        }

        if (val_store_2_pipe_succ) {
          auto entry_slot_and_idx_pair = store_idx_2_fifo[0];           
          PipelinedLSU::store(vertices.get_pointer() + entry_slot_and_idx_pair.snd, val_store_2);
          store_entries[entry_slot_and_idx_pair.fst].val = val_store_2;
          store_entries[entry_slot_and_idx_pair.fst].countdown = STORE_LATENCY;

          #pragma unroll
          for (uint i=1; i<STORE_Q_SIZE; ++i) {
            store_idx_2_fifo[i-1] = store_idx_2_fifo[i];
          }
          store_idx_2_fifo_head--;

          i_store_2_val++;
        }
        /* End Store 2 Logic */
      
        if (!end_signal) end_lsq_signal_pipe::read(end_signal);
      }

    });
  });



  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}
