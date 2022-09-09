#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#include "memory_utils.hpp"
#include "store_queue.hpp"

#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

#ifndef Q_SIZE
  #define Q_SIZE 8
#endif

double maximal_matching_kernel(queue &q, const std::vector<int> &h_edges, std::vector<int> &h_vertices,
                               int *h_out, const int num_edges) {
  #if dynamic_no_forward_sched
  constexpr bool IS_FORWARDING_Q = false;
  std::cout << "Dynamic (no forward) HLS\n";
#else
  constexpr bool IS_FORWARDING_Q = true;
  std::cout << "Dynamic HLS\n";
#endif

  const int* edges = toDevice(h_edges, q);
  int* vertices = toDevice(h_vertices, q);
  int* out = toDevice(h_out, 1, q);

  constexpr int kNumStoreOps = 2;
  constexpr int kNumLdPipes = 2;
  
  using idx_ld_pipes = PipeArray<class idx_ld_pipe_class, pair, 64, kNumLdPipes>;
  using val_ld_pipes = PipeArray<class val_ld_pipe_class, int, 64, kNumLdPipes>;

  using u_load_pipe = idx_ld_pipes::PipeAt<0>;
  using v_load_pipe = idx_ld_pipes::PipeAt<1>;
  using vertex_u_pipe = val_ld_pipes::PipeAt<0>;
  using vertex_v_pipe = val_ld_pipes::PipeAt<1>;

  using u_pipe = pipe<class u_load_forked_pipe_class, int, 64>;
  using v_pipe = pipe<class v_load_forked_pipe_class, int, 64>;

  using u_store_idx_pipe = pipe<class u_store_idx_pipe_class, pair, 64>;
  using v_store_idx_pipe = pipe<class v_store_idx_pipe_class, pair, 64>;
  using u_store_val_pipe = pipe<class u_store_val_pipe_class, int, 64>;
  using v_store_val_pipe = pipe<class v_store_val_pipe_class, int, 64>;

  using idx_st_pipe = pipe<class store_idx_pipe_class, pair, 64>;
  using val_st_pipe = pipe<class store_val_pipe_class, int, 64>;

  using end_storeq_signal_pipe = pipe<class end_lsq_signal_pipe_class, bool>;
  
  using val_merge_pred_pipe = pipe<class val_merge_pred_class, bool, 64>;

  q.submit([&](handler &hnd) {
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
    hnd.single_task<class LoadEdges2>([=]() [[intel::kernel_args_restrict]] {
      int i = 0;

      while (i < num_edges) {
        int j = i * 2;

        int u = edges[j];
        int v = edges[j + 1];

        u_load_pipe::write({u, i*kNumStoreOps});
        v_load_pipe::write({v, i*kNumStoreOps});

        i = i + 1;
      }
    });
  });

  
  StoreQueue<idx_ld_pipes, val_ld_pipes, kNumLdPipes, idx_st_pipe, val_st_pipe, 
             end_storeq_signal_pipe, IS_FORWARDING_Q, Q_SIZE, 12> (q, device_ptr<int>(vertices));


  q.submit([&](handler &hnd) {
    hnd.single_task<class StoreIdxMerge>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < num_edges*2; ++i) {
        if (i % 2 == 0)
          idx_st_pipe::write(u_store_idx_pipe::read());
        else
          idx_st_pipe::write(v_store_idx_pipe::read());
      }
    });
  });
  
  q.submit([&](handler &hnd) {
    hnd.single_task<class StoreValMerge>([=]() [[intel::kernel_args_restrict]] {
      while (val_merge_pred_pipe::read()) {
        for (int i=0; i<2; i++) {
          if (i % 2 == 0)
            val_st_pipe::write(u_store_val_pipe::read());
          else
            val_st_pipe::write(v_store_val_pipe::read());
        }
      }
    });
  });

  auto event = q.submit([&](handler &hnd) {
    hnd.single_task<class Calculation>([=]() [[intel::kernel_args_restrict]] {
      int i = 0;
      int out_res = 0;
      while (i < num_edges) {

        auto u = u_pipe::read();
        auto v = v_pipe::read();
        
        auto vertex_u = vertex_u_pipe::read();
        auto vertex_v = vertex_v_pipe::read();

        if ((vertex_u < 0) && (vertex_v < 0)) {
          u_store_idx_pipe::write({u, i*kNumStoreOps + 1});
          v_store_idx_pipe::write({v, i*kNumStoreOps + 2});

          val_merge_pred_pipe::write(1);
          u_store_val_pipe::write(v);
          v_store_val_pipe::write(u);

          out_res += 1;
        }
        else {
          // Always need to provide tag.
          u_store_idx_pipe::write({-1, i*kNumStoreOps + 1});
          v_store_idx_pipe::write({-1, i*kNumStoreOps + 2});
        }

        i = i + 1;
      }

      val_merge_pred_pipe::write(0);
      end_storeq_signal_pipe::write(1);

      *out = out_res;
    });
  });

  event.wait();
  q.memcpy(h_vertices.data(), vertices, sizeof(h_vertices[0]) * h_vertices.size()).wait();
  q.memcpy(h_out, out, sizeof(h_out[0])).wait();

  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}