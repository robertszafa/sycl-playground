#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/detail/common.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include <CL/sycl.hpp>
#include <cstdio>
#include <iostream>
#include <limits>
#include <vector>

#include <sycl/ext/intel/fpga_extensions.hpp>

#include "store_queue.hpp"

using namespace sycl;

// The default PipelinedLSU will start a load/store immediately, which the memory disambiguation 
// logic relies upon.
// A BurstCoalescedLSU would instead of waiting for more requests to arrive for a coalesced access.
using PipelinedLSU = ext::intel::lsu<>;

#ifndef Q_SIZE
  #define Q_SIZE 8
#endif

constexpr uint STORE_Q_SIZE = Q_SIZE;

/// <val, tag>
struct pair {
  int first; 
  int second; 
};


// Maybe have a seq (1 for exec, 0 for not) pipe for each ld, st and count the tokens such that 
// loads don't overtake store_idxs
double spmv_kernel(queue &q, 
                   std::vector<float> &matrix,       
                   const std::vector<int> &row,
                   const std::vector<int> &col,
                   std::vector<float> &a,             
                   const int M) {
#if dynamic_no_forward
  constexpr bool IS_FORWARDING_Q = false;
  std::cout << "Dynamic (no forward) HLS\n";
#else
  constexpr bool IS_FORWARDING_Q = true;
  std::cout << "Dynamic HLS\n";
#endif

  buffer matrix_buf(matrix);
  buffer row_buf(row);
  buffer col_buf(col);
  buffer a_buf(a);

  constexpr int kNumLdPipes = 2;
  using idx_ld_pipes = PipeArray<class idx_ld_pipes_class, pair, 64, kNumLdPipes>;
  using val_ld_pipes = PipeArray<class val_ld_pipes_class, float, 64, kNumLdPipes>;
  using idx_st_pipe = pipe<class idx_store_pipe_class, pair, 64>;
  using val_st_pipe = pipe<class val_store_pipe_class, float, 64>;

  using ld_a_pipe = pipe<class ld_a_class, float, 64>;

  using end_storeq_signal_pipe = pipe<class end_lsq_signal_class, bool>;
  

  q.submit([&](handler &hnd) {
    accessor a(a_buf, hnd, read_only);

    hnd.single_task<class LoadA>([=]() [[intel::kernel_args_restrict]] {
      for (int k = 1; k < M; k++) {
        for (int p = 0; p < M; p++) {
          ld_a_pipe::write(a[p]);
        }
      }
    });
  });

  q.submit([&](handler &hnd) {
    accessor col(col_buf, hnd, read_only);

    hnd.single_task<class LoadIdx1>([=]() [[intel::kernel_args_restrict]] {
      int tag = 0;
      for (int k = 1; k < M; k++) {
        for (int p = 0; p < M; p++) {
          auto load_idx_1 = (k - 1) * M + col[p];
          idx_ld_pipes::PipeAt<0>::write({load_idx_1, tag});

          tag++;
        }
      }
    });
  });

  q.submit([&](handler &hnd) {
    accessor row(row_buf, hnd, read_only);

    hnd.single_task<class LoadIdxs2>([=]() [[intel::kernel_args_restrict]] {
      int tag = 0;
      for (int k = 1; k < M; k++) {
        for (int p = 0; p < M; p++) {
          auto load_idx_2 = k * M + row[p];
          idx_ld_pipes::PipeAt<1>::write({load_idx_2, tag});

          tag++;
        }
      }
    });
  });

  q.submit([&](handler &hnd) {
    accessor row(row_buf, hnd, read_only);

    hnd.single_task<class StoreIdxs>([=]() [[intel::kernel_args_restrict]] {
      int tag = 0;
      for (int k = 1; k < M; k++) {
        for (int p = 0; p < M; p++) {
          auto store_idx = k * M + row[p];

          idx_st_pipe::write({store_idx, tag});
          tag++;
        }
      }
    });
  });


  StoreQueue<idx_ld_pipes, val_ld_pipes, kNumLdPipes, pair, idx_st_pipe, val_st_pipe, 
             end_storeq_signal_pipe, IS_FORWARDING_Q, Q_SIZE, 12> (q, matrix_buf);


  auto event = q.submit([&](handler &hnd) {
    hnd.single_task<class spmv_dynamic>([=]() [[intel::kernel_args_restrict]] {
      for (int k = 1; k < M; k++) {
        for (int p = 0; p < M; p++) {
          auto load_x_1 = val_ld_pipes::PipeAt<0>::read(); // matrix[(k - 1) * M + col[p]];
          auto load_x_2 = val_ld_pipes::PipeAt<1>::read(); // matrix[k*M + row[p]];
          auto load_a = ld_a_pipe::read();

          auto store_x = load_x_2 + load_a * load_x_1;
          
          val_st_pipe::write(store_x);
        }
      }

      end_storeq_signal_pipe::write(1);
    });
  });

  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}
