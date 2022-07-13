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

// From oneAPI-samples/DirectProgramming/DPC++FPGA/include
#include "pipe_utils.hpp"
#include "unrolled_loop.hpp"

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
constexpr uint STORE_LATENCY = 8; // This should be gotten from static analysis.

struct store_entry {
  int idx; // This should be the full address in a real impl.
  int tag;
};
struct pair {
  int fst; 
  int snd; 
};

// Forward declare store port kernels and I/O pipes.
template <int T> class StorePortKernel;
using PortInPipeArray = fpga_tools::PipeArray< // Defined in "pipe_utils.hpp".
    class PortIn,                              // An identifier for the pipe.
    bool,                                       // The type of data in the pipe.
    0,                                         // The capacity of each pipe.
    STORE_Q_SIZE                               // array dimension.
    >;
using PortOutPipeArray = fpga_tools::PipeArray< 
    class PortOut,                              
    bool,                                       
    0,
    STORE_Q_SIZE                               
    >;


// Maybe have a seq (1 for exec, 0 for not) pipe for each ld, st and count the tokens such that 
// loads don't overtake store_idxs
double spmv_kernel(queue &q, 
                   std::vector<float> &matrix,       
                   const std::vector<int> &row,
                   const std::vector<int> &col,
                   std::vector<float> &a,             
                   const int M) {

  std::cout << "Dynamic (no forwarding) HLS\n";

  buffer matrix_buf(matrix);
  buffer row_buf(row);
  buffer col_buf(col);
  buffer a_buf(a);

  using idx_load_1_pipe = pipe<class idx_load_1_pipe_class, pair, 64>;
  using idx_load_2_pipe = pipe<class idx_load_2_pipe_class, pair, 64>;
  using val_load_1_pipe = pipe<class val_load_1_pipe_class, float, 64>;
  using val_load_2_pipe = pipe<class val_load_2_pipe_class, float, 64>;
  using idx_store_pipe = pipe<class idx_store_pipe_class, pair, 64>;
  using val_store_pipe = pipe<class val_store_pipe_class, float, 64>;

  using ld_a_pipe = pipe<class ld_a_class, float, 64>;

  using end_lsq_signal_pipe = pipe<class end_lsq_signal_class, bool>;
  

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
          idx_load_1_pipe::write({load_idx_1, tag});

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
          idx_load_2_pipe::write({load_idx_2, tag});

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

          idx_store_pipe::write({store_idx, tag});
          tag++;
        }
      }
    });
  });

  fpga_tools::UnrolledLoop<0, STORE_Q_SIZE>([&](auto i) {
    q.submit([&](handler &hnd) {
      hnd.single_task<StorePortKernel<i>>([=]() [[intel::kernel_args_restrict]] {
        while(1) {
          bool terminate = PortInPipeArray::PipeAt<i>::read();
          if (terminate) break;

          // TODO: Use latency anchor extension instead of countdown.
          // uint count = STORE_LATENCY;
          // while(count > 0) count--;

          PortOutPipeArray::PipeAt<i>::write(0);
        }
      });
    });
  });


  q.submit([&](handler &hnd) {
    accessor matrix(matrix_buf, hnd, read_write);

    hnd.single_task<class LSQ>([=]() [[intel::kernel_args_restrict]] {
      [[intel::fpga_register]] store_entry store_entries[STORE_Q_SIZE];
      // A FIFO of {idx_into_store_q, idx_into_x} pairs.
      [[intel::fpga_register]] pair store_idx_fifo[STORE_Q_SIZE];

      #pragma unroll
      for (int i=0; i<STORE_Q_SIZE; ++i) {
        store_entries[i] = {-1, -1};
      }

      int i_store_val = 0;
      int i_store_idx = 0;
      int store_idx_fifo_head = 0;

      int tag_store = 0;
      int idx_store;
      float val_store;
      pair idx_tag_pair_store;

      float val_load_1;
      int idx_load_1, tag_load_1 = 0; 
      bool val_load_1_pipe_write_succ = true;
      bool is_load_1_waiting = false;
      pair idx_tag_pair_load_1;

      float val_load_2;
      int  idx_load_2, tag_load_2 = 0; 
      bool val_load_2_pipe_write_succ = true;
      bool is_load_2_waiting = false;
      pair idx_tag_pair_load_2;

      bool end_signal = false;

      [[intel::ivdep]] 
      while (!end_signal || i_store_idx > i_store_val) {
        /* Start Load 1 Logic */
        if (tag_load_1 <= tag_store && val_load_1_pipe_write_succ) {
          bool idx_load_pipe_succ = false;

          if (!is_load_1_waiting) {
            idx_tag_pair_load_1 = idx_load_1_pipe::read(idx_load_pipe_succ);
          }

          if (idx_load_pipe_succ || is_load_1_waiting) {
            idx_load_1 = idx_tag_pair_load_1.fst;
            tag_load_1 = idx_tag_pair_load_1.snd;

            bool _is_load_1_waiting = false;
            #pragma unroll
            for (uint i=0; i<STORE_Q_SIZE; ++i) {
              auto st_entry = store_entries[i];
              if (st_entry.idx == idx_load_1 && st_entry.tag < tag_load_1) {
                _is_load_1_waiting |= true;
              }
            }

            is_load_1_waiting = _is_load_1_waiting;

            if (!is_load_1_waiting) {
              val_load_1 = PipelinedLSU::load(matrix.get_pointer() + idx_load_1);
              val_load_1_pipe_write_succ = false;
            }
          }
        }
        if (!val_load_1_pipe_write_succ) {
          val_load_1_pipe::write(val_load_1, val_load_1_pipe_write_succ);
        }
        /* End Load 1 Logic */

        /* Start Load 2 Logic */
        if (tag_load_2 <= tag_store && val_load_2_pipe_write_succ) {
          bool idx_load_pipe_succ = false;

          if (!is_load_2_waiting) {
            idx_tag_pair_load_2 = idx_load_2_pipe::read(idx_load_pipe_succ);
          }

          if (idx_load_pipe_succ || is_load_2_waiting) {
            idx_load_2 = idx_tag_pair_load_2.fst;
            tag_load_2 = idx_tag_pair_load_2.snd;

            bool _is_load_2_waiting = false;
            #pragma unroll
            for (uint i=0; i<STORE_Q_SIZE; ++i) {
              auto st_entry = store_entries[i];
              if (st_entry.idx == idx_load_2 && st_entry.tag < tag_load_2) {
                _is_load_2_waiting |= true;
              }
            }

            is_load_2_waiting = _is_load_2_waiting;

            if (!is_load_2_waiting) {
              val_load_2 = PipelinedLSU::load(matrix.get_pointer() + idx_load_2);
              val_load_2_pipe_write_succ = false;
            }
          }
        }
        if (!val_load_2_pipe_write_succ) {
          val_load_2_pipe::write(val_load_2, val_load_2_pipe_write_succ);
        }
        /* End Load 2 Logic */
      
        /* Start Store Logic */
        // Check if any store on the store ports has completed
        fpga_tools::UnrolledLoop<0, STORE_Q_SIZE>([&](auto i) {
          bool succ = false;
          PortOutPipeArray::PipeAt<i>::read(succ);
          if (succ) {
            store_entries[i].idx = -1;
          }
        });

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
          store_entries[next_entry_slot] = {idx_store, tag_store};
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
          PipelinedLSU::store(matrix.get_pointer() + entry_slot_and_idx_pair.snd, val_store);

          // Start countdown on the correct port.
          fpga_tools::UnrolledLoop<0, STORE_Q_SIZE>([&](auto i) {
            if (i == entry_slot_and_idx_pair.fst) {
              PortInPipeArray::PipeAt<i>::write(0);
            }
          });

          #pragma unroll
          for (uint i=1; i<STORE_Q_SIZE; ++i) {
            store_idx_fifo[i-1] = store_idx_fifo[i];
          }
          store_idx_fifo_head--;

          i_store_val++;
        }
        /* End Store Logic */
      
        if (!end_signal)
          end_lsq_signal_pipe::read(end_signal);

      }

      // Stop all port countdown kernels.
      fpga_tools::UnrolledLoop<0, STORE_Q_SIZE>([&](auto i) {
        PortInPipeArray::PipeAt<i>::write(1);
      });

    });
  });

  auto event = q.submit([&](handler &hnd) {
    hnd.single_task<class spmv_dynamic>([=]() [[intel::kernel_args_restrict]] {
      for (int k = 1; k < M; k++) {
        for (int p = 0; p < M; p++) {
          auto load_x_1 = val_load_1_pipe::read(); // matrix[(k - 1) * M + col[p]];
          auto load_x_2 = val_load_2_pipe::read(); // matrix[k*M + row[p]];
          auto load_a = ld_a_pipe::read();

          auto store_x = load_x_2 + load_a * load_x_1;
          
          val_store_pipe::write(store_x);
        }
      }

      end_lsq_signal_pipe::write(1);
    });
  });

  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}
