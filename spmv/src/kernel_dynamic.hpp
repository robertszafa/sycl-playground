#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/detail/common.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include <CL/sycl.hpp>
#include <cstdio>
#include <iostream>
#include <vector>

#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;
#define sycl_print sycl::ext::oneapi::experimental::printf

using PipelinedLSU = ext::intel::lsu<>;

constexpr uint STORE_Q_SIZE = Q_SIZE;
constexpr uint STORE_LATENCY = 16; // This should be gotten from static analysis.


struct store_entry {
  int idx; // This should be the full address in a real impl.
  float val;
  bool executed;
  int countdown;
  int tag;
};

struct load_entry {
  int idx; // This should be the full address in a real impl.
  int tag;
  bool consumer; 
};

constexpr store_entry INVALID_ENTRY = {-1, 0, false, -1, -1};

struct pair {
  int fst; 
  int snd; 
};
struct triple {
  int fst; 
  int snd; 
  int thrd; 
};

struct candidate {
  int tag; 
  bool forward;
};


// Maybe have a seq (1 for exec, 0 for not) pipe for each ld, st and count the tokens such that 
// loads don't overtake store_idxs
double spmv_kernel(queue &q, 
                   std::vector<float> &matrix,       
                   const std::vector<int> &row,
                   const std::vector<int> &col,
                   std::vector<float> &a,             
                   const int M) {

  std::cout << "Dynamic HLS\n";

  // matrix will store A^i * matrix fir every i. The initial matrix will be at i=0.
  buffer matrix_buf(matrix);
  buffer row_buf(row);
  buffer col_buf(col);
  buffer a_buf(a);

  using idx_load_1_pipe = pipe<class idx_load_1_pipe_class, pair, 64>;
  using idx_load_2_pipe = pipe<class idx_load_2_pipe_class, pair, 64>;
  using idx_load_pipe = pipe<class idx_load_pipe_class, triple, 64>;
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
    accessor row(row_buf, hnd, read_only);
    accessor col(col_buf, hnd, read_only);

    hnd.single_task<class LoadIdxs>([=]() [[intel::kernel_args_restrict]] {
      int tag = 0;
      for (int k = 1; k < M; k++) {
        for (int p = 0; p < M; p++) {
          auto load_idx_1 = (k - 1) * M + col[p];
          auto load_idx_2 = k * M + row[p];

          idx_load_pipe::write({load_idx_1, tag, 0});
          idx_load_pipe::write({load_idx_2, tag, 1});

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

  q.submit([&](handler &hnd) {
    accessor matrix(matrix_buf, hnd, read_write);

    hnd.single_task<class LSQ>([=]() [[intel::kernel_args_restrict]] {
      [[intel::fpga_register]] store_entry store_entries[STORE_Q_SIZE];
      // A FIFO of {idx_into_store_q, tag} pairs to pick the youngest store from the store entries.
      [[intel::fpga_register]] candidate forward_candidates[STORE_Q_SIZE];
      // A FIFO of {idx_into_store_q, idx_into_x} pairs.
      [[intel::fpga_register]] pair store_idx_fifo[STORE_Q_SIZE];

      // All entries are invalid at the start.
      #pragma unroll
      for (uint i=0; i<STORE_Q_SIZE; ++i) {
        store_entries[i] = INVALID_ENTRY;
      }

      int i_store_val = 0;
      int i_store_idx = 0;
      int store_idx_fifo_head = 0;

      int idx_load_1, idx_load_2, idx_store, idx_load; 
      float val_store, val_load;
      bool consumer_id_load;
      int tag_load = 0; 
      int tag_store = 0;

      bool val_load_pipe_write_succ = true;
      bool is_load_waiting_for_val = false;
      bool is_load_waiting_for_store_idx = false;

      bool end_signal = false;

      [[intel::ivdep]] 
      while (!end_signal || i_store_idx < i_store_val) {
        /* Start Load Logic */
        if (val_load_pipe_write_succ) {
          bool idx_load_pipe_succ = false;

          is_load_waiting_for_store_idx = (tag_load > tag_store);

          if (!is_load_waiting_for_val && !is_load_waiting_for_store_idx) {
            auto idx_tag_consumer = idx_load_pipe::read(idx_load_pipe_succ);
            idx_load = idx_tag_consumer.fst;
            tag_load = idx_tag_consumer.snd;
            consumer_id_load = idx_tag_consumer.thrd;
          }

          if (!is_load_waiting_for_store_idx && (idx_load_pipe_succ || is_load_waiting_for_val)) {
            is_load_waiting_for_val = false;

            #pragma unroll
            for (uint i=0; i<STORE_Q_SIZE; ++i) {
              if (store_entries[i].idx == idx_load && store_entries[i].countdown > 0 &&
                  store_entries[i].tag < tag_load) {
                // The forward_candidates.forward field corresponds to store_entries.executed.
                forward_candidates[i] = {store_entries[i].tag, store_entries[i].executed};
              }
              else {
                forward_candidates[i] = {-1, false};
              }
            }

            int max_tag = -1;
            #pragma unroll
            for (uint i=0; i<STORE_Q_SIZE; ++i) {
              max_tag = max(max_tag, forward_candidates[i].tag);
            }

            if (max_tag >= 0) {
              is_load_waiting_for_val = true;
              #pragma unroll
              for (uint i=0; i<STORE_Q_SIZE; ++i) {
                if (max_tag == forward_candidates[i].tag && forward_candidates[i].forward) {
                  val_load = store_entries[i].val;
                  is_load_waiting_for_val = false;
                }
              }
            }
            else {
              val_load = PipelinedLSU::load(matrix.get_pointer() + idx_load);
            }

            if (!is_load_waiting_for_val) {
              val_load_pipe_write_succ = false;
            }
          }
        }

        if (!val_load_pipe_write_succ) {
          if (consumer_id_load == 0)
            val_load_1_pipe::write(val_load, val_load_pipe_write_succ);
          else
            val_load_2_pipe::write(val_load, val_load_pipe_write_succ);
        }
        /* End Load Logic */
      

        /* Start Store Logic */
        int next_entry_slot = -1;
        #pragma unroll
        for (uint i=0; i<STORE_Q_SIZE; ++i) {
          if (store_entries[i].executed && store_entries[i].countdown > 0) {
            store_entries[i].countdown--;
          }

          if (store_entries[i].countdown <= 0) {
            next_entry_slot = i;
          }
        }

        bool idx_store_pipe_succ = false;
        if (next_entry_slot != -1 && store_idx_fifo_head < STORE_Q_SIZE) {
          auto idx_tag_pair = idx_store_pipe::read(idx_store_pipe_succ);
          idx_store = idx_tag_pair.fst;
          tag_store = idx_tag_pair.snd;
        }

        if (idx_store_pipe_succ) {
          store_entries[next_entry_slot] = {(int) idx_store, 1000, false, STORE_LATENCY, tag_store};
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
          store_entries[entry_slot_and_idx_pair.fst].val = val_store;
          store_entries[entry_slot_and_idx_pair.fst].executed = true;

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

      // sycl_print("Finished LSQ\n");
    });
  });

  auto event = q.submit([&](handler &hnd) {
    hnd.single_task<class spmv_dynamic>([=]() [[intel::kernel_args_restrict]] {
      for (int k = 1; k < M; k++) {
        for (int p = 0; p < M; p++) {
          auto load_a = ld_a_pipe::read();
          auto load_x_1 = val_load_1_pipe::read(); // matrix[(k - 1) * M + col[p]];
          auto load_x_2 = val_load_2_pipe::read(); // matrix[k*M + row[p]];

          auto store_x = load_x_2 + load_a * load_x_1;
          
          val_store_pipe::write(store_x);
        }
      }

      end_lsq_signal_pipe::write(true);
    });
  });

  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}
