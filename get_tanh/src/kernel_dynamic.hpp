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

constexpr uint STORE_Q_SIZE = Q_SIZE; // Min q_size=2 because the queue is a circ. buffer. 
constexpr uint STORE_LATENCY = 16;    // This should be gotten from static analysis (min 7 cycles).

struct store_entry {
  int idx; // This should be the full address in a real impl.
  int tag;
  bool waiting_for_val;
  int countdown;
  int val;
};

struct pair {
  int first; 
  int second; 
};



double get_tanh_kernel(queue &q, std::vector<int> &A, const std::vector<int> addr_in, 
                       const std::vector<int> addr_out) {
  std::cout << "Dynamic HLS\n";

  const uint array_size = A.size();

  buffer A_buf(A);
  buffer addr_in_buf(addr_in);
  buffer addr_out_buf(addr_out);

  using beta_in_pipe = pipe<class beta_in_pipe_class, int, 64>;
  using result_out_pipe = pipe<class result_out_pipe_class, int, 64>;
  
  using predicate_calc_pipe = pipe<class predicate_calc_pipe_class, bool, 64>;
  using end_storeq_signal_pipe = pipe<class end_storeq_signal_pipe_class, bool>;

  using idx_ld_pipe = pipe<class idx_ld_pipe_class, pair, 64>;
  using idx_st_pipe = pipe<class idx_st_pipe_class, pair, 64>;
  using val_ld_pipe = pipe<class val_ld_pipe_class, int, 64>;
  using val_st_pipe = pipe<class val_st_pipe_class, int, 64>;

  auto event = q.submit([&](handler &hnd) {
    accessor addr_in(addr_in_buf, hnd, read_only);
    accessor addr_out(addr_out_buf, hnd, read_only);

    hnd.single_task<class MainKernel>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; i++) {
        // Input angle
        auto beta = val_ld_pipe::read(); // beta = A[addr_in[i]];
        // Result of tanh, sinh and cosh
        int result = 4096; // Saturation effect

        if (beta < 20480) {
          predicate_calc_pipe::write(1);
          beta_in_pipe::write(beta);
          result = result_out_pipe::read();
        }

        val_st_pipe::write(result); // A[addr_out[i]] = result;
      }

      predicate_calc_pipe::write(0);
      end_storeq_signal_pipe::write(0);
    });
  });

  q.submit([&](handler &hnd) {
    accessor addr_in(addr_in_buf, hnd, read_only);

    hnd.single_task<class LoadIdxLd>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; i++) {
        int ld_i = addr_in[i];
        idx_ld_pipe::write({ld_i, i});
        // PRINTF("Load idx pipe %d\n", ld_i);
      }
    });
  });

  q.submit([&](handler &hnd) {
    accessor addr_out(addr_out_buf, hnd, read_only);

    hnd.single_task<class LoadIdxSt>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; i++) {
        int st_i = addr_out[i];
        idx_st_pipe::write({st_i, i});
        // PRINTF("Store idx pipe %d\n", st_i);
      }
    });
  });

  q.submit([&](handler &hnd) {

    hnd.single_task<class CalcKernel>([=]() [[intel::kernel_args_restrict]] {
      int atanh[12] = {0x08C9, 0x0416, 0x0202, 0x0100, 0x0080, 0x0064,
                       0x0032, 0x0010, 0x0008, 0x0004, 0x0002, 0x0001};
      int cosh[5] = {0x1000, 0x18B0, 0x3C31, 0xA115, 0x1B4EE};
      int sinh[5] = {0x0, 0x12CD, 0x3A07, 0xA049, 0x1B4A3};

      // int x = 0x1351;
      // int y = 0;
      // int beta;

      #pragma ivdep
      while(predicate_calc_pipe::read()) {
        int x = 0x1351;
        int y = 0;
        int x_new;
        int index_trigo;
        int result_cosh, result_sinh;
        int outputcosh, outputsinh;

        int beta = beta_in_pipe::read();

        // Implement approximate range of the hyperbolic CORDIC block
        if (beta >= 8192) {
          index_trigo = 4;
        } else if (beta >= 12288) {
          index_trigo = 3;
        } else if (beta >= 8192) {
          index_trigo = 2;
        } else if (beta >= 4096) {
          index_trigo = 1;
        } else {
          index_trigo = 0;
        }
        beta = beta - index_trigo * 4096;          

        // Call to the hyperbolic CORDIC block
        #pragma unroll
        for (int k = 1; k <= 12; k++) {
          // force the 3k+1 th iteration to be repeated
          if (((k % 3) == 1) && (k != 1)) {
            #pragma unroll
            for (int j = 1; j <= 2; j++) {
              // beta<0 anti-clockwise rotation
              if (beta < 0) {
                x_new = x - (y >> k);
                y -= x >> k;
                beta += atanh[k - 1];
              }
              // beta>0 clockwise rotation
              else {
                x_new = x + (y >> k);
                y += (x >> k);
                beta -= atanh[k - 1];
              }
              x = x_new;
            }
          } else {
            if (beta < 0) {
              x_new = x - (y >> k);
              y -= x >> k;
              beta += atanh[k - 1];
            }
            // beta>0 clockwise rotation
            else {
              x_new = x + (y >> k);
              y += (x >> k);
              beta -= atanh[k - 1];
            }
            x = x_new;
          }
        }
        outputcosh = x;
        outputsinh = y;

        // Trigonometric rules application
        result_cosh = (sinh[index_trigo] * outputcosh + cosh[index_trigo] * outputsinh);
        result_sinh = (cosh[index_trigo] * outputcosh + sinh[index_trigo] * outputsinh) >> 12;
        // Central symmetry correction
        int result = result_cosh / result_sinh;

        result_out_pipe::write(result);
      }

    });
  });


  q.submit([&](handler &hnd) {
    accessor A(A_buf, hnd, read_write);

    hnd.single_task<class StoreQueue>([=]() [[intel::kernel_args_restrict]] {
      [[intel::fpga_register]] store_entry store_entries[STORE_Q_SIZE];

      #pragma unroll
      for (uint i=0; i<STORE_Q_SIZE; ++i) {
        store_entries[i] = {-1, -2};
      }

      int i_store_val = 0;
      int i_store_idx = 0;
      // Points to store_q entry where the next store value should be read in.
      int stq_tail = 0;
      // Points to store_q entry where the next store index should be read in.
      int stq_head = 0;
      uint num_enqued = 0;
      // Two tags because there could be two stores in the same iteration.
      // In general (nested loops, multiple stores in same scope, etc.) the tag is an n-tuple.
      int tag_store = -1;
      int idx_store;
      int val_store;
      pair idx_tag_pair_store;

      int val_load;
      int idx_load; 
      int tag_load = -1; 
      bool consumer_load_succ = true;
      bool is_load_waiting = false;
      pair idx_tag_pair_load;

      bool end_signal = false;

      [[intel::ivdep]] 
      while (!end_signal) {
        /* Start Load 1 Logic */
        if ((tag_load <= tag_store && consumer_load_succ) || is_load_waiting) {
          // Check for new ld requests.
          bool idx_load_pipe_succ = false;
          if (!is_load_waiting) {
            idx_tag_pair_load = idx_ld_pipe::read(idx_load_pipe_succ);
          }

          // If new ld request, or if we are still waiting for a previous one, then check store_q.
          if (idx_load_pipe_succ || is_load_waiting) {
            idx_load = idx_tag_pair_load.first;
            tag_load = idx_tag_pair_load.second;

            is_load_waiting = false;
            int max_tag = -1; 
            #pragma unroll
            for (uint i=0; i<STORE_Q_SIZE; ++i) {
              auto st_entry = store_entries[i];
              // If found, make sure it's the youngest store occuring before this ld. 
              if (st_entry.idx == idx_load && st_entry.tag < tag_load && st_entry.tag > max_tag) {
                is_load_waiting = st_entry.waiting_for_val;
                val_load = st_entry.val;
                max_tag = st_entry.tag;
              }
            }

            if (!is_load_waiting && max_tag == -1) {
              // Not found in store_q, so issue ld.
              val_load = PipelinedLSU::load(A.get_pointer() + idx_load);
              PRINTF("Loading from %d\n", idx_load);
            }

            // Setting 'consumer_load_succ' to false forces a write to consumer pipe.
            consumer_load_succ = is_load_waiting;
          }
        }
        if (!consumer_load_succ) {
          val_ld_pipe::write(val_load, consumer_load_succ);
        }
        /* End Load 1 Logic */
      
        /* Start Store 1 Logic */
        #pragma unroll
        for (uint i=0; i<STORE_Q_SIZE; ++i) {
          // Decrement count, and invalidate idexes if count below 0.
          const int count = store_entries[i].countdown;
          if (count > 0) store_entries[i].countdown--;
          // if (count == 1) num_enqued--;
          if (count <= 1 && !store_entries[i].waiting_for_val) store_entries[i].idx = -1;
        }

        // If store_q not full, check for new store_idx requests.
        bool is_stq_full = ((num_enqued % STORE_Q_SIZE) == 0 && num_enqued != 0) && (stq_head == stq_tail);
        PRINTF("is_stq_full %d\n", is_stq_full);
        if (!is_stq_full) {
          bool idx_store_pipe_succ = false;
          idx_tag_pair_store = idx_st_pipe::read(idx_store_pipe_succ);

          if (idx_store_pipe_succ) {
            idx_store = idx_tag_pair_store.first;
            tag_store = idx_tag_pair_store.second;

            // Requests with idx_store=-1 are only sent to update the store tag 
            // (lets loads know that this iteration doesn't store anything (e.g. conditional store)).
            if (idx_store != -1) {
              PRINTF("\ni_store_idx %d\n", i_store_idx);
              PRINTF("stq_head   %d --> ", stq_head);

              store_entries[stq_head] = {idx_store, tag_store, true};

              i_store_idx++;
              num_enqued++;
              stq_head = (stq_head+1) % STORE_Q_SIZE;
              PRINTF("%d\n\n", stq_head);
            }
            
          }
        }
        
        // If we have more store indexes read than store values, then check for new store_vals.
        if (i_store_idx > i_store_val) {
          bool val_store_pipe_succ = false;
          val_store = val_st_pipe::read(val_store_pipe_succ);

          if (val_store_pipe_succ) {
            // Add value to corresponding store entry, store to mem, and start counter.
            PRINTF("\n i_store_val %d\n", i_store_val);
            PRINTF("stq_tail   %d --> ", stq_tail);

            PRINTF("Storing to %d\n", store_entries[stq_tail].idx);
            PipelinedLSU::store(A.get_pointer() + store_entries[stq_tail].idx, val_store);
            store_entries[stq_tail].val = val_store;
            store_entries[stq_tail].countdown = STORE_LATENCY;
            store_entries[stq_tail].waiting_for_val = false;

            i_store_val++;
            stq_tail = (stq_tail+1) % STORE_Q_SIZE;
            PRINTF("%d\n\n", stq_tail);
          }
        }
        /* End Store 1 Logic */
      
        if (!end_signal) end_storeq_signal_pipe::read(end_signal);
      }

      // PRINTF("Finished Store Q\n")

    });
  });
  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}
