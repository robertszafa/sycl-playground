/// Generic Store Queue 

#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#include <sycl/ext/intel/fpga_extensions.hpp>

#include "pipe_utils.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"


using namespace sycl;
using namespace fpga_tools;

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

// Forward declaration to avoid name mangling.
class StoreQueueKernel;

template <typename ld_idx_pipes, typename ld_val_pipes, int num_lds, typename T_idx_tag_pair,
          typename st_idx_pipe, typename st_val_pipe, typename end_signal_pipe, 
          bool FORWARD=true, uint QUEUE_SIZE=8, uint ST_LATENCY=12, typename T_val>
void StoreQueue(queue &q, buffer<T_val> &data_buf) {
  struct store_entry {
    int idx; // This should be the full address in a real impl.
    int tag;
    bool waiting_for_val;
    int countdown;
  };

  q.submit([&](handler &hnd) {
    accessor data(data_buf, hnd, read_write);

    hnd.single_task<StoreQueueKernel>([=]() [[intel::kernel_args_restrict]] {
      [[intel::fpga_register]] store_entry store_entries[QUEUE_SIZE];
      [[intel::fpga_register]] T_val store_entries_val[QUEUE_SIZE];

      #pragma unroll
      for (uint i=0; i<QUEUE_SIZE; ++i) {
        store_entries[i] = {-1, -2, 0, 0};
      }

      // How many store (valid) indexes were read from st_idx pipe.
      int i_store_idx = 0;
      // How many store values were accepted from st_val pipe.
      int i_store_val = 0;
      // Pointers into the store_entries circular buffer. Tail is for values, Head for idxs.
      int stq_tail = 0;
      int stq_head = 0;

      int tag_store = -1;
      int idx_store;
      T_val val_store;
      T_idx_tag_pair idx_tag_pair_store;

      NTuple<T_val, num_lds> val_load_tp;
      NTuple<int, num_lds> idx_load_tp;
      NTuple<int, num_lds> tag_load_tp;
      NTuple<bool, num_lds> consumer_load_succ_tp;
      NTuple<bool, num_lds> is_load_waiting_tp;
      NTuple<T_idx_tag_pair, num_lds> idx_tag_pair_load_tp;

      UnrolledLoop<num_lds>([&](auto k) {
        consumer_load_succ_tp. template get<k>() = true;
        tag_load_tp. template get<k>() = -1;
        is_load_waiting_tp. template get<k>() = false;
      });

      bool end_signal = false;

      [[intel::ivdep]] 
      while (!end_signal || (i_store_idx > i_store_val)) {
        /* Start Load  Logic */
        // All loads can proceed in parallel. The below unrolls the template pipe array. 
        UnrolledLoop<num_lds>([&](auto k) {
          // Use shorter names.
          auto& val_load = val_load_tp. template get<k>();
          auto& idx_load = idx_load_tp. template get<k>();
          auto& tag_load = tag_load_tp. template get<k>();
          auto& consumer_load_succ = consumer_load_succ_tp. template get<k>();
          auto& is_load_waiting = is_load_waiting_tp. template get<k>();
          auto& idx_tag_pair_load = idx_tag_pair_load_tp. template get<k>();

          if ((tag_load <= tag_store && consumer_load_succ) || is_load_waiting) {
            // Check for new ld requests.
            bool idx_load_pipe_succ = false;
            if (!is_load_waiting) {
              idx_tag_pair_load = ld_idx_pipes:: template PipeAt<k>::read(idx_load_pipe_succ);
            }

            // If new ld request, or if we are still waiting for a previous one, then check store_q.
            if (idx_load_pipe_succ || is_load_waiting) {
              idx_load = idx_tag_pair_load.first;
              tag_load = idx_tag_pair_load.second;

              is_load_waiting = false;
              
              if constexpr (FORWARD) {
                // If found, make sure it's the youngest store occuring before this ld
                // by finding the store with the max_tag, such that max_tag < this_ld_tag
                int max_tag = -1; 
                #pragma unroll
                for (uint i=0; i<QUEUE_SIZE; ++i) {
                  auto st_entry = store_entries[i];
                  if (st_entry.idx == idx_load && st_entry.tag < tag_load && st_entry.tag > max_tag) {
                    is_load_waiting = st_entry.waiting_for_val;
                    val_load = store_entries_val[i];
                    max_tag = st_entry.tag;
                  }
                }

                if (!is_load_waiting && max_tag == -1) {
                  // Not found in store_q, so issue ld.
                  val_load = PipelinedLSU::load(data.get_pointer() + idx_load);
                }
              }
              else {
                int max_tag = -1; 
                #pragma unroll
                for (uint i=0; i<QUEUE_SIZE; ++i) {
                  auto st_entry = store_entries[i];
                  if (st_entry.idx == idx_load && st_entry.tag < tag_load) {
                    is_load_waiting = true;
                  }
                }

                if (!is_load_waiting) {
                  // Not found in store_q, so issue ld.
                  val_load = PipelinedLSU::load(data.get_pointer() + idx_load);
                }
              }

              // Setting 'consumer_load_succ' to false forces a write to consumer pipe.
              consumer_load_succ = is_load_waiting;
            }
          }
          if (!consumer_load_succ) {
            ld_val_pipes:: template PipeAt<k>::write(val_load, consumer_load_succ);
          }
        });
        /* End Load Logic */
      
        /* Start Store 1 Logic */
        bool is_space_in_stq = false;
        #pragma unroll
        for (uint i=0; i<QUEUE_SIZE; ++i) {
          // Decrement count, and invalidate idexes if count below 0.
          const int count = store_entries[i].countdown;
          const int ste_idx = store_entries[i].idx;

          if (count > 0) store_entries[i].countdown--;
          if (ste_idx == -1) is_space_in_stq |= true;

          if (count == 1 && !store_entries[i].waiting_for_val) {
            store_entries[i].idx = -1;
            is_space_in_stq |= true;
          }
        }

        // If store_q not full, check for new store_idx requests.
        if (is_space_in_stq) {
          bool idx_store_pipe_succ = false;
          idx_tag_pair_store = st_idx_pipe::read(idx_store_pipe_succ);

          if (idx_store_pipe_succ) {
            idx_store = idx_tag_pair_store.first;
            tag_store = idx_tag_pair_store.second;

            // Requests with idx_store=-1 are only sent to update the store tag 
            // (lets loads know that this iteration doesn't store anything (e.g. conditional store)).
            if (idx_store != -1) {

              store_entries[stq_head] = {idx_store, tag_store, true};

              i_store_idx++;
              stq_head = (stq_head+1) % QUEUE_SIZE;
            }
            
          }
        }
        
        // If we have more store indexes read than store values, then check for new store_vals.
        if (i_store_idx > i_store_val) {
          bool val_store_pipe_succ = false;
          val_store = st_val_pipe::read(val_store_pipe_succ);

          if (val_store_pipe_succ) {
            // Add value to corresponding store entry, store to mem, and start counter.
            const auto stq_idx = stq_tail;
            PipelinedLSU::store(data.get_pointer() + store_entries[stq_idx].idx, val_store);
            store_entries[stq_idx].countdown = ST_LATENCY;
            store_entries[stq_idx].waiting_for_val = false;

            if constexpr (FORWARD) {
              store_entries_val[stq_idx] = val_store;
            }

            i_store_val++;
            stq_tail = (stq_tail+1) % QUEUE_SIZE;
          }
        }
        /* End Store 1 Logic */
      
        if (!end_signal) end_signal_pipe::read(end_signal);
      }

      // PRINTF("Finished Store Queue\n")

    });
  });

}
