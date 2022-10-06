/// Generic Store Queue 

#ifndef __STORE_QUEUE_HPP__
#define __STORE_QUEUE_HPP__

#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>

#include "pipe_utils.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"
#include "constexpr_math.hpp"


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
// logic relies upon. A BurstCoalescedLSU would instead of waiting for more requests to arrive for 
// a coalesced access. (Use sycl::ext::intel::experimental::lsu<> if want latency control).
using PipelinedLSU = ext::intel::lsu<>;
  
// Forward declaration to avoid name mangling.
class StoreQueueKernel;

struct pair {
  int first;
  int second;
};

template <typename ld_idx_pipes, typename ld_val_pipes, int num_lds,
          typename st_idx_pipe, typename st_val_pipe, typename end_signal_pipe, 
          bool FORWARD=true, int QUEUE_SIZE=8, int ST_LATENCY=12, typename T_val>
event StoreQueue(queue &q, device_ptr<T_val> data) {

  constexpr int kQueueLoopIterBitSize = fpga_tools::BitsForMaxValue<QUEUE_SIZE+1>();
  using storeq_idx_t = ac_int<kQueueLoopIterBitSize, false>;
  
  struct store_entry {
    int idx; // This should be the full address in a real impl.
    int tag;
    bool waiting_for_val;
    int16_t countdown;
  };

  auto event = q.submit([&](handler &hnd) {
    hnd.single_task<StoreQueueKernel>([=]() [[intel::kernel_args_restrict]] {
      [[intel::fpga_register]] store_entry store_entries[QUEUE_SIZE];
      [[intel::fpga_register]] T_val store_entries_val[QUEUE_SIZE];

      #pragma unroll
      for (uint i=0; i<QUEUE_SIZE; ++i) store_entries[i] = {-1};

      // How many store (valid) indexes were read from st_idx pipe.
      int i_store_idx = 0;
      // How many store values were accepted from st_val pipe.
      int i_store_val = 0;
      // How many stores have committed to memory
      int i_committed_stores = 0;
      // Total number of stores to commit (supplied by the end_signal).
      int total_req_stores = 0;
      // Pointers into the store_entries circular buffer. Tail is for values, Head for idxs.
      storeq_idx_t stq_tail = 0;
      storeq_idx_t stq_head = 0;

      int tag_store = 0;
      int idx_store;
      T_val val_store;
      pair idx_tag_pair_store;

      // TODO: to be more general, the initial load tags should be a template arg coming
      //       from program analysis: i.e. tags should reflect program order.
      // Scalar book-keeping values, one per load (NTuple is expanded at compile time).
      NTuple<T_val, num_lds> val_load_tp;
      NTuple<pair, num_lds> idx_tag_pair_load_tp;
      NTuple<int, num_lds> idx_load_tp;
      NTuple<int, num_lds> tag_load_tp;
      NTuple<bool, num_lds> consumer_load_succ_tp;
      NTuple<bool, num_lds> is_load_waiting_tp;
      NTuple<bool, num_lds> is_load_rq_finished_tp;

      UnrolledLoop<num_lds>([&](auto k) {
        consumer_load_succ_tp. template get<k>() = true;
        tag_load_tp. template get<k>() = 0;
        is_load_waiting_tp. template get<k>() = false;
        is_load_rq_finished_tp. template get<k>() = true;
      });

      bool end_signal = false;

      [[intel::ivdep]] 
      while (!end_signal || i_committed_stores < total_req_stores) {
        /* Start Load  Logic */
        // All loads can proceed in parallel. The below unrolls the template PipeArray/NTuple. 
        UnrolledLoop<num_lds>([&](auto k) {
          // Use shorter names.
          auto& val_load = val_load_tp. template get<k>();
          auto& idx_tag_pair_load = idx_tag_pair_load_tp. template get<k>();
          auto& idx_load = idx_load_tp. template get<k>();
          auto& tag_load = tag_load_tp. template get<k>();
          auto& consumer_load_succ = consumer_load_succ_tp. template get<k>();
          auto& is_load_waiting = is_load_waiting_tp. template get<k>();
          auto& is_load_rq_finished = is_load_rq_finished_tp. template get<k>();

          // Check for new ld requests.
          if (is_load_rq_finished) {
            bool idx_load_pipe_succ = false;
            idx_tag_pair_load = ld_idx_pipes:: template PipeAt<k>::read(idx_load_pipe_succ);

            if (idx_load_pipe_succ) {
              // Don't check this load port until the rq is served.
              is_load_rq_finished = false;
              idx_load = idx_tag_pair_load.first;
              tag_load = idx_tag_pair_load.second;
              // PRINTF("  idx_load = %d, tag_load = %d\n", idx_load, tag_load);
            }
          }

          if (!is_load_rq_finished && (consumer_load_succ || is_load_waiting)) {
            is_load_waiting = (tag_load > tag_store);
            
            if constexpr (FORWARD) {
              // If found, make sure it's the youngest store occuring before this ld
              // by finding the store with the max_tag, such that max_tag <= this_ld_tag
              int max_tag = -1; 
              #pragma unroll
              for (uint i=0; i<QUEUE_SIZE; ++i) {
                auto st_entry = store_entries[i];
                if (st_entry.idx == idx_load && st_entry.tag <= tag_load && st_entry.tag > max_tag) {
                  is_load_waiting |= st_entry.waiting_for_val;
                  val_load = store_entries_val[i];
                  max_tag = st_entry.tag;
                }
              }

              if (!is_load_waiting && max_tag == -1) {
                val_load = PipelinedLSU::load(data + idx_load);
              }
            }
            else {
              int max_tag = -1; 
              #pragma unroll
              for (storeq_idx_t i=0; i<QUEUE_SIZE; ++i) {
                is_load_waiting |= (store_entries[i].idx == idx_load && store_entries[i].tag <= tag_load);
              }

              if (!is_load_waiting) {
                val_load = PipelinedLSU::load(data + idx_load);
              }
            }

            // Setting 'consumer_load_succ' to false forces a write to load consumer pipe.
            consumer_load_succ = is_load_waiting;
          }

          if (!consumer_load_succ) {
            // The rq is deemed finished once the consumer pipe has been successfully written.
            ld_val_pipes:: template PipeAt<k>::write(val_load, consumer_load_succ);
            is_load_rq_finished = consumer_load_succ;

            // if (consumer_load_succ) PRINTF("  %d: val_load = %f\n", consumer_load_succ, val_load);
          }
        }); 
        /* End Load Logic */
      

        /* Start Store 1 Logic */
        bool is_space_in_stq = (store_entries[stq_head].idx == -1);
        #pragma unroll
        for (storeq_idx_t i = 0; i < QUEUE_SIZE; ++i) {
          if (store_entries[i].countdown < int16_t(1) && !store_entries[i].waiting_for_val &&
              store_entries[i].idx != -1)
            i_committed_stores++;

          // Invalidate idx if count WILL GO to 0 on this iter.
          if (store_entries[i].countdown < int16_t(1) && !store_entries[i].waiting_for_val) 
            store_entries[i].idx = -1;
          else 
            store_entries[i].countdown--;
        }

        // If store_q not full, check for new store_idx requests.
        if (is_space_in_stq) {
          bool idx_store_pipe_succ = false;
          idx_tag_pair_store = st_idx_pipe::read(idx_store_pipe_succ);

          if (idx_store_pipe_succ) {
            idx_store = idx_tag_pair_store.first;
            tag_store = idx_tag_pair_store.second;

            // PRINTF("  idx_store = %d, tag_store = %d\n", idx_store, tag_store);

            // Requests with idx_store=-1 are only sent to update the store tag 
            // (this is to deal with conditional stores that don't always occur).
            if (idx_store != -1) {
              store_entries[stq_head] = {idx_store, tag_store, true};

              i_store_idx++;
              stq_head = (stq_head+1) % QUEUE_SIZE;
            }
          }
        }
        
        // If we have read more store indexes than store values, then check for new store_vals.
        if (i_store_idx > i_store_val) {
          bool val_store_pipe_succ = false;
          val_store = st_val_pipe::read(val_store_pipe_succ);

          if (val_store_pipe_succ) {
            PipelinedLSU::store(data + store_entries[stq_tail].idx, val_store);
            store_entries[stq_tail].waiting_for_val = false;
            store_entries[stq_tail].countdown = int16_t(ST_LATENCY);
            if constexpr (FORWARD) store_entries_val[stq_tail] = val_store;

            // PRINTF("val_store = %f\n", val_store);
            i_store_val++;
            stq_tail = (stq_tail+1) % QUEUE_SIZE;
          }
        }
        /* End Store 1 Logic */
      
        // The end signal supplies the total number of stores that need to be 
        // committed before terminating the store queue logic.
        if (!end_signal) {
          total_req_stores = end_signal_pipe::read(end_signal);
        }
        
      }

      // PRINTF("stq_head == stq_tail = %d\n", stq_head == stq_tail);
      // PRINTF("stq_tail = %d\n", stq_tail);
      // PRINTF("Done Store Queue\n");
    });
  });

  return event;
}

#endif
