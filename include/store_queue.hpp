/*
Robert Szafarczyk, Glasgow, 2022

Memory disambiguation kernel for C/C++/OpenCL/SYCL based HLS.
Store queue with early execution of loads when all preceding stores have calculated their addresses.
*/

#ifndef __STORE_QUEUE_HPP__
#define __STORE_QUEUE_HPP__

#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include <CL/sycl.hpp>

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
constexpr int kStoreCommitCycles = 72;
  
// Forward declaration to avoid name mangling.
class StoreQueueKernel;

/// Used for {idx, tag} pairs.
struct pair_t { int first; int second; };


template <typename ld_idx_pipes, typename ld_val_pipes, int num_lds,
          typename st_idx_pipe, typename st_val_pipe, typename end_signal_pipe, 
          bool FORWARD=true, int QUEUE_SIZE=8, int ST_LATENCY=12, typename value_t>
event StoreQueue(queue &q, device_ptr<value_t> data) {

  // How many cycles need to pass before the next iteration can start.
  // If we have a cycle per each CAM entry, then critical path doesn't increase. 
  constexpr int kLoopII = QUEUE_SIZE;
  // How many iterations need to pass before a store is commited to DRAM?
  // TODO: this will increase when mem. bandwidth becomes saturated -- then it needs to increase. 
  constexpr int kStoreIterationLatency = (kStoreCommitCycles + kLoopII) / kLoopII;
  // Don't waste bits for store_q indices.
  constexpr int kQueueLoopIterBitSize = fpga_tools::BitsForMaxValue<QUEUE_SIZE+1>();
  using storeq_idx_t = ac_int<kQueueLoopIterBitSize, false>;
  
  struct store_entry {
    int idx; 
    int tag; // TODO: use uint
    bool waiting_for_val;
    int16_t countdown;
    value_t value;
  };

  auto event = q.submit([&](handler &hnd) {
    hnd.single_task<StoreQueueKernel>([=]() [[intel::kernel_args_restrict]] {
      /// The store queue is a circular buffer.
      [[intel::fpga_register]] store_entry store_entries[QUEUE_SIZE];

      // Start with no valid entries in store queue.
      #pragma unroll
      for (uint i=0; i<QUEUE_SIZE; ++i) 
        store_entries[i] = {-1};

      // The below are variables kept around across iterations.
      bool end_signal = false;
      // How many store (valid) indexes were read from st_idx pipe.
      int i_store_idx = 0;
      // How many store values were accepted from st_val pipe.
      int i_store_val = 0;
      // Total number of stores to commit (supplied by the end_signal).
      int total_req_stores = 0;
      // Pointers into the store_entries circular buffer. Tail is for values, Head for idxs.
      storeq_idx_t stq_tail = 0;
      storeq_idx_t stq_head = 0;
      int tag_store = 0;
      bool is_store_idx_rq_finished = true;
      pair_t idx_tag_pair_store;
      // How many cycles need to pass between issuing last store and terminating the queue logic.
      int final_countdown = kStoreIterationLatency;

      // Scalar book-keeping values for the load logic (one per load, NTuple expanded at compile).
      NTuple<value_t, num_lds> val_load_tp;
      NTuple<pair_t, num_lds> idx_tag_pair_load_tp;
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


      /* Start Store Queue loop */
      // Setting Inititation Interval to the number of store_q entries ensures that we are not
      // increasing the critical path of the resulting circuit by too much, or not at all in most 
      // cases. ivdep (ignore mem dependencies): The logic guarantees dependencies are honoured.
      [[intel::initiation_interval(kLoopII)]]
      [[intel::ivdep]] 
      while (!end_signal || final_countdown > 0) {
        /* Start Load Logic */
        // All loads can proceed in parallel. The below unrolls the template PipeArray/NTuple. 
        UnrolledLoop<num_lds>([&](auto k) {
          // Use shorter names.
          auto& val_load = val_load_tp. template get<k>();
          auto& idx_tag_pair_load = idx_tag_pair_load_tp. template get<k>();
          auto& idx_load = idx_load_tp. template get<k>();
          auto& tag_load = tag_load_tp. template get<k>();
          auto& consumer_pipe_succ = consumer_load_succ_tp. template get<k>();
          auto& is_load_waiting = is_load_waiting_tp. template get<k>();
          auto& is_load_rq_finished = is_load_rq_finished_tp. template get<k>();

          // Check for new ld requests, only once the prev one was completed.
          if (is_load_rq_finished) {
            bool idx_load_pipe_succ = false;
            idx_tag_pair_load = ld_idx_pipes:: template PipeAt<k>::read(idx_load_pipe_succ);

            if (idx_load_pipe_succ) {
              is_load_rq_finished = false;
              idx_load = idx_tag_pair_load.first;
              tag_load = idx_tag_pair_load.second;
            }
          }

          if (!is_load_rq_finished) {
            // If the load tag sequence has overtaken the store tags, then we cannot possibly
            // disambiguate -- need to wait for more store idxs to arrive. 
            is_load_waiting = (tag_load > tag_store);
            int max_tag = -1; 

            // CAM search of the queue for the youngest store to the same idx as the ld request.
            // If found, and the store is not waiting for a value, then pass the value to the ld.
            #pragma unroll
            for (storeq_idx_t i=0; i<QUEUE_SIZE; ++i) {
              if (store_entries[i].idx == idx_load && // If found store with same idx as ld,
                  store_entries[i].tag <= tag_load && // make sure the store occured before the ld,
                  store_entries[i].tag > max_tag) {   // and it is the youngest that did so. 
                is_load_waiting |= store_entries[i].waiting_for_val;
                val_load = store_entries[i].value;
                max_tag = store_entries[i].tag;
              }
            }

            // If true, this means that the requested idx is not in the store queue.  
            // Else, val_load would have been assigned in the search loop above.
            if (!is_load_waiting && max_tag == -1) 
              val_load = PipelinedLSU::load(data + idx_load);

            // Setting consumer_load_succ=false forces a write to load consumer pipe.
            consumer_pipe_succ = is_load_waiting;
          }

          if (!consumer_pipe_succ) {
            // The ld. req. is deemed finished once the consumer pipe has been successfully written.
            ld_val_pipes:: template PipeAt<k>::write(val_load, consumer_pipe_succ);
            is_load_rq_finished = consumer_pipe_succ;
          }
        }); 
        /* End Load Logic */
      

        /* Start Store Logic */
        // On every iteration, decrement counter for stores in-flight.
        #pragma unroll
        for (storeq_idx_t i = 0; i < QUEUE_SIZE; ++i)
          if (store_entries[i].countdown > 0) store_entries[i].countdown--;

        // If previous idx_tag pair was processed, check for new a request.
        // Flip is_store_idx_rq_finished from '1' to '0', if new request arrives.
        bool idx_store_pipe_succ = false;
        if (is_store_idx_rq_finished) {
          idx_tag_pair_store = st_idx_pipe::read(idx_store_pipe_succ);
          is_store_idx_rq_finished = !idx_store_pipe_succ;
        }

        bool can_evict_head = can_evict_head = !store_entries[stq_head].waiting_for_val && 
                                               store_entries[stq_head].countdown <= 0;
        // If we've got an idx_tag req. to process, and we can evict the store entry at the head. 
        if (!is_store_idx_rq_finished && can_evict_head) {
          int idx_store = idx_tag_pair_store.first;
          tag_store = idx_tag_pair_store.second;
          is_store_idx_rq_finished = true;

          // idx=-1 is used for control dependent stores that were not issued for a given tag. 
          // We update the store_tag (so that dependent loads can proceed), but enqueue anything.
          if (idx_store != -1) {
            store_entries[stq_head] = {idx_store, tag_store, true};
            stq_head = (stq_head+1) % QUEUE_SIZE;
            i_store_idx++;
          }
        }

        // Only check for store value the corresponding index has been received (1:1 mapping).
        if (i_store_idx > i_store_val) {
          bool val_store_pipe_succ = false;
          value_t val_store = st_val_pipe::read(val_store_pipe_succ);

          if (val_store_pipe_succ) {
            store_entries[stq_tail].value = val_store;
            store_entries[stq_tail].waiting_for_val = false;
            PipelinedLSU::store(data + store_entries[stq_tail].idx, val_store);
            store_entries[stq_tail].countdown = int16_t(kStoreIterationLatency);
            
            i_store_val++;
            stq_tail = (stq_tail + 1) % QUEUE_SIZE;
          }
        }
        /* End Store Logic */

        // The end signal supplies the total number of stores supplied to the store queue.
        // Start the final_countdown once all stores have been issued.
        if (!end_signal) 
          total_req_stores = end_signal_pipe::read(end_signal);
        else if (end_signal && i_store_val >= (total_req_stores-1)) 
          final_countdown--;

      } /* End Store Queue loop */

    });
  });

  return event;
}

#endif
