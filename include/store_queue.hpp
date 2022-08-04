/// Generic Store Queue 

#pragma once

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
// logic relies upon.
// A BurstCoalescedLSU would instead of waiting for more requests to arrive for a coalesced access.
using PipelinedLSU = ext::intel::lsu<>;

// Forward declaration to avoid name mangling.
class StoreQueueKernel;


template <typename ld_idx_pipes, typename ld_val_pipes, int num_lds, typename idx_tag_pair_t, 
          typename st_idx_pipe, typename st_val_pipe, typename end_signal_pipe, typename base_ld_tags,
          int base_st_tag, int NUM_TAGS, bool FORWARD=true, int QUEUE_SIZE=8, int ST_LATENCY=12, 
          typename T_val>
void StoreQueue(queue &q, buffer<T_val> &data_buf) {

  constexpr int kQueueLoopIterBitSize = fpga_tools::BitsForMaxValue<QUEUE_SIZE+1>();
  using storeq_idx_t = ac_int<kQueueLoopIterBitSize, false>;

  // Extract the tag type (usually some kind of n-tuple with a comp. operator).
  using tag_t = NTuple<int, NUM_TAGS>; // decltype(idx_tag_pair_t{}.second);
  /// One less than lowest loop iter value .
  tag_t base_tag;
  UnrolledLoop<NUM_TAGS>([&](auto k) { base_tag. template get<k>() = -1; } );
  // auto& kNumTags = idx_tag_pair_t{}.second.NumTys;

  auto is_tag_younger = [&] (tag_t& first, tag_t& second) {
    bool is_younger = first. template get<0>() <= second. template get<0>();
    UnrolledLoop<1, NUM_TAGS-1>([&](auto j) {
      is_younger = is_younger && (first. template get<j>() < second. template get<j>());
    });

    if constexpr (NUM_TAGS == 1) return is_younger;

    return is_younger && (first. template get<NUM_TAGS-1>() < second. template get<NUM_TAGS-1>());
  };

  struct store_entry {
    int idx; // This should be the full address in a real impl.
    tag_t tag;
    bool waiting_for_val;
    int8_t countdown;
  };

  q.submit([&](handler &hnd) {
    accessor data(data_buf, hnd, read_write);

    hnd.single_task<StoreQueueKernel>([=]() [[intel::kernel_args_restrict]] {
      [[intel::fpga_register]] store_entry store_entries[QUEUE_SIZE];
      [[intel::fpga_register]] T_val store_entries_val[QUEUE_SIZE];
      // [[intel::fpga_register]] tag_t store_entries_tag[QUEUE_SIZE];

      #pragma unroll
      for (uint i=0; i<QUEUE_SIZE; ++i) {
        store_entries[i] = {-1}; //, base_tag, 0, 0};
      }

      // How many store (valid) indexes were read from st_idx pipe.
      int i_store_idx = 0;
      // How many store values were accepted from st_val pipe.
      int i_store_val = 0;
      // Pointers into the store_entries circular buffer. Tail is for values, Head for idxs.
      storeq_idx_t stq_tail = 0;
      storeq_idx_t stq_head = 0;

      tag_t tag_store;
      tag_store. template get<0>() = -1;
      tag_store. template get<1>() = 1;

      int idx_store;
      T_val val_store;
      idx_tag_pair_t idx_tag_pair_store;

      NTuple<T_val, num_lds> val_load_tp;
      NTuple<int, num_lds> idx_load_tp;
      NTuple<tag_t, num_lds> tag_load_tp;
      NTuple<bool, num_lds> consumer_load_succ_tp;
      NTuple<bool, num_lds> is_load_waiting_tp;
      NTuple<idx_tag_pair_t, num_lds> idx_tag_pair_load_tp;

      UnrolledLoop<num_lds>([&](auto k) {
        consumer_load_succ_tp. template get<k>() = true;
        // tag_load_tp. template get<k>() = base_tag;
        tag_load_tp. template get<k>(). template get<0>() = -1;
        tag_load_tp. template get<k>(). template get<1>() = 0;
        is_load_waiting_tp. template get<k>() = false;
      });

      bool end_signal = false;

      [[intel::ivdep]] 
      while (!end_signal || (i_store_idx > i_store_val)) {
        /* Start Load  Logic */
        // All loads can proceed in parallel. The below unrolls the template pipe array. 
        // PRINTF("tag_load_tp.0 %d  tag_load_tp.1 %d\n", std::get<0>(tag_load_tp. template get<0>()), 
        //                                              std::get<1>(tag_load_tp. template get<0>()));
        UnrolledLoop<num_lds>([&](auto k) {
          // Use shorter names.
          auto& val_load = val_load_tp. template get<k>();
          auto& idx_load = idx_load_tp. template get<k>();
          auto& tag_load = tag_load_tp. template get<k>();
          auto& consumer_load_succ = consumer_load_succ_tp. template get<k>();
          auto& is_load_waiting = is_load_waiting_tp. template get<k>();
          auto& idx_tag_pair_load = idx_tag_pair_load_tp. template get<k>();

          if (is_load_waiting || (is_tag_younger(tag_load, tag_store) && consumer_load_succ)) {
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
                tag_t max_tag = base_tag; 
                bool is_match = false;
                #pragma unroll
                for (storeq_idx_t i=0; i<QUEUE_SIZE; ++i) {
                  auto st_entry_tag = store_entries[i].tag;
                  const bool entry_before_ld = is_tag_younger(st_entry_tag, tag_load); 
                  const bool entry_youngest = is_tag_younger(max_tag, st_entry_tag);
                  if (store_entries[i].idx == idx_load && entry_before_ld && entry_youngest) {
                    is_match = true;
                    is_load_waiting = store_entries[i].waiting_for_val;
                    val_load = store_entries_val[i];
                    max_tag = st_entry_tag;
                  }
                }

                if (!is_load_waiting && is_match) {
                  // Not found in store_q, so issue ld.
                  val_load = PipelinedLSU::load(data.get_pointer() + idx_load);
                }
              }
              else {
                tag_t max_tag = base_tag; 
                #pragma unroll
                for (storeq_idx_t i=0; i<QUEUE_SIZE; ++i) {
                  if (store_entries[i].idx == idx_load) {
                    if (is_tag_younger(store_entries[i].tag, tag_load)) {
                      is_load_waiting = true;
                    }
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
        for (storeq_idx_t i=0; i<QUEUE_SIZE; ++i) {
          // Decrement count, and invalidate idexes if count below 0.
          const auto count = store_entries[i].countdown;
          const auto ste_idx = store_entries[i].idx;

          if (count > int8_t(0)) store_entries[i].countdown--;
          if (ste_idx == -1) is_space_in_stq |= true;

          if (count == int8_t(1) && !store_entries[i].waiting_for_val) {
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
              // store_entries_tag[stq_head] = tag_store;

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
            store_entries[stq_idx].countdown = int8_t(ST_LATENCY);
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
