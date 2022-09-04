#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

#include <sycl/ext/intel/fpga_extensions.hpp>

#include "store_queue.hpp"
#include "memory_utils.hpp"

using namespace sycl;

#ifndef Q_SIZE
  #define Q_SIZE 8
#endif

constexpr uint STORE_Q_SIZE = Q_SIZE;

constexpr int LAT = 23;

double histogram_kernel(queue &q, const std::vector<uint> &h_feature, const std::vector<uint> &h_weight,
                        std::vector<uint> &h_hist) {
#if dynamic_no_forward_sched
  constexpr bool IS_FORWARDING_Q = false;
  std::cout << "Dynamic (no forward) HLS\n";
#else
  constexpr bool IS_FORWARDING_Q = true;
  std::cout << "Dynamic HLS\n";
#endif

  const int array_size = h_feature.size();

  const auto feature = toDevice(h_feature, q);
  const auto weight = toDevice(h_weight, q);
  auto hist = toDevice(h_hist, q);

  constexpr int kNumLdPipes = 1;
  using idx_ld_pipe = pipe<class feature_load_pipe_class, int, 64>;
  using val_ld_pipes = PipeArray<class hist_load_pipe_class, uint, 64, kNumLdPipes>;
  using weight_load_pipe = pipe<class weight_load_pipe_class, uint, 64>;
  using idx_st_pipe = pipe<class feature_store_pipe_class, int, 64>;
  using val_st_pipe = pipe<class hist_store_pipe_class, uint, 64>;

  using end_storeq_signal_pipe = pipe<class end_storeq_signal_pipe_class, bool>;

  using ord_pipe = pipe<class ord_pipe_class, bool, 64>;
  using ord_ld_pipe = pipe<class ord_ld_pipe_class, bool, 64>;
  using ord_st_pipe = pipe<class ord_st_pipe_class, bool, 64>;


  q.submit([&](handler &hnd) {
    // Calculate ordering tokens for load from memory accesses.
    hnd.single_task<class OrdCalc>([=]() [[intel::kernel_args_restrict]] {
      int store_window[LAT+1];
      #pragma unroll
      for (int i = 0; i <= LAT; ++i) 
        store_window[i] = -1;

      [[intel::ivdep]]
      for (int i = 0; i < array_size; ++i) {
        uint idx = feature[i];
        bool stall = false;
        
        #pragma unroll
        for (int i = 0; i <= LAT; ++i) 
          stall |= (store_window[i] == idx);
        
        ord_ld_pipe::write(stall);

        #pragma unroll
        for (int i = 0; i < LAT; ++i) 
          store_window[i] = store_window[i+1];

        store_window[LAT] = idx;
      }
    });
  });

  q.submit([&](handler &hnd) {
    hnd.single_task<class LoadWeight1>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) {
        uint wt = weight[i];
        weight_load_pipe::write(wt);
      }
    });
  });

  auto event = q.submit([&](handler &hnd) {
    hnd.single_task<class MemoryDisambiguation>([=]() [[intel::kernel_args_restrict]] {
      int last_store_tag = -1;
      int last_commited_store_tag = -1;
      bool last_store_tag_succ = false;
      bool end_signal = false;
      int last_load_tag = -1;
      bool ord = false;
      bool ord_succ = false;
      bool val_succ = false;
      bool done_prev_ld = true;
      int8_t counter = -1; 

      bool store_window[LAT+1];
      #pragma unroll
      for (int i = 0; i <= LAT; ++i) 
        store_window[i] = false;

      [[intel::ivdep]]
      while (!end_signal || last_store_tag < last_load_tag) {
        #pragma unroll
        for (int i = 0; i < LAT; ++i) 
          store_window[i] = store_window[i+1];

        if (store_window[0]) {
          last_commited_store_tag++;
        }
        if (!ord && done_prev_ld) {
          ord = ord_ld_pipe::read(ord_succ);
          done_prev_ld = false;
        }

        if (ord_succ && (!ord || (last_load_tag == last_commited_store_tag))) {
          val_ld_pipes::PipeAt<0>::write(PipelinedLSU::load(hist + feature[last_load_tag]));
          last_load_tag++;
          ord = false;
          ord_succ = false;
          done_prev_ld = true;
        } // else wait for last_commited_store_tag to catch up

        val_succ = false;
        auto val = val_st_pipe::read(val_succ);

        if (val_succ) {
          last_store_tag++;
          PipelinedLSU::store((device_ptr<uint>(hist)) + feature[last_store_tag], val);
        }
        store_window[LAT] = val_succ;

        if (!end_signal) end_storeq_signal_pipe::read(end_signal);
      }
    });
  });

  q.submit([&](handler &hnd) {
    hnd.single_task<class Compute>([=]() [[intel::kernel_args_restrict]] {
      [[intel::ivdep]]
      for (int i = 0; i < array_size; ++i) {
        uint wt = weight_load_pipe::read();
        uint hist_val = val_ld_pipes::PipeAt<0>::read();
        auto new_hist = hist_val + wt;
        val_st_pipe::write(new_hist);
      }
      end_storeq_signal_pipe::write(0);
    });
  });

  event.wait();
  q.wait();
  q.memcpy(h_hist.data(), hist, sizeof(h_hist[0]) * h_hist.size()).wait();
  sycl::free(hist, q);
  sycl::free(feature, q);
  sycl::free(weight, q);

  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}

