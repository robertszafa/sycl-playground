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

// The default PipelinedLSU will start a load/store immediately, which the memory disambiguation 
// logic relies upon.
// A BurstCoalescedLSU would instead of waiting for more requests to arrive for a coalesced access.
// using PipelinedLSU = ext::intel::lsu<>;

#ifndef Q_SIZE
  #define Q_SIZE 8
#endif

constexpr uint STORE_Q_SIZE = Q_SIZE;



constexpr int LAT = 13;

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
  // using idx_ld_pipes = PipeArray<class feature_load_pipe_class, pair, 64, kNumLdPipes>;
  using idx_ld_pipes = pipe<class feature_load_pipe_class, pair, 64>;
  using val_ld_pipes = PipeArray<class hist_load_pipe_class, uint, 64, kNumLdPipes>;
  using weight_load_pipe = pipe<class weight_load_pipe_class, uint, 64>;
  // using idx_st_pipe = pipe<class feature_store_pipe_class, int, 64>;
  using idx_st_pipe = PipeArray<class feature_store_pipe_class, pair, 64, LAT>;
  using val_st_pipe = pipe<class hist_store_pipe_class, uint, 64>;

  using idx_ld_pipe2 = pipe<class feature_load_pipe_2_class, int, 64>;
  using idx_st_pipe2 = pipe<class feature_store_pipe_2_class, pair, 64>;

  using end_storeq_signal_pipe = pipe<class end_storeq_signal_pipe_class, bool>;

  using ord_ld_pipe = pipe<class ord_ld_pipe_class, bool, 64>;
  using ord_st_pipe = pipe<class ord_st_pipe_class, bool, 64>;
  using ord_done_pipe = pipe<class ord_done_pipe_class, bool, 64>;

  std::vector<uint8_t> h_ord(array_size);
  for (int i=0; i<h_feature.size(); ++i) {
    std::cout << h_feature[i] << ", ";
    auto base = &h_feature[std::max(i-LAT, 0)];
    auto end = &h_feature[i];
    h_ord[i] = (std::find(base, end, h_feature[i]) != end);
  }
  // std::cout << "\n";
  // for (int i=0; i<h_feature.size(); ++i) 
  //   std::cout << int(h_ord[i]) << ", ";
  // std::cout << "\n";
  const auto ord = toDevice(h_ord, q);

  q.submit([&](handler &hnd) {
    hnd.single_task<class LoadWeight1>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) {
        uint wt = weight[i];
        weight_load_pipe::write(wt);
      }
    });
  });

  q.submit([&](handler &hnd) {
    hnd.single_task<class LoadWeight2>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) {
        ord_ld_pipe::write(bool(ord[i]));
      }
    });
  });
  q.submit([&](handler &hnd) {
    hnd.single_task<class LoadWeight3>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 1; i < array_size; ++i) {
        ord_st_pipe::write(bool(ord[i]));
      }
    });
  });
  
  q.submit([&](handler &hnd) {
    hnd.single_task<class LoadFeature>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i)  {
        bool ord = ord_ld_pipe::read();
        if (ord) ord_done_pipe::read();
        val_ld_pipes::PipeAt<0>::write(PipelinedLSU::load(hist + feature[i]));
      }
    });
  });

  q.submit([&](handler &hnd) {
    hnd.single_task<class Storer>([=]() [[intel::kernel_args_restrict]] {
      [[intel::ivdep]]
      for (int i = 0; i < array_size; ++i) {
        auto val = val_st_pipe::read();
        bool ord = false;
        
        if (i < array_size-1)
          ord = ord_st_pipe::read();

        // TODO: latency achor
        // PipelinedLSU::store((device_ptr<uint>(hist)) + int(feature[i]), val,
        //                     ext::oneapi::experimental::properties(latency_anchor_id<0>));
        // if (ord) ord_done_pipe::write(1, ext::oneapi::experimental::properties(
        //              latency_constraint<0, latency_control_type::min, 10>));
        PipelinedLSU::store((device_ptr<uint>(hist)) + int(feature[i]), val);
        if (ord) ord_done_pipe::write(1);
        
      }
    });
  });


  auto event = q.submit([&](handler &hnd) {
    hnd.single_task<class Compute>([=]() [[intel::kernel_args_restrict]] {

      #pragma ivdep
      for (int i = 0; i < array_size; ++i) {
        // PRINTF("Compute i %d\n", i);

        uint wt = weight_load_pipe::read();
        uint hist_val = val_ld_pipes::PipeAt<0>::read();
        auto new_hist = hist_val + wt;
        val_st_pipe::write(new_hist);
      }

    });
  });

  q.wait();
  q.memcpy(h_hist.data(), hist, sizeof(h_hist[0]) * h_hist.size()).wait();
  sycl::free(hist, q);
  sycl::free(feature, q);
  sycl::free(weight, q);
  sycl::free(ord, q);

  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}

