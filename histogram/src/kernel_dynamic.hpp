#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#include <sycl/ext/intel/fpga_extensions.hpp>

#include "store_queue.hpp"
#include "memory_utils.hpp"

using namespace sycl;

#ifndef Q_SIZE
  #define Q_SIZE 8
#endif

constexpr int STORE_Q_SIZE = Q_SIZE;


double histogram_kernel(queue &q, const std::vector<int> &h_feature, const std::vector<int> &h_weight,
                        std::vector<int> &h_hist) {
#if dynamic_no_forward_sched
  constexpr bool IS_FORWARDING_Q = false;
  std::cout << "Dynamic (no forward) HLS\n";
#else
  constexpr bool IS_FORWARDING_Q = true;
  std::cout << "Dynamic HLS\n";
#endif

  const int array_size = h_feature.size();

  int* feature = toDevice(h_feature, q);
  int* weight = toDevice(h_weight, q);
  int* hist = toDevice(h_hist, q);

  constexpr int kNumLdPipes = 1;
  using idx_ld_pipes = PipeArray<class feature_load_pipe_class, pair, 64, kNumLdPipes>;
  using val_ld_pipes = PipeArray<class hist_load_pipe_class, int, 64, kNumLdPipes>;
  using weight_load_pipe = pipe<class weight_load_pipe_class, int, 64>;
  using idx_st_pipe = pipe<class feature_store_pipe_class, pair, 64>;
  using val_st_pipe = pipe<class hist_store_pipe_class, int, 64>;

  using end_storeq_signal_pipe = pipe<class end_storeq_signal_pipe_class, bool>;

  q.submit([&](handler &hnd) {
    hnd.single_task<class LoadWeight>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) {
        int wt = weight[i];
        weight_load_pipe::write(wt);
      }
    });
  });

  constexpr int kNumStoreOps = 1;
  q.submit([&](handler &hnd) {
    hnd.single_task<class LoadFeature>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) 
        idx_ld_pipes::PipeAt<0>::write({int(feature[i]), i*kNumStoreOps + 0});
    });
  });
  q.submit([&](handler &hnd) {
    hnd.single_task<class LoadFeature2>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) 
        idx_st_pipe::write({int(feature[i]), i*kNumStoreOps + 1});
    });
  });

  auto event = q.submit([&](handler &hnd) {
    hnd.single_task<class Compute>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) {
        int wt = weight_load_pipe::read();
        int hist = val_ld_pipes::PipeAt<0>::read();

        auto new_hist = hist + wt;

        val_st_pipe::write(new_hist);
      }

      end_storeq_signal_pipe::write(0);
    });
  });

  StoreQueue<idx_ld_pipes, val_ld_pipes, kNumLdPipes, idx_st_pipe, val_st_pipe, 
             end_storeq_signal_pipe, IS_FORWARDING_Q, Q_SIZE, 12> (q, device_ptr<int>(hist)).wait();

  event.wait();
  q.copy(hist, h_hist.data(), h_hist.size()).wait();

  sycl::free(hist, q);
  sycl::free(feature, q);
  sycl::free(weight, q);

  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}

