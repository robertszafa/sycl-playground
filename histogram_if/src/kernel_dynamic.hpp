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
using namespace fpga_tools;


#ifndef Q_SIZE
  #define Q_SIZE 8
#endif

double histogram_if_kernel(queue &q, const std::vector<int> &h_feature, 
                           const std::vector<int> &h_weight, std::vector<int> &h_hist) {

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
  using idx_ld_pipes = PipeArray<class feature_load_pipe_class, pair_t, 64, kNumLdPipes>;
  using val_ld_pipes = PipeArray<class hist_load_pipe_class, int, 64, kNumLdPipes>;
  using val_st_pipe = pipe<class hist_store_pipe_class, int, 64>;
  using idx_st_pipe = pipe<class feature_store_pipe_class, pair_t, 64>;

  using weight_load_pipe = pipe<class weight_load_pipe_class, int, 64>;
  using weight_load_2_pipe = pipe<class weight_load_2_pipe_class, int, 64>;

  using weight_load_3_pipe = pipe<class weight_load_3_pipe_class, int, 64>;
  using hist_load_3_pipe = pipe<class hist_load_3_pipe_class, int, 64>;

  using calc_predicate_pipe = pipe<class calc_predicate_pipe_class, bool, 64>;
  using end_storeq_signal_pipe = pipe<class end_signal_pipe_class, int>;

  constexpr int kNumStoreOps = 1;
  q.submit([&](handler &hnd) {
    hnd.single_task<class LoadFeature>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) {
        idx_ld_pipes::PipeAt<0>::write({int(feature[i]), i*kNumStoreOps + 0});
      }
    });
  });
  q.submit([&](handler &hnd) {
    hnd.single_task<class LoadFeature2>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) {
        auto wt = weight[i];
        auto idx = (wt > 0) ? int(feature[i]) : -1;
        idx_st_pipe::write({idx, i*kNumStoreOps + 1});
      }
    });
  });

  StoreQueue<idx_ld_pipes, val_ld_pipes, kNumLdPipes, idx_st_pipe, val_st_pipe, 
             end_storeq_signal_pipe, Q_SIZE> (q, device_ptr<int>(hist));

  // q.submit([&](handler &hnd) {
  //   hnd.single_task<class ActualCalc>([=]() [[intel::kernel_args_restrict]] {
  //     while (calc_predicate_pipe::read()) {
  //       auto new_hist = hist_load_3_pipe::read() + weight_load_3_pipe::read();
  //       weight_load_2_pipe::write(new_hist);
  //     }
  //   });
  // });

  auto event = q.submit([&](handler &hnd) {
    hnd.single_task<class Compute>([=]() [[intel::kernel_args_restrict]] {
      int total_req_stores = 0;
      for (int i = 0; i < array_size; ++i) {
        auto wt = weight[i];
        int hist = val_ld_pipes::PipeAt<0>::read();

        if (wt > 0) {
          // calc_predicate_pipe::write(1);
          // hist_load_3_pipe::write(hist);
          // weight_load_3_pipe::write(wt);

          // int new_hist = weight_load_2_pipe::read();
          int new_hist = hist + wt;

          val_st_pipe::write(new_hist);
          total_req_stores++;
        }
        
      }

      end_storeq_signal_pipe::write(total_req_stores);
      // calc_predicate_pipe::write(0);
    });
  });

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

