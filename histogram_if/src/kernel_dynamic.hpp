#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#include <sycl/ext/intel/fpga_extensions.hpp>

#include "store_queue.hpp"

using namespace sycl;

#ifdef __SYCL_DEVICE_ONLY__
  #define CL_CONSTANT __attribute__((opencl_constant))
#else
  #define CL_CONSTANT
#endif

#define PRINTF(format, ...) { \
            static const CL_CONSTANT char _format[] = format; \
            sycl::ext::oneapi::experimental::printf(_format, ## __VA_ARGS__); }
using PipelinedLSU = ext::intel::lsu<>;

#ifndef Q_SIZE
  #define Q_SIZE 8
#endif

constexpr uint STORE_Q_SIZE = Q_SIZE;

/// <val, tag>
struct pair {
  int first; 
  int second; 
};

double histogram_if_kernel(queue &q, const std::vector<uint> &feature, 
                           const std::vector<uint> &weight, std::vector<uint> &hist) {
  std::cout << "Dynamic HLS\n";

  const uint array_size = feature.size();

  buffer feature_buf(feature);
  buffer weight_buf(weight);
  buffer hist_buf(hist);

  constexpr int kNumLdPipes = 1;
  using idx_ld_pipes = PipeArray<class feature_load_pipe_class, pair, 64, kNumLdPipes>;
  using val_ld_pipes = PipeArray<class hist_load_pipe_class, uint, 64, kNumLdPipes>;
  using val_st_pipe = pipe<class hist_store_pipe_class, uint, 64>;
  using idx_st_pipe = pipe<class feature_store_pipe_class, pair, 64>;

  using weight_load_pipe = pipe<class weight_load_pipe_class, uint, 64>;
  using weight_load_2_pipe = pipe<class weight_load_2_pipe_class, uint, 64>;
  using weight_load_3_pipe = pipe<class weight_load_3_pipe_class, uint, 64>;

  using calc_predicate_pipe = pipe<class calc_predicate_pipe_class, bool, 64>;
  using end_storeq_signal_pipe = pipe<class end_signal_pipe_class, bool>;

  q.submit([&](handler &hnd) {
    accessor weight(weight_buf, hnd, read_only);
    hnd.single_task<class LoadWeight>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) {
        uint wt = weight[i];
        weight_load_2_pipe::write(wt);
        weight_load_3_pipe::write(wt);

        if (wt > 0) {
          calc_predicate_pipe::write(1);
          weight_load_pipe::write(wt);
        }
      }
      calc_predicate_pipe::write(0);
    });
  });
  
  q.submit([&](handler &hnd) {
    accessor feature(feature_buf, hnd, read_only);
    hnd.single_task<class LoadFeature>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) {
        if (weight_load_2_pipe::read() > 0)
          idx_ld_pipes::PipeAt<0>::write({int(feature[i]), i});
      }
    });
  });
  q.submit([&](handler &hnd) {
    accessor feature(feature_buf, hnd, read_only);
    hnd.single_task<class LoadFeature2>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) {
        if (weight_load_3_pipe::read() > 0)
          idx_st_pipe::write({int(feature[i]), i});
        else 
          idx_st_pipe::write({-1, i});
      }
    });
  });
 
  StoreQueue<idx_ld_pipes, val_ld_pipes, kNumLdPipes, pair, 
             idx_st_pipe, val_st_pipe, end_storeq_signal_pipe, Q_SIZE, 12> (q, hist_buf);


  auto event = q.submit([&](handler &hnd) {
    hnd.single_task<class Compute>([=]() [[intel::kernel_args_restrict]] {
      while (calc_predicate_pipe::read()) {
        uint wt = weight_load_pipe::read();
        uint hist = val_ld_pipes::PipeAt<0>::read();

        auto new_hist = hist + wt;
        val_st_pipe::write(new_hist);
      }

      end_storeq_signal_pipe::write(0);
    });
  });



  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}

