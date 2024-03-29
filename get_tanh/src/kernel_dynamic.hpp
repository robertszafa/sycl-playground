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


double get_tanh_kernel(queue &q, std::vector<int> &h_A, const std::vector<int> h_addr_in,
                       const std::vector<int> h_addr_out) {
#if dynamic_no_forward_sched
  constexpr bool IS_FORWARDING_Q = false;
  std::cout << "Dynamic (no forward) HLS\n";
#else
  constexpr bool IS_FORWARDING_Q = true;
  std::cout << "Dynamic HLS\n";
#endif

  const uint array_size = h_A.size();

  int* A = toDevice(h_A, q);
  int* addr_in = toDevice(h_addr_in, q);
  int* addr_out = toDevice(h_addr_out, q);

  using beta_in_pipe = pipe<class beta_in_pipe_class, int, 64>;
  using result_out_pipe = pipe<class result_out_pipe_class, int, 64>;
  
  using predicate_calc_pipe = pipe<class predicate_calc_pipe_class, bool, 64>;
  using end_storeq_signal_pipe = pipe<class end_storeq_signal_pipe_class, int>;

  constexpr int kNumLdPipes = 1;
  using idx_ld_pipes = PipeArray<class idx_ld_pipe_class, pair_t, 64, kNumLdPipes>;
  using val_ld_pipes = PipeArray<class val_ld_pipe_class, int, 64, kNumLdPipes>;
  using idx_st_pipe = pipe<class idx_st_pipe_class, pair_t, 64>;
  using val_st_pipe = pipe<class val_st_pipe_class, int, 64>;

  q.submit([&](handler &hnd) {
    hnd.single_task<class LoadIdxSt>([=]() [[intel::kernel_args_restrict]] {
      int tag = 0;
      for (int i = 0; i < array_size; i++) {
        int ld_i = addr_in[i];
        int st_i = addr_out[i];
        idx_ld_pipes::PipeAt<0>::write({ld_i, tag});
        tag++;
        idx_st_pipe::write({st_i, tag});
      }
    });
  });


  StoreQueue<idx_ld_pipes, val_ld_pipes, kNumLdPipes, idx_st_pipe, val_st_pipe, 
             end_storeq_signal_pipe, Q_SIZE> (q, device_ptr<int>(A));


  auto event = q.submit([&](handler &hnd) {
    hnd.single_task<class MainKernel>([=]() [[intel::kernel_args_restrict]] {
      int total_req_stores = 0;
      for (int i = 0; i < array_size; i++) {
        // Input angle
        auto beta = val_ld_pipes::PipeAt<0>::read(); // beta = A[addr_in[i]];
        // Result of tanh, sinh and cosh
        int result = 4096; // Saturation effect

        if (beta < 20480) {
          predicate_calc_pipe::write(1);
          beta_in_pipe::write(beta);
          result = result_out_pipe::read();
        }

        val_st_pipe::write(result); // A[addr_out[i]] = result;
        total_req_stores++;
      }

      predicate_calc_pipe::write(0);
      end_storeq_signal_pipe::write(total_req_stores);
    });
  });

  q.submit([&](handler &hnd) {

    hnd.single_task<class CalcKernel>([=]() [[intel::kernel_args_restrict]] {
      [[intel::fpga_register]] int atanh[12] = {0x08C9, 0x0416, 0x0202, 0x0100, 0x0080, 0x0064,
                       0x0032, 0x0010, 0x0008, 0x0004, 0x0002, 0x0001};
      [[intel::fpga_register]] int cosh[5] = {0x1000, 0x18B0, 0x3C31, 0xA115, 0x1B4EE};
      [[intel::fpga_register]] int sinh[5] = {0x0, 0x12CD, 0x3A07, 0xA049, 0x1B4A3};

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

  event.wait();
  q.copy(A, h_A.data(), h_A.size()).wait();

  sycl::free(A, q);
  sycl::free(addr_in, q);
  sycl::free(addr_out, q);

  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}
