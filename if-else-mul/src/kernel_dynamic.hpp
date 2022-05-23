#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include "CL/sycl/queue.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#if FPGA || FPGA_EMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif

using namespace sycl;

class comp_true;
class switch_unit;


template<typename T>
double if_else_mul_kernel(queue &q, const std::vector<int> &wet, std::vector<T> &B,
                          const int array_size) {
  std::cout << "Dynamic HLS\n";

  buffer wet_buf(wet);
  buffer B_buf(B);

  using read_wet_pipe = pipe<class p_00, int, 0>;
  using wet_pipe = pipe<class p_0, int, 0>;
  using etan_backedge_pipe = pipe<class p_1, T, 0>;
  using wet_selector_pipe = pipe<class p_8, bool, 0>;

  using comp_false_predicate_pipe = pipe<class p_10, bool, 0>;
  using etan_pipe_false = pipe<class p_3, T, 0>;
  using etan_pipe_out_false = pipe<class p_5, T, 0>;

  using comp_true_predicate_pipe = pipe<class p_9, bool, 0>;
  using wet_pipe_true = pipe<class p_7, int, 0>;
  using etan_pipe_true = pipe<class p_2, T, 1>;
  using etan_pipe_out_true = pipe<class p_4, T, 0>;


  event e = q.submit([&](handler &hnd) {
    accessor B(B_buf, hnd, write_only);
    accessor wet(wet_buf, hnd, read_only);

    hnd.single_task<switch_unit>([=]() [[intel::kernel_args_restrict]] {
      T etan = 0.0;
      bool selector = false;

      for (int i = 0; i < array_size; ++i) {
        int wet_val = wet[i];
        bool wet_predicate = (wet_val > 0);

        // if (i > 0) {
          // etan = etan_backedge_pipe::read();
        if (selector)
          etan = etan_pipe_out_true::read();
          // else
          //   etan = etan_pipe_out_false::read();
        // }

        if (wet_predicate) {
          // comp_true_predicate_pipe::write(true);
          wet_pipe_true::write(wet_val);
          etan_pipe_true::write(etan);
        } else {
          // sink wet
          etan -= 0.01;
          // comp_false_predicate_pipe::write(true);
          // etan_pipe_false::write(etan);
        }

        // wet_selector_pipe::write(wet_predicate);
        selector = wet_predicate;
      }

      if (selector)
        etan = etan_pipe_out_true::read();
      // else
        // etan = etan_pipe_out_false::read();
      // etan = etan_backedge_pipe::read();

      B[0] = etan - 2.0;
      // comp_true_predicate_pipe::write(false);
      // comp_false_predicate_pipe::write(false);

    });
  });

  q.submit([&](handler &hnd) {
    hnd.single_task<comp_true>([=]() [[intel::kernel_args_restrict]] {
      // while (comp_true_predicate_pipe::read()) {
      while (1) {
        int wet = wet_pipe_true::read();
        auto etan = etan_pipe_true::read();
        auto t = 0.25 + etan * T(wet) / 2.0 + exp(etan);
        etan = etan + t + exp(t+etan);

        etan_pipe_out_true::write(etan);
      }
    });
  });

  // q.submit([&](handler &hnd) {
  //   hnd.single_task<class comp_false>([=]() [[intel::kernel_args_restrict]] {
  //     // while (comp_false_predicate_pipe::read()) {
  //     while (1) {
  //       auto etan = etan_pipe_false::read();
  //       etan -= 0.01;

  //       etan_pipe_out_false::write(etan);
  //     }
  //   });
  // });

  // q.submit([&](handler &hnd) {
  //   hnd.single_task<class pick_out_etan>([=]() [[intel::kernel_args_restrict]] {
  //     T etan;
  //     // for (int i = 0; i < array_size; ++i) {
  //     while (1) {
  //       bool wet_predicate = wet_selector_pipe::read();
  //       if (wet_predicate) {
  //         etan = etan_pipe_out_true::read();
  //       } else {
  //         etan = etan_pipe_out_false::read();
  //       }

  //       // if (i < (array_size - 1)) {
  //         etan_backedge_pipe::write(etan);
  //       // }
  //     }

  //   });
  // });

  auto start = e.get_profiling_info<info::event_profiling::command_start>();
  auto end = e.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}


