#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include "CL/sycl/queue.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

// Forward declare kernel names to avoid name mangling.
class CompTrue;
class SwitchUnit;

template <typename T>
double if_else_mul_kernel(queue &q, const std::vector<int> &wet, std::vector<T> &B,
                          const int array_size) {
  std::cout << "Dynamic HLS\n";

  buffer wet_buf(wet);
  buffer B_buf(B);

  using read_wet_pipe = pipe<class p_00, int>;
  using wet_pipe = pipe<class p_0, int>;
  using etan_backedge_pipe = pipe<class p_1, T>;
  using wet_selector_pipe = pipe<class p_8, bool>;

  using comp_false_predicate_pipe = pipe<class p_10, bool>;
  using etan_pipe_false = pipe<class p_3, T>;
  using etan_pipe_out_false = pipe<class p_5, T>;

  using CompTrue_predicate_pipe = pipe<class p_9, bool>;
  using wet_pipe_true = pipe<class p_7, int>;
  using etan_pipe_true = pipe<class p_2, T>;
  using etan_pipe_out_true = pipe<class p_4, T>;

  event e = q.submit([&](handler &hnd) {
    accessor B(B_buf, hnd, write_only);
    accessor wet(wet_buf, hnd, read_only);

    hnd.single_task<SwitchUnit>([=]() [[intel::kernel_args_restrict]] {
      T etan = 0.0;
      bool selector = false;

      for (int i = 0; i < array_size; ++i) {
        int wet_val = wet[i];
        bool wet_predicate = (wet_val > 0);

        if (selector)
          etan = etan_pipe_out_true::read();

        if (wet_predicate) {
          wet_pipe_true::write(wet_val);
          etan_pipe_true::write(etan);
        } else {
          etan -= 0.01;
        }

        selector = wet_predicate;
      }

      if (selector)
        etan = etan_pipe_out_true::read();

      B[0] = etan - 2.0;
    });
  });

  q.submit([&](handler &hnd) {
    hnd.single_task<CompTrue>([=]() [[intel::kernel_args_restrict]] {
      while (1) {
        int wet = wet_pipe_true::read();
        auto etan = etan_pipe_true::read();
        auto t = 0.25 + etan * T(wet) / 2.0 + exp(etan);
        etan = etan + t + exp(t + etan);

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

  auto start = e.get_profiling_info<info::event_profiling::command_start>();
  auto end = e.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}
