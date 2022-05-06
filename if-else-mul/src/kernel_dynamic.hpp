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

double if_else_mul_kernel(queue &q, const std::vector<int> &wet, std::vector<float> &B,
                          const int array_size) {
  std::cout << "Dynamic HLS\n";

  buffer wet_buf(wet);
  buffer B_buf(B);

  constexpr unsigned int pipe_depth = 40;

  using read_wet_pipe = pipe<class p_00, int, pipe_depth>;
  using wet_pipe = pipe<class p_0, int, pipe_depth>;
  using etan_backedge_pipe = pipe<class p_1, float, pipe_depth>;
  using etan_pipe_true = pipe<class p_2, float, pipe_depth>;
  using etan_pipe_false = pipe<class p_3, float, pipe_depth>;
  using etan_pipe_out_true = pipe<class p_4, float, pipe_depth>;
  using etan_pipe_out_false = pipe<class p_5, float, pipe_depth>;
  using wet_pipe_true = pipe<class p_7, int, pipe_depth>;
  using wet_predicate_pipe = pipe<class p_8, bool, pipe_depth>;
  
  q.submit([&](handler &hnd) {
    accessor wet(wet_buf, hnd, read_only);

    hnd.single_task<class read_wet>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) {
        read_wet_pipe::write(wet[i]);
      }
    });
  });

  auto event = q.submit([&](handler &hnd) {
    // accessor wet(wet_buf, hnd, read_only);
    accessor B(B_buf, hnd, write_only);
    // sycl::stream out(1024, 256, hnd);

    hnd.single_task<class switch_unit>([=]() [[intel::kernel_args_restrict]] {
      float etan = 0.0;
      for (int i = 0; i < array_size; ++i) {
        // out << i << " ";

        int wet_val = read_wet_pipe::read();
        bool wet_predicate = (wet_val > 0);

        if (i > 0) {
          etan = etan_backedge_pipe::read();
        }

        if (wet_predicate) {
          wet_pipe_true::write(wet_val);
          etan_pipe_true::write(etan);
        } else {
          // sink wet
          etan_pipe_false::write(etan);
        }

        wet_predicate_pipe::write(wet_predicate);
      }

      etan = etan_backedge_pipe::read();
      B[0] = etan;
    });
  });

  q.submit([&](handler &hnd) {
    hnd.single_task<class comp_true>([=]() [[intel::kernel_args_restrict]] {
      while (true) {
        int wet = wet_pipe_true::read();
        float etan = etan_pipe_true::read();
        float t = 0.25 + etan * float(wet) / 2.0;
        etan += t;

        etan_pipe_out_true::write(etan);
      }
    });
  });

  q.submit([&](handler &hnd) {
    accessor B(B_buf, hnd, write_only);

    hnd.single_task<class comp_false>([=]() [[intel::kernel_args_restrict]] {
      while (true) {
        float etan = etan_pipe_false::read();
        etan += 0.25;

        etan_pipe_out_false::write(etan);
      }
    });
  });

  q.submit([&](handler &hnd) {
    accessor B(B_buf, hnd, write_only);

    hnd.single_task<class pick_out_etan>([=]() [[intel::kernel_args_restrict]] {
      while (true) {
        bool wet_predicate = wet_predicate_pipe::read();
        float etan;
        if (wet_predicate) {
          etan = etan_pipe_out_true::read();
        } else {
          etan = etan_pipe_out_false::read();
        }

        etan_backedge_pipe::write(etan);
      }
    });
  });

  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}
