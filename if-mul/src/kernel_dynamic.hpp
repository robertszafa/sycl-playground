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


double if_mul_kernel(queue &q, const std::vector<int> &wet, std::vector<float> &B, const int array_size) {
  std::cout << "Dynamic HLS\n";

  buffer wet_buf(wet);
  buffer B_buf(B);

  // TODO: pipe depths will need to be determined automatically.
  constexpr unsigned int pipe_depth = 2;
  using wet_pipe = pipe<class wet_pipe_class, int, pipe_depth>;
  using wet_pipe_2 = pipe<class wet_pipe_2_class, int, pipe_depth>;
  using wet_predicate_pipe = pipe<class wet_predicate_pipe_class, bool, pipe_depth>;
  using wet_predicate_pipe_2 = pipe<class wet_predicate_pipe_2_class, bool, pipe_depth>;

  /*
  The function is broken down into 3 kernels, 
  communicating via elastic buffers (pipes).
  TODO: do this automatically at the LLVM/SPIR-V level.
  */

  auto event = q.submit([&](handler &hnd) {
    accessor wet(wet_buf, hnd, read_only);

    hnd.single_task<class read_wet>([=]() [[intel::kernel_args_restrict]] {
      for (int i=0; i < array_size; ++i) {
        if (wet[i] > 0)
          wet_predicate_pipe::write(true);
        else
          wet_predicate_pipe::write(false);
      
        wet_pipe::write(wet[i]);
      } 
    });
  });

  q.submit([&](handler &hnd) {
    hnd.single_task<class switch_unit>([=]() [[intel::kernel_args_restrict]] {
      for (int i=0; i < array_size; ++i) {
        bool wet_predicate = wet_predicate_pipe::read();
        int wet = wet_pipe::read();
        if (wet_predicate) {
          wet_pipe_2::write(wet);
          wet_predicate_pipe_2::write(wet_predicate);
        }
        // else sink
      } 

      // Sends a "stop" signal to the FU that does the computation.
      // This is needed because we're using an "eta" node from PSSA.
      wet_predicate_pipe_2::write(false);
    });
  });

  q.submit([&](handler &hnd) {
    accessor B(B_buf, hnd, write_only);

    hnd.single_task<class comp>([=]() [[intel::kernel_args_restrict]] {
      float etan, t = 0.0;
      // This is more of an "eta" node from predicated SSA.
      // An "eta" node fires once for every "true" predicate, 
      // and stops altogether once a "false" predicate is encountered.
      while (wet_predicate_pipe_2::read()) {
        int wet = wet_pipe_2::read();
        t = 0.25 + etan * float(wet) / 2.0;
        etan += t;
      } 

      B[0] = etan;
    });
  });



  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}
