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

  // 35 cycles of stall
  using wet_pipe = pipe<class wet_pipe_class, int, 35>;
  using wet_pipe_2 = pipe<class wet_pipe_2_class, int, 35>;
  using wet_predicate_pipe = pipe<class wet_predicate_pipe_class, bool, 35>;
  using wet_predicate_pipe_2 = pipe<class wet_predicate_pipe_2_class, bool, 35>;


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

      wet_predicate_pipe_2::write(false);
    });
  });

  q.submit([&](handler &hnd) {
    accessor B(B_buf, hnd, write_only);

    hnd.single_task<class comp>([=]() [[intel::kernel_args_restrict]] {
      float etan, t = 0.0;
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
