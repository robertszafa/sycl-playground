#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#include <sycl/ext/intel/fpga_extensions.hpp>

#include "common.h"

using namespace sycl;

// using reg = ext::intel::fpga_reg;

class Compute;

double q_sim_kernel(queue &q, std::vector<uint> &problem, std::vector<cfloat> &state,
                    const uint n_controls) {
  std::cout << "Static HLS\n";

  const uint n_gates = problem.size();
  const uint n_states = state.size();

  buffer problem_buf(problem);
  buffer state_buf(state);

  event e = q.submit([&](handler &hnd) {
    accessor problem(problem_buf, hnd, read_write);
    accessor state(state_buf, hnd, read_write);

    // From Benchmarks for High-Level Synthesis (Jianyi Cheng)
    hnd.single_task<Compute>([=]() [[intel::kernel_args_restrict]] {
      for (uint i_gate = 0; i_gate < n_gates; i_gate += (2 + n_controls)) {
        uint gateCode = problem[i_gate];
        uint t = problem[i_gate + 1];

        // The below is not needed. We can just read from problem[i_gate + 2 + i_control]
        // for (int i = 0; i < n_controls; i++) {
        //   controls[i] = problem[i_gate + 2 + i];
        // }

        // Skip r gate for now since its construction has a larger latency.
        cfloat mat0, mat1, mat2, mat3;
        if (gateCode == 0) {
          mat0 = (cfloat){M_SQRT1_2, 0.0f};
          mat1 = (cfloat){M_SQRT1_2, 0.0f};
          mat2 = (cfloat){M_SQRT1_2, 0.0f};
          mat3 = (cfloat){-M_SQRT1_2, 0.0f};
        } else if (gateCode == 2) {
          mat0 = (cfloat){0.0f, 0.0f};
          mat1 = (cfloat){1.0f, 0.0f};
          mat2 = (cfloat){1.0f, 0.0f};
          mat3 = (cfloat){0.0f, 0.0f};
        } else if (gateCode == 3) {
          mat0 = (cfloat){0.0f, 0.0f};
          mat1 = (cfloat){0.0f, -1.0f};
          mat2 = (cfloat){0.0f, 1.0f};
          mat3 = (cfloat){0.0f, 0.0f};
        } else if (gateCode == 4) {
          mat0 = (cfloat){1.0f, 0.0f};
          mat1 = (cfloat){0.0f, 0.0f};
          mat2 = (cfloat){0.0f, 0.0f};
          mat3 = (cfloat){-1.0f, 0.0f};
        }

        for (uint i_state = 0; i_state < n_states; ++i_state) {
          // Get state indices.
          // Can these alias with other i_state iterations?
          int zero_state = nthCleared(i_state, t);
          int one_state = zero_state | (1 << t);

          // Perform the computation
          cpair inVec;
          inVec.a = state[zero_state];
          inVec.b = state[one_state];
          auto performZeroResult = cdot((cpair){mat0, mat1}, inVec);
          auto performOneResult = cdot((cpair){mat2, mat3}, inVec);

          // Process controls
          // See oneAPI-samples/DirectProgramming/DPC++FPGA/Tutorials/Features/fpga_reg
          // for using registers to achieve a skewed reduction tree for higher fmax.
          bool performZero = true;
          bool performOne = true;
          for (uint i_con = 0; i_con < n_controls; ++i_con) {
            // clang-format off
            auto this_control = problem[i_gate + 2 + i_con];
            int cond = this_control != ext::intel::fpga_reg(t);
            auto prevPerformZero = ext::intel::fpga_reg(performZero);
            performZero = prevPerformZero & (cond && ((1 << this_control) & ext::intel::fpga_reg(zero_state)) > 0) || (!cond && prevPerformZero);
            auto prevPerformOne = ext::intel::fpga_reg(performOne);
            performOne = prevPerformOne & (cond && ((1 << this_control) & ext::intel::fpga_reg(one_state)) > 0) || (!cond && prevPerformOne);
            // clang-format on
          }

          // Select result based on control.
          if (performZero)
            state[zero_state] = performZeroResult;
          if (performOne)
            state[one_state] = performOneResult;
        }
      }
    });
  });

  auto start = e.get_profiling_info<info::event_profiling::command_start>();
  auto end = e.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}
