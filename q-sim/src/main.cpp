#include <CL/sycl.hpp>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <stdlib.h>

#include <sycl/ext/intel/fpga_extensions.hpp>

#if static_sched
#include "kernel_static.hpp"
#else
#include "kernel_dynamic.hpp"
#endif

using TYPE = float;

using namespace sycl;

void init_data(std::vector<uint> &problem, std::vector<cfloat> &state, const uint n_controls) {
  const uint n_gates = problem.size();
  const uint n_states = state.size();

  for (uint i_gate = 0; i_gate < n_gates; i_gate += (2 + n_controls)) {
    problem[i_gate] = 2 + i_gate % 2; // gate code, choose only range [2-4]
    problem[i_gate+1] = rand(); // t

    for (uint i_control = 0; i_control < n_controls; ++i_control) {
      problem[i_gate + 2 + i_control] = rand() % 32; // I guess the range is [0-63]
    }
  }

  for (uint i_state = 0; i_state < n_states; i_state += (2 + n_controls)) {
    state[i_state] = (cfloat){1.0,0.0};;
  }
}

// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    } catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

int main(int argc, char *argv[]) {
  // Create device selector for the device of your interest.
#if FPGA_EMULATOR
  // DPC++ extension: FPGA emulator selector on systems without FPGA card.
  ext::intel::fpga_emulator_selector d_selector;
#elif FPGA
  // DPC++ extension: FPGA selector on systems with FPGA card.
  ext::intel::fpga_selector d_selector;
#elif GPU
  gpu_selector d_selector;
#else
  // The default device selector will select the most performant device.
  default_selector d_selector;
#endif

  try {
    // Enable profiling.
    property_list properties{property::queue::enable_profiling()};
    queue q(d_selector, exception_handler, properties);

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: " << q.get_device().get_info<info::device::name>() << "\n";

    // host data
    // inputs
    uint n_qubits = 4; 
    uint n_controls = 2; 
    uint n_gates = 4; 
    uint n_states = std::pow(2, n_qubits); 
    // each i_problem has: 1 gate_code; 1 t; n controls 
    std::vector<uint> problem(n_gates * (n_controls + 2)); 
    std::vector<cfloat> state(n_states);

    init_data(problem, state, n_controls);

    auto start = std::chrono::steady_clock::now();
    double kernel_time = 0;

    kernel_time = q_sim_kernel(q, problem, state, n_controls);

    // Wait for all work to finish.
    // q.wait();

    std::cout << "\nKernel time (ms): " << kernel_time << "\n";
    std::cout << "state[0].x = " << state[0].x << "\n";

    auto stop = std::chrono::steady_clock::now();
    double total_time = (std::chrono::duration<double>(stop - start)).count() * 1000.0;
    // std::cout << "Total time (ms): " << total_time << "\n";
  } catch (exception const &e) {
    std::cout << "An exception was caught.\n";
    std::terminate();
  }

  return 0;
}
