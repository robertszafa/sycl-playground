#include <CL/sycl.hpp>
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

template <typename T>
void init_data(std::vector<T> &a, std::vector<T> &r, const uint N, const uint M) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      a[i * N + j] = rand() / 1000.0f;
      r[i * N + j] = rand() / 1000.0f;
    }
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

    const uint N = 64;
    const uint M = 64;

    // host data
    // inputs
    std::vector<TYPE> a(N*M);
    std::vector<TYPE> r(N*M);
    srand(9);

    init_data(a,r, N, M);

    auto start = std::chrono::steady_clock::now();
    double kernel_time = 0;

    kernel_time = gram_schmidt_kernel<TYPE>(q, a, r, N, M);

    // Wait for all work to finish.
    // q.wait();

    std::cout << "\nKernel time (ms): " << kernel_time << "\n";
    std::cout << "r[0] = " << r[0] << "\n";

    auto stop = std::chrono::steady_clock::now();
    double total_time = (std::chrono::duration<double>(stop - start)).count() * 1000.0;
    // std::cout << "Total time (ms): " << total_time << "\n";
  } catch (exception const &e) {
    std::cout << "An exception was caught.\n";
    std::terminate();
  }

  return 0;
}
