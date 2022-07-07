#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <stdlib.h>

#include <sycl/ext/intel/fpga_extensions.hpp>

#if static_sched
#include "kernel_static.hpp"
#else 
#include "kernel_dynamic.hpp"
#endif
#include "tables.hpp"

using namespace sycl;

void init_data(std::vector<uint> &feature, std::vector<uint> &weight, std::vector<uint> &hist) {
  for (int i = 0; i < feature.size(); i++) {
    feature[i] = (feature.size() >= 4) 
                 ? random_indx_1024[i % 1024] 
                 : i % 3;
    // feature[i] = i;

    weight[i] = (i % 2 == 0) ? 1 : 0;
    hist[i] = 0;
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
    queue q2(d_selector, exception_handler, properties);

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: " << q.get_device().get_info<info::device::name>() << "\n";

    const uint ARRAY_SIZE = A_SIZE;

    std::cout << "Array size = " << ARRAY_SIZE << "\n";

    // host data
    // inputs
    std::vector<uint> feature(ARRAY_SIZE);
    std::vector<uint> weight(ARRAY_SIZE);
    std::vector<uint> hist(ARRAY_SIZE);

    init_data(feature, weight, hist);

    auto start = std::chrono::steady_clock::now();
    double kernel_time = 0;

    kernel_time = histogram_kernel(q, feature, weight, hist);

    // Wait for all work to finish.
    q.wait();
    
    std::cout << "\nKernel time (ms): " << kernel_time << "\n";
    std::cout << "sum(hist) = " << std::accumulate(hist.begin(), hist.end(), 0) << "\n";

    auto stop = std::chrono::steady_clock::now();
    double total_time = (std::chrono::duration<double> (stop - start)).count() * 1000.0;
    // std::cout << "Total time (ms): " << total_time << "\n";
  } catch (exception const &e) {
    std::cout << "An exception was caught.\n";
    std::terminate();
  }

  return 0;
}
