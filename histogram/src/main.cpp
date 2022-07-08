#include <CL/sycl.hpp>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <vector>

#include <sycl/ext/intel/fpga_extensions.hpp>

#if static_sched
  #include "kernel_static.hpp"
#elif dynamic_sched
  #include "kernel_dynamic.hpp"
#else
  #include "kernel_dynamic_no_forward.hpp"
#endif

#include "tables.hpp"

using namespace sycl;

enum data_distribution { FORWARDING_FRIENDLY, NO_DEPENDENCIES, RANDOM };

void init_data(std::vector<uint> &feature, std::vector<uint> &weight, std::vector<uint> &hist,
               data_distribution distr) {
  for (int i = 0; i < feature.size(); i++) {
    if (distr == data_distribution::FORWARDING_FRIENDLY)
      feature[i] = (feature.size() >= 4) ? random_indx_1024[i % 1024] : i % feature.size();
    else if (distr == data_distribution::NO_DEPENDENCIES)
      feature[i] = i;
    else
      feature[i] = rand() % feature.size();

    weight[i] = (i % 2 == 0) ? 1 : 0;
    hist[i] = 0;
  }
}

void histogram_cpu(const std::vector<uint> &feature, const std::vector<uint> &weight,
                   std::vector<uint> &hist, const uint array_size) {
  for (int i = 0; i < array_size; ++i) {
    uint wt = weight[i];
    uint idx = feature[i];
    uint x = hist[idx];
    hist[idx] = x + wt;
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
  // Get A_SIZE and forward/no-forward from args.
  // defaulats
  uint ARRAY_SIZE = 64;
  auto DATA_DISTR = data_distribution::FORWARDING_FRIENDLY;
  try {
    if (argc > 1) {
      ARRAY_SIZE = uint(atoi(argv[1]));
    }
    if (argc > 2) {
      DATA_DISTR = data_distribution(atoi(argv[2]));
    }
}  catch (exception const &e) {
    std::cout << "Incorrect argv.\nUsage:\n";
    std::cout << "  ./hist [ARRAY_SIZE] [data_distribution (0/1/2)]\n";
    std::cout << "    0 - forward-friendly, 1 - no dependencies, 2 - random\n";
    std::terminate();
  }

#if FPGA_EMULATOR
  ext::intel::fpga_emulator_selector d_selector;
#elif FPGA
  ext::intel::fpga_selector d_selector;
#else
  default_selector d_selector;
#endif
  try {
    // Enable profiling.
    property_list properties{property::queue::enable_profiling()};
    queue q(d_selector, exception_handler, properties);
    queue q2(d_selector, exception_handler, properties);

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: " << q.get_device().get_info<info::device::name>() << "\n";

    std::cout << "Array size = " << ARRAY_SIZE << "\n";

    // host data
    // inputs
    std::vector<uint> feature(ARRAY_SIZE);
    std::vector<uint> weight(ARRAY_SIZE);
    std::vector<uint> hist(ARRAY_SIZE);

    init_data(feature, weight, hist, DATA_DISTR);

    std::vector<uint> hist_cpu(ARRAY_SIZE);
    std::copy(hist.begin(), hist.end(), hist_cpu.begin());

    auto start = std::chrono::steady_clock::now();
    double kernel_time = 0;

    #if (NO_FORWARD == 1)
      kernel_time = histogram_kernel_no_forward(q, feature, weight, hist);
    #else
      kernel_time = histogram_kernel(q, feature, weight, hist);
    #endif

    // Wait for all work to finish.
    q.wait();

    std::cout << "\nKernel time (ms): " << kernel_time << "\n";

    histogram_cpu(feature, weight, hist_cpu, ARRAY_SIZE);
    if (std::equal(hist.begin(), hist.end(), hist_cpu.begin())) {
      std::cout << "Passed\n";
    }
    else {
      std::cout << "Failed";
      std::cout << " sum(hist) = " << std::accumulate(hist.begin(), hist.end(), 0) << "\n";
    }

    auto stop = std::chrono::steady_clock::now();
    double total_time = (std::chrono::duration<double>(stop - start)).count() * 1000.0;
    // std::cout << "Total time (ms): " << total_time << "\n";
  } catch (exception const &e) {
    std::cout << "An exception was caught.\n";
    std::terminate();
  }

  return 0;
}
