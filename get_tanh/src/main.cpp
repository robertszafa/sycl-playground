#include <CL/sycl.hpp>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
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

using namespace sycl;

enum data_distribution { ALL_WAIT, NO_WAIT, PERCENTAGE_WAIT };

void init_data(std::vector<int> &A, std::vector<int> &addr_in, std::vector<int> &addr_out,
               const data_distribution distr, const uint percentage) {
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, 100);
  auto dice = std::bind(distribution, generator);

  for (int i = 0; i < A.size(); i++) {
    A[i] = 30000; // i % K == 0? rand()%1000 : 30000;

    if (distr == data_distribution::ALL_WAIT) {
      addr_in[i] = std::max(i - 1, 0);
      addr_out[i] = std::min(max(i + 1, 0), int(A.size()-1));
    } else if (distr == data_distribution::NO_WAIT) {
      addr_in[i] = i;
      addr_out[i] = i;
    } else {
      addr_in[i] = (dice() <= percentage) ? std::max(i - 1, 0) : i;
      addr_out[i] = addr_in[i];
    }
  }
}

void get_tanh_cpu(std::vector<int> &A, const std::vector<int> addr_in,
                  const std::vector<int> addr_out) {
  int atanh[12] = {0x08C9, 0x0416, 0x0202, 0x0100, 0x0080, 0x0064,
                   0x0032, 0x0010, 0x0008, 0x0004, 0x0002, 0x0001};
  int cosh[5] = {0x1000, 0x18B0, 0x3C31, 0xA115, 0x1B4EE};
  int sinh[5] = {0x0, 0x12CD, 0x3A07, 0xA049, 0x1B4A3};

  for (int i = 0; i < A.size(); i++) {
    // Input angle
    int beta = A[addr_in[i]];
    // Result of tanh, sinh and cosh
    int result = 4096; // Saturation effect

    // Implement approximate range of the hyperbolic CORDIC block
    if (beta < 20480) {
      int x = 0x1351;
      int y = 0;
      int x_new;
      int index_trigo;
      int result_cosh, result_sinh;
      int outputcosh, outputsinh;

      if (beta >= 8192) {
        index_trigo = 4;
      } else if (beta >= 12288) {
        index_trigo = 3;
      } else if (beta >= 8192) {
        index_trigo = 2;
      } else if (beta >= 4096) {
        index_trigo = 1;
      } else {
        index_trigo = 0;
      }
      beta = beta - index_trigo * 4096;

      // Call to the hyperbolic CORDIC block
      for (int k = 1; k <= 12; k++) {
        // force the 3k+1 th iteration to be repeated
        if (((k % 3) == 1) && (k != 1)) {
          for (int j = 1; j <= 2; j++) {
            // beta<0 anti-clockwise rotation
            if (beta < 0) {
              x_new = x - (y >> k);
              y -= x >> k;
              beta += atanh[k - 1];
            }
            // beta>0 clockwise rotation
            else {
              x_new = x + (y >> k);
              y += (x >> k);
              beta -= atanh[k - 1];
            }
            x = x_new;
          }
        } else {
          if (beta < 0) {
            x_new = x - (y >> k);
            y -= x >> k;
            beta += atanh[k - 1];
          }
          // beta>0 clockwise rotation
          else {
            x_new = x + (y >> k);
            y += (x >> k);
            beta -= atanh[k - 1];
          }
          x = x_new;
        }
      }
      outputcosh = x;
      outputsinh = y;

      // Trigonometric rules application
      result_cosh = (sinh[index_trigo] * outputcosh + cosh[index_trigo] * outputsinh);
      result_sinh = (cosh[index_trigo] * outputcosh + sinh[index_trigo] * outputsinh) >> 12;
      result = result_cosh / result_sinh;
    }

    // Central symmetry correction
    A[addr_out[i]] = result;
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
  auto DATA_DISTR = data_distribution::ALL_WAIT;
  uint PERCENTAGE = 5;
  try {
    if (argc > 1) {
      ARRAY_SIZE = uint(atoi(argv[1]));
    }
    if (argc > 2) {
      DATA_DISTR = data_distribution(atoi(argv[2]));
    }
    if (argc > 3) {
      PERCENTAGE = uint(atoi(argv[3]));
      std::cout << "Percentage is " << PERCENTAGE << "\n";
      if (PERCENTAGE < 0 || PERCENTAGE > 100)
        throw std::invalid_argument("Invalid percentage.");
    }
  } catch (exception const &e) {
    std::cout << "Incorrect argv.\nUsage:\n";
    std::cout << "  ./hist [ARRAY_SIZE] [data_distribution (0/1/2)] [PERCENTAGE (only for "
                 "data_distr 2)]\n";
    std::cout << "    0 - all_wait, 1 - no_wait, 2 - PERCENTAGE wait\n";
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
    std::vector<int> A(ARRAY_SIZE);
    std::vector<int> addr_in(ARRAY_SIZE);
    std::vector<int> addr_out(ARRAY_SIZE);

    init_data(A, addr_in, addr_out, DATA_DISTR, PERCENTAGE);

    std::vector<int> A_cpu(ARRAY_SIZE);
    std::copy(A.begin(), A.end(), A_cpu.begin());

    auto start = std::chrono::steady_clock::now();
    double kernel_time = 0;

    kernel_time = get_tanh_kernel(q, A, addr_in, addr_out);

    // Wait for all work to finish.
    q.wait();

    std::cout << "\nKernel time (ms): " << kernel_time << "\n";

    get_tanh_cpu(A_cpu, addr_in, addr_out);
    if (std::equal(A.begin(), A.end(), A_cpu.begin())) {
      std::cout << "Passed\n";
    } else {
      std::cout << "Failed";
      std::cout << " sum(A_fpga) = " << std::accumulate(A.begin(), A.end(), 0) << "\n";
      std::cout << " sum(A_cpu) = " << std::accumulate(A_cpu.begin(), A_cpu.end(), 0) << "\n";
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
