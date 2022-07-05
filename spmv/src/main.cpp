#include <CL/sycl.hpp>
#include <iostream>
#include <numeric>
#include <vector>
#if FPGA || FPGA_EMULATOR
  #include <sycl/ext/intel/fpga_extensions.hpp>
#endif

#if static_sched
  #include "kernel_static.hpp"
#else
  #include "kernel_dynamic.hpp"
#endif

using namespace sycl;

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

void init_data(std::vector<float> &matrix,  std::vector<float> &vec, std::vector<int> &col_index, 
               std::vector<int> &row_ptr,  const uint M) {
  for (int r = 0; r < M; ++r) {
    col_index[r] = r % 2;
    row_ptr[r] = (r + 8) % 4;
    vec[r] = float(1);

    for (int c = 0; c < M; ++c) {
      matrix[r * M + c] = 1;
    }
  }
}

uint dense2sparse(const std::vector<float> &matrix, std::vector<int> &col_index, 
                  std::vector<int> &row_ptr, const int M) {
  uint nz = 0;
  int start_row = -1;
  for (int r = 0; r < M; ++r) {
    int this_row_count = 0;
    for (int c = 0; c < M; ++c) {
      auto val = matrix[r * M + c];
      if (val != 0) {
        nz++;
        this_row_count++;
        col_index.push_back(c);
      }
    }
    row_ptr.push_back(this_row_count);
  }
  row_ptr.push_back(col_index.size());

  return nz;
}

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

    const int M = M_SIZE;

    std::vector<float> matrix(M * M);
    std::vector<float> vec(M);

    std::vector<int> row_ptr(M);
    std::vector<int> col_index(M);

    init_data(matrix, vec, col_index, row_ptr, M);
    // const uint nz = dense2sparse(matrix, col_index, row_ptr, M);

    auto start = std::chrono::steady_clock::now();
    double kernel_time = 0;

    spmv_kernel(q, matrix, row_ptr, col_index, vec, M);

    // Wait for all work to finish.
    q.wait();

    auto stop = std::chrono::steady_clock::now();
    double total_time = (std::chrono::duration<double>(stop - start)).count() * 1000.0;

    std::cout << "Kernel time (ms): " << kernel_time << "\n";
    float sum = std::accumulate(matrix.begin(), matrix.end(), 0.0);
    std::cout << "Sum = " << sum << "\n";

    // std::cout << "Total time (ms): " << total_time << "\n";
  } catch (exception const &e) {
    std::cout << "An exception was caught.\n";
    std::terminate();
  }

  return 0;
}
