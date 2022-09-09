#include <CL/sycl.hpp>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <vector>
#include <random>

#include <sycl/ext/intel/fpga_extensions.hpp>

#if static_sched
  #include "kernel_static.hpp"
#else
  #include "kernel_dynamic.hpp"
#endif

#include "tables.hpp"

using namespace sycl;

enum data_distribution { ALL_WAIT, NO_WAIT, PERCENTAGE_WAIT };

void init_data(std::vector<int> &edges, std::vector<int> &vertices, data_distribution distr, 
               const int num_edges, const uint percentage) {
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, 100);
  auto dice = std::bind (distribution, generator);

  edges[0] = 0;
  edges[1] = 1;
  vertices[0] = -1;
  vertices[1] = -1;
  for (int i = 2; i < num_edges*2; i += 2) {
    if (distr == data_distribution::ALL_WAIT) {
      edges[0] = 0;
      edges[1] = 2;
      edges[i] = i-2;
      edges[i+1] = i;
    }
    else if (distr == data_distribution::NO_WAIT) {
      edges[i] = i;
      edges[i+1] = i+1;
    }
    else {
      edges[i] = (dice() <= percentage) ? edges[i-2] : i;
      edges[i+1] = (dice() <= percentage) ? edges[i-1] : i+1;
    }

    vertices[i] = -1;
    vertices[i+1] = -1;
  }
}

int maximal_matching_cpu(const std::vector<int> &edges, std::vector<int> &vertices, const int num_edges) {
  int i = 0;
  int out = 0;

  while (i < num_edges) {

    int j = i * 2;

    int u = edges[j];
    int v = edges[j + 1];

    if ((vertices[u] < 0) && (vertices[v] < 0)) {
      vertices[u] = v;
      vertices[v] = u;

      out = out + 1;
    }

    i = i + 1;
  }

  return out;
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
  uint NUM_EDGES = 64;
  auto DATA_DISTR = data_distribution::ALL_WAIT;
  uint PERCENTAGE = 5;
  try {
    if (argc > 1) {
      NUM_EDGES = uint(atoi(argv[1]));
      if (NUM_EDGES < 2) throw std::invalid_argument("At least 2 edges rq.");
    }
    if (argc > 2) {
      DATA_DISTR = data_distribution(atoi(argv[2]));
    }
    if (argc > 3) {
      PERCENTAGE = uint(atoi(argv[3]));
      std::cout << "Percentage is " << PERCENTAGE << "\n";
      if (PERCENTAGE < 0 || PERCENTAGE > 100) throw std::invalid_argument("Invalid percentage.");
    }
  }  catch (exception const &e) {
    std::cout << "Incorrect argv.\nUsage:\n";
    std::cout << "  ./mm [ARRAY_SIZE] [data_distribution (0/1/2)] [PERCENTAGE (only for data_distr 2)]\n";
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

    std::cout << "Array size = " << NUM_EDGES << "\n";

    // host data
    // inputs
    std::vector<int> edges(NUM_EDGES*2);
    std::vector<int> vertices(NUM_EDGES*2);

    init_data(edges, vertices, DATA_DISTR, NUM_EDGES, PERCENTAGE);

    std::vector<int> vertices_cpu(NUM_EDGES*2);
    std::copy(vertices.begin(), vertices.end(), vertices_cpu.begin());

    auto start = std::chrono::steady_clock::now();
    double kernel_time = 0;

    int out = 0;

    kernel_time = maximal_matching_kernel(q, edges, vertices, &out, NUM_EDGES);

    // Wait for all work to finish.
    q.wait();

    std::cout << "\nKernel time (ms): " << kernel_time << "\n";

    int out_cpu = maximal_matching_cpu(edges, vertices_cpu, NUM_EDGES);
    if (out == out_cpu) {
      std::cout << "Passed\n";
    }
    else {
      std::cout << "Failed\n";
      std::cout << "  out fpga = " <<  out << "\n";
      std::cout << "  out cpu = " <<  out_cpu << "\n";
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
