#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <oneapi/dpl/random>
#include <vector>

#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

// Forward declare kernel names to avoid name mangling.
class Compute;

int long_func(oneapi::dpl::minstd_rand &engine, oneapi::dpl::uniform_int_distribution<int> &distr,
              const int array_size) {
  auto rand_0 = distr(engine) % (array_size - 1);
  auto rand_1 = distr(engine);
  auto rand_2 = distr(engine);
  auto rand_3 = distr(engine);
  return (rand_0 + rand_1 + rand_2 + rand_3) % (array_size - 1);
}

template <typename T>
double if_else_mul_kernel(queue &q, const std::vector<int> &wet, std::vector<T> &B,
                          const int array_size) {
  std::cout << "Static HLS\n";

  buffer wet_buf(wet);
  buffer B_buf(B);

  event e = q.submit([&](handler &hnd) {
    accessor wet(wet_buf, hnd, read_only);
    accessor B(B_buf, hnd, write_only);

    hnd.single_task<Compute>([=]() [[intel::kernel_args_restrict]] {
      oneapi::dpl::minstd_rand engine(77, 100);
      oneapi::dpl::uniform_int_distribution<int> distr;

      T etan, t = 0.0;
      for (int i = 0; i < array_size; ++i) {
        // RAW depenendency accross iterations for etan and engine.
        if (wet[i] > 0) {
          t = 0.25 + etan * T(wet[i]) / 2.0 + exp(etan);
          auto t = 0.25 + etan * T(wet[i]) / 2.0 + exp(etan);
          auto w_i = long_func(engine, distr, array_size);
          etan += etan + t + exp(t + etan + wet[w_i]);
        } else {
          etan -= 0.01;
        }

        B[i] = etan;
      }
    });
  });

  auto start = e.get_profiling_info<info::event_profiling::command_start>();
  auto end = e.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}
