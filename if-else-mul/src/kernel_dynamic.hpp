#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include "CL/sycl/queue.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <oneapi/dpl/random>
#include <vector>

#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

int long_func(oneapi::dpl::minstd_rand &engine, oneapi::dpl::uniform_int_distribution<int> &distr,
              const int array_size) {
  auto rand_0 = distr(engine) % (array_size - 1);
  auto rand_1 = distr(engine);
  auto rand_2 = distr(engine);
  auto rand_3 = distr(engine);
  return (rand_0 + rand_1 + rand_2 + rand_3) % (array_size - 1);
}

// Forward declare kernel names to avoid name mangling.
class CompTrue;
class SwitchUnit;
class Read;
class Write;

// Forward declare pipe class names to avoid name mangling.
class read_wet_pipe_class;
class wet_pipe_class;
class etan_backedge_pipe_class;
class wet_selector_pipe_class;

class comp_false_predicate_pipe_class;
class etan_pipe_false_class;
class etan_pipe_out_false_class;

class comp_true_predicate_pipe_class;
class wet_pipe_true_class;
class etan_pipe_true_class;
class etan_pipe_out_true_class;

template <typename T>
double if_else_mul_kernel(queue &q, const std::vector<int> &wet, std::vector<T> &B,
                          const int array_size) {
  std::cout << "Dynamic HLS\n";

  buffer wet_buf(wet);
  buffer B_buf(B);

  using read_wet_pipe = pipe<read_wet_pipe_class, int>;
  using wet_pipe = pipe<wet_pipe_class, int>;
  using etan_backedge_pipe = pipe<etan_backedge_pipe_class, T>;
  using wet_selector_pipe = pipe<wet_selector_pipe_class, bool>;

  using comp_false_predicate_pipe = pipe<comp_false_predicate_pipe_class, bool>;
  using etan_pipe_false = pipe<etan_pipe_false_class, T>;
  using etan_pipe_out_false = pipe<etan_pipe_out_false_class, T>;

  using comp_true_predicate_pipe = pipe<comp_true_predicate_pipe_class, bool>;
  using wet_pipe_true = pipe<wet_pipe_true_class, int>;
  using etan_pipe_true = pipe<etan_pipe_true_class, T>;
  using etan_pipe_out_true = pipe<etan_pipe_out_true_class, T>;

  event e = q.submit([&](handler &hnd) {
    accessor B(B_buf, hnd, write_only);
    accessor wet(wet_buf, hnd, read_only);

    hnd.single_task<SwitchUnit>([=]() [[intel::kernel_args_restrict]] {
      T etan;

      for (int i = 0; i < array_size; ++i) {
        int wet_val = wet[i];
        bool wet_predicate = (wet_val > 0);

        // Extract only the long latency branch.
        // A more general approach would be to extract all branches
        // (if their latency differs), and have a "pick" unit at the end.
        if (wet_predicate) {
          comp_true_predicate_pipe::write(true);
          etan_pipe_true::write(etan);
          wet_pipe_true::write(wet_val);
          etan = etan_pipe_out_true::read();
        } else {
          etan -= 0.01;
        }

        B[i] = etan;
      }

      comp_true_predicate_pipe::write(false);
    });
  });

  q.submit([&](handler &hnd) {
    accessor wet(wet_buf, hnd, read_only);

    hnd.single_task<CompTrue>([=]() [[intel::kernel_args_restrict]] {
      oneapi::dpl::minstd_rand engine(77, 100);
      oneapi::dpl::uniform_int_distribution<int> distr;

      // Cannot use "while (1)" if there is preamble before the while block (deadlock).
      while (comp_true_predicate_pipe::read()) {
        auto etan = etan_pipe_true::read();
        int wet_val = wet_pipe_true::read();

        auto w_i = long_func(engine, distr, array_size);
        auto t = 0.25 + etan * T(wet_val) / 2.0 + exp(etan);
        etan += etan + t + exp(t + etan + wet[w_i]);

        etan_pipe_out_true::write(etan);
      }
    });
  });

  auto start = e.get_profiling_info<info::event_profiling::command_start>();
  auto end = e.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}
