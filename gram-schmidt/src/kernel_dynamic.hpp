#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

class Compute;

template <typename T>
double gram_schmidt_kernel(queue &q, std::vector<T> &a, std::vector<T> &r, const uint N,
                           const uint M) {
  std::cout << "Static HLS\n";

  buffer<T, 2> a_buf(a.data(), range<2>{N, M});
  buffer<T, 2> r_buf(r.data(), range<2>{N, M});

  event e = q.submit([&](handler &hnd) {
    accessor a(a_buf, hnd, read_write);
    accessor r(r_buf, hnd, read_write);

    // From Benchmarks for High-Level Synthesis (Jianyi Cheng)
    hnd.single_task<Compute>([=]() [[intel::kernel_args_restrict]] {
      float tol = 0.1f;
      for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        for (int j = 0; j < M; j++)
          sum += a[i][j] * a[i][j];
        // sum = sqrt(sum);
        sum -= 4.0f;
        sum = 0.0019f * ((sum - 8.0f) * sum + 16.0f) * sum + 2.0f;
        r[i][i] = sum;

        if (sum > tol) { // a_i = a_i/r_ii
          for (int j = 0; j < N; j++)
            a[i][j] = a[i][j] / sum;
        } else if (i == 0) { // set a[0] = [1 0 0 ... 0]^T
          for (int j = 0; j < N; j++)
            a[i][j] = (j == 0) ? 1.0f : 0.0f;
        } else { // need to choose a_i orthogonal to < a_1, ... a_{i-1} >
          for (int j = 0; j < N; j++)
            a[i][j] = -a[0][i] * a[0][j];

          a[i][i] += 1.0f;
          for (int j = 1; j < N; j++) {
            float d = a[j][i];
            for (int k = 0; k < N; k++)
              a[i][k] -= a[j][k] * d;
          }
          float anorm = 0.0f;
          for (int j = 0; j < N; j++)
            anorm += a[i][j] * a[i][j];
          for (int j = 0; j < N; j++)
            a[0][i] = a[0][i] / anorm;
        }

        for (int j = i + 1; j < N; j++) {
          float sum = 0.0f;
          for (int k = 0; k < N; k++)
            sum += a[i][k] * a[j][k]; // r_ij = a_i*a_j
          for (int k = 0; k < N; k++)
            a[j][k] -= a[i][k] * sum; // a_j -= r_ij a_i
          r[j][i] = sum;
        }
      }
    });
  });

  auto start = e.get_profiling_info<info::event_profiling::command_start>();
  auto end = e.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}
