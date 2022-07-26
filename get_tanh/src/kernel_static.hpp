#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#include <sycl/ext/intel/fpga_extensions.hpp>

using namespace sycl;

class get_tanhKernel;

using PipelinedLSU = ext::intel::lsu<>;

double get_tanh_kernel(queue &q, std::vector<int> &A, const std::vector<int> addr_in, const std::vector<int> addr_out) {
  std::cout << "Static HLS\n";

  const uint array_size = A.size();

  buffer A_buf(A);
  buffer addr_in_buf(addr_in);
  buffer addr_out_buf(addr_out);

  auto event = q.submit([&](handler &hnd) {
    accessor A(A_buf, hnd, read_write);
    accessor addr_out(addr_out_buf, hnd, read_only);
    accessor addr_in(addr_in_buf, hnd, read_only);

    hnd.single_task<get_tanhKernel>([=]() [[intel::kernel_args_restrict]] {
      int atanh[12] = { 0x08C9, 0x0416, 0x0202, 0x0100, 0x0080, 0x0064, 0x0032, 0x0010, 0x0008, 0x0004, 0x0002, 0x0001 };
      int cosh[5] = { 0x1000, 0x18B0, 0x3C31, 0xA115, 0x1B4EE};
      int sinh[5] = { 0x0, 0x12CD, 0x3A07, 0xA049, 0x1B4A3};


      for (int i = 0; i < array_size; i++) {
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
          #pragma unroll
          for (int k = 1; k <= 12; k++) {
            // force the 3k+1 th iteration to be repeated
            if (((k % 3) == 1) && (k != 1)) {
              #pragma unroll
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
    });
  });

  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}
