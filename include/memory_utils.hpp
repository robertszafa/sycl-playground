#ifndef __MEMORY_UTILS_HPP__
#define __MEMORY_UTILS_HPP__

#include <type_traits>

#include "metaprogramming_utils.hpp"

//
// The utilities in this file are used for converting streaming data to/from
// memory from/to a pipe.
//

namespace fpga_tools {

namespace detail {

//
// Helper to check if a SYCL pipe and pointer have the same base type
//
template <typename PipeT, typename PtrT>
struct pipe_and_pointer_have_same_base {
  using PipeBaseT =
      std::conditional_t<fpga_tools::has_subscript_v<PipeT>,
                         std::decay_t<decltype(std::declval<PipeT>()[0])>,
                         PipeT>;
  using PtrBaseT = std::decay_t<decltype(std::declval<PtrT>()[0])>;
  static constexpr bool value = std::is_same_v<PipeBaseT, PtrBaseT>;
};

template <typename PipeT, typename PtrT>
inline constexpr bool pipe_and_pointer_have_same_base_v =
    pipe_and_pointer_have_same_base<PipeT, PtrT>::value;

//
// Streams data from 'in_ptr' into 'Pipe', 'elements_per_cycle' elements at a
// time
//
template <typename Pipe, int elements_per_cycle, typename PtrT>
void MemoryToPipeRemainder(PtrT in_ptr, size_t full_count,
                           size_t remainder_count) {
  static_assert(fpga_tools::is_sycl_pipe_v<Pipe>);
  using PipeT = decltype(Pipe::read());
  static_assert(fpga_tools::has_subscript_v<PipeT>);
  static_assert(fpga_tools::has_subscript_v<PtrT>);
  static_assert(PipeT::size == elements_per_cycle);
  static_assert(pipe_and_pointer_have_same_base_v<PipeT, PtrT>);

  for (size_t i = 0; i < full_count; i++) {
    PipeT pipe_data;
#pragma unroll
    for (int j = 0; j < elements_per_cycle; j++) {
      pipe_data[j] = in_ptr[i * elements_per_cycle + j];
    }
    Pipe::write(pipe_data);
  }

  PipeT pipe_data;
  for (size_t i = 0; i < remainder_count; i++) {
    pipe_data[i] = in_ptr[full_count * elements_per_cycle + i];
  }
  Pipe::write(pipe_data);
}

//
// Streams data from 'in_ptr' into 'Pipe', 'elements_per_cycle' elements at a
// time with the guarantee that 'elements_per_cycle' is a multiple of 'count'
//
template <typename Pipe, int elements_per_cycle, typename PtrT>
void MemoryToPipeNoRemainder(PtrT in_ptr, size_t count) {
  static_assert(fpga_tools::is_sycl_pipe_v<Pipe>);
  using PipeT = decltype(Pipe::read());
  static_assert(fpga_tools::has_subscript_v<PipeT>);
  static_assert(fpga_tools::has_subscript_v<PtrT>);
  static_assert(PipeT::size == elements_per_cycle);
  static_assert(pipe_and_pointer_have_same_base_v<PipeT, PtrT>);

  for (size_t i = 0; i < count; i++) {
    PipeT pipe_data;
#pragma unroll
    for (int j = 0; j < elements_per_cycle; j++) {
      pipe_data[j] = in_ptr[i * elements_per_cycle + j];
    }
    Pipe::write(pipe_data);
  }
}

//
// Streams data from 'Pipe' to 'out_ptr', 'elements_per_cycle' elements at a
// time
//
template <typename Pipe, int elements_per_cycle, typename PtrT>
void PipeToMemoryRemainder(PtrT out_ptr, size_t full_count,
                           size_t remainder_count) {
  static_assert(fpga_tools::is_sycl_pipe_v<Pipe>);
  using PipeT = decltype(Pipe::read());
  static_assert(fpga_tools::has_subscript_v<PipeT>);
  static_assert(fpga_tools::has_subscript_v<PtrT>);
  static_assert(PipeT::size == elements_per_cycle);
  static_assert(pipe_and_pointer_have_same_base_v<PipeT, PtrT>);

  for (size_t i = 0; i < full_count; i++) {
    auto pipe_data = Pipe::read();
#pragma unroll
    for (int j = 0; j < elements_per_cycle; j++) {
      out_ptr[i * elements_per_cycle + j] = pipe_data[j];
    }
  }

  auto pipe_data = Pipe::read();
  for (size_t i = 0; i < remainder_count; i++) {
    out_ptr[full_count * elements_per_cycle + i] = pipe_data[i];
  }
}

//
// Streams data from 'Pipe' to 'out_ptr', 'elements_per_cycle' elements at a
// time with the guarantee that 'elements_per_cycle' is a multiple of 'count'
//
template <typename Pipe, int elements_per_cycle, typename PtrT>
void PipeToMemoryNoRemainder(PtrT out_ptr, size_t count) {
  static_assert(fpga_tools::is_sycl_pipe_v<Pipe>);
  using PipeT = decltype(Pipe::read());
  static_assert(fpga_tools::has_subscript_v<PipeT>);
  static_assert(fpga_tools::has_subscript_v<PtrT>);
  static_assert(PipeT::size == elements_per_cycle);
  static_assert(pipe_and_pointer_have_same_base_v<PipeT, PtrT>);

  for (size_t i = 0; i < count; i++) {
    auto pipe_data = Pipe::read();
#pragma unroll
    for (int j = 0; j < elements_per_cycle; j++) {
      out_ptr[i * elements_per_cycle + j] = pipe_data[j];
    }
  }
}

}  // namespace detail

//
// Streams data from memory to a SYCL pipe 1 element a time
//
template <typename Pipe, typename PtrT>
void MemoryToPipe(PtrT in_ptr, size_t count) {
  static_assert(fpga_tools::is_sycl_pipe_v<Pipe>);
  using PipeT = decltype(Pipe::read());
  static_assert(fpga_tools::has_subscript_v<PtrT>);
  static_assert(detail::pipe_and_pointer_have_same_base_v<PipeT, PtrT>);

  for (size_t i = 0; i < count; i++) {
    Pipe::write(in_ptr[i]);
  }
}

//
// Streams data from memory to a SYCL pipe 'elements_per_cycle' elements a time
//
template <typename Pipe, int elements_per_cycle, bool remainder, typename PtrT>
void MemoryToPipe(PtrT in_ptr, size_t count) {
  if constexpr (!remainder) {
    // user promises there is not remainder
    detail::MemoryToPipeNoRemainder<Pipe, elements_per_cycle>(in_ptr, count);
  } else {
    // might have a remainder and it was not specified, so calculate it
    auto full_count = (count / elements_per_cycle) * elements_per_cycle;
    auto remainder_count = count % elements_per_cycle;
    detail::MemoryToPipeRemainder<Pipe, elements_per_cycle>(in_ptr, full_count,
                                                            remainder_count);
  }
}

//
// Streams data from memory to a SYCL pipe 'elements_per_cycle' elements a time
// In this version, the user has specified a the amount of remainder
//
template <typename Pipe, int elements_per_cycle, bool remainder, typename PtrT>
void MemoryToPipe(PtrT in_ptr, size_t full_count, size_t remainder_count) {
  if constexpr (!remainder) {
    // user promises there is not remainder
    detail::MemoryToPipeNoRemainder<Pipe, elements_per_cycle>(in_ptr,
                                                              full_count);
  } else {
    // might have a remainder that was specified by the user
    detail::MemoryToPipeRemainder<Pipe, elements_per_cycle>(in_ptr, full_count,
                                                            remainder_count);
  }
}

//
// Streams data from a SYCL pipe to memory 1 element a time
//
template <typename Pipe, typename PtrT>
void PipeToMemory(PtrT out_ptr, size_t count) {
  using PipeT = decltype(Pipe::read());
  static_assert(fpga_tools::has_subscript_v<PtrT>);
  static_assert(detail::pipe_and_pointer_have_same_base_v<PipeT, PtrT>);

  for (size_t i = 0; i < count; i++) {
    out_ptr[i] = Pipe::read();
  }
}

//
// Streams data from a SYCL pipe to memory 'elements_per_cycle' elements a time
//
template <typename Pipe, int elements_per_cycle, bool remainder, typename PtrT>
void PipeToMemory(PtrT out_ptr, size_t count) {
  if constexpr (!remainder) {
    detail::PipeToMemoryNoRemainder<Pipe, elements_per_cycle>(out_ptr, count);
  } else {
    auto full_count = (count / elements_per_cycle) * elements_per_cycle;
    auto remainder_count = count % elements_per_cycle;
    detail::PipeToMemoryRemainder<Pipe, elements_per_cycle>(out_ptr, full_count,
                                                            remainder_count);
  }
}

//
// Streams data from a SYCL pipe to memory 'elements_per_cycle' elements a time
// In this version, the user has specified a the amount of remainder
//
template <typename Pipe, int elements_per_cycle, bool remainder, typename PtrT>
void PipeToMemory(PtrT out_ptr, size_t full_count, size_t remainder_count) {
  if constexpr (!remainder) {
    detail::PipeToMemoryNoRemainder<Pipe, elements_per_cycle>(out_ptr,
                                                              full_count);
  } else {
    detail::PipeToMemoryRemainder<Pipe, elements_per_cycle>(out_ptr, full_count,
                                                            remainder_count);
  }
}

/// 1. Allocate device_memory (same num bytes as in host_vector)
/// 2. Transfer data host_vector->device_memory
/// 3. Return device_ptr to device_memory
template<typename T>
device_ptr<T> toDevice(const std::vector<T> &host_vector, queue &q) {
  T* device_data = malloc_device<T>(host_vector.size(), q);
  q.copy(host_vector.data(), device_data, host_vector.size()).wait();
  return device_ptr<T>(device_data);
}
template<typename T, int N>
device_ptr<T> toDevice(const T* host_array[N], queue &q) {
  T* device_data = malloc_device<T>(N, q);
  q.copy(host_array, device_data, N).wait();
  return device_ptr<T>(device_data);
}
template<typename T>
device_ptr<T> toDevice(const T* host_array, const int N, queue &q) {
  T* device_data = malloc_device<T>(N, q);
  q.copy(host_array, device_data, N).wait();
  return device_ptr<T>(device_data);
}

}  // namespace fpga_tools

#endif /* __MEMORY_UTILS_HPP__ */