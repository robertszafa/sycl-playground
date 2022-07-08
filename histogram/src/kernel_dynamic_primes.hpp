/* 
  A sketch of an idea to use prime factorisation to detect which addresses are currently written to.
  Each array index is mapped to a unique prime number. The store_queue has a single COMPOSITE_PRIME
  that is a factor of primes/indices currently being in flight to storage. 
  
  When a new store arrives, the prime number of that new store_index is multiplied with the 
  COMPOSITE_PRIME.
  
  Once the store finishes (e.g. a number of cycles has passed), the COMPOSITE_PRIME is divided by
  that store_index.
  
  When a load request arrives with an index/prime pair, we can check if the prime divides the 
  COMPOSITE_PRIME. If it does, then there is an in-flight store the associated index, and we know
  that we have to wait.


  Altough mathematically pleasing, this idea is not practical because the primes become too large 
  to quickly. One could decrease the maximum number of in-flight stores to n such that 
  the COMPOSITE_PRIME is only a factor of n primes, but even this still results in large numbers.
*/


#include "CL/sycl/access/access.hpp"
#include "CL/sycl/builtins.hpp"
#include "CL/sycl/properties/accessor_properties.hpp"
#include <CL/sycl.hpp>
#include <iostream>
#include <vector>

#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>

#include "tables.hpp"


using namespace sycl;

// The default PipelinedLSU will start a load/store immediately, which the memory disambiguation 
// logic relies upon.
// A BurstCoalescedLSU would instead of waiting for more requests to arrive for a coalesced access.
using PipelinedLSU = ext::intel::lsu<>;

#ifndef Q_SIZE
  #define Q_SIZE 8
#endif

constexpr uint STORE_Q_SIZE = Q_SIZE;
constexpr uint STORE_LATENCY = 16; // This should be gotten from static analysis.


struct store_entry {
  int idx; // This should be the full address in a real impl.
  uint val;
  bool executed;
  int countdown;
  int tag;
};

constexpr store_entry INVALID_ENTRY = {-1, 0, false, -1, -1};

struct pair {
  uint fst; 
  uint snd; 
};
struct triple {
  uint fst; 
  uint snd; 
  uint thrd; 
};

struct candidate {
  int tag; 
  bool forward;
};


double histogram_kernel(queue &q, const std::vector<uint> &feature, const std::vector<uint> &weight,
                        std::vector<uint> &hist) {
  std::cout << "Dynamic HLS\n";

  const uint array_size = feature.size();

  buffer feature_buf(feature);
  buffer weight_buf(weight);
  buffer hist_buf(hist);

  buffer primes_buf(primes, range<1>(1398));

  constexpr uint PIPE_D = 64;
  using weight_load_pipe = pipe<class weight_load_pipe_class, uint, PIPE_D>;
  using idx_load_pipe = pipe<class feature_load_pipe_class, pair, PIPE_D>;
  using idx_store_pipe = pipe<class feature_store_pipe_class, uint, PIPE_D>;
  using val_load_pipe = pipe<class hist_load_pipe_class, uint, PIPE_D>;
  using store_pipe = pipe<class hist_store_pipe_class, triple, PIPE_D>;

  using store_ack_pipe_in = pipe<class store_ack_pipe_in_class, uint, PIPE_D>;
  using store_ack_pipe_out = pipe<class store_ack_pipe_out_class, uint, PIPE_D>;
  
  using pred_ack_pipe = pipe<class pred_ack_pipe_class, bool, PIPE_D>;

  q.submit([&](handler &hnd) {
    hnd.single_task<class StoreAck>([=]() [[intel::kernel_args_restrict]] {
      while (pred_ack_pipe::read()) {
        auto in = store_ack_pipe_in::read();
        // Todo latency anchor
        store_ack_pipe_out::write(in);
      }
      // ext::oneapi::experimental::printf("Done ack\n");
    });
  });
 

  q.submit([&](handler &hnd) {
    accessor hist(hist_buf, hnd, read_write);

    hnd.single_task<class LoadStoreHist>([=]() [[intel::kernel_args_restrict]] {
      uint64_t COMPOSITE_PRIME = 1;

      int i_store_val = 0;
      int i_store_idx = 0;
      int i_load = 0;

      uint idx_load, idx_store, prime_store, prime_load, val_load, val_store;
      int store_idx_fifo_head = 0;

      bool val_load_pipe_write_succ = true;
      bool is_load_waiting_for_val = false;

      [[intel::ivdep]] 
      while (i_store_val < array_size) {

        if (val_load_pipe_write_succ) {
          bool load_pipe_succ = false;
          if (!is_load_waiting_for_val) {
            auto load_pair = idx_load_pipe::read(load_pipe_succ);
            idx_load = load_pair.fst;
            prime_load = load_pair.snd;
          }

          if (load_pipe_succ || is_load_waiting_for_val) {
            auto prime_load_long = uint64_t(prime_load);

            if ((COMPOSITE_PRIME / prime_load_long) * prime_load_long == COMPOSITE_PRIME) {
              is_load_waiting_for_val = true;
            }
            else {
              val_load = PipelinedLSU::load(hist.get_pointer() + idx_load);
              val_load_pipe_write_succ = false;
              is_load_waiting_for_val = false;
            }
          }
        }
        if (!val_load_pipe_write_succ) {
          val_load_pipe::write(val_load, val_load_pipe_write_succ);
        }

        bool store_pipe_ack_succ = false;
        auto ack_prime = store_ack_pipe_out::read(store_pipe_ack_succ);
        if (store_pipe_ack_succ) {
          COMPOSITE_PRIME = COMPOSITE_PRIME / uint64_t(ack_prime);
          i_store_val++;
        }

        bool store_pipe_succ = false;
        auto store_triple = store_pipe::read(store_pipe_succ);
        if (store_pipe_succ) {
          idx_store = store_triple.fst;
          val_store = store_triple.snd;
          prime_store = store_triple.thrd;
          COMPOSITE_PRIME = COMPOSITE_PRIME * uint64_t(prime_store);
          PipelinedLSU::store(hist.get_pointer() + idx_store, val_store);
          pred_ack_pipe::write(1);
          store_ack_pipe_in::write(prime_store);
        }
      }

      // ext::oneapi::experimental::printf("Done LSQ\n");
      pred_ack_pipe::write(0);
    });
  });

  auto event = q.submit([&](handler &hnd) {
    accessor feature(feature_buf, hnd, read_only);
    accessor weight(weight_buf, hnd, read_only);
    accessor primes(primes_buf, hnd, read_only);

    hnd.single_task<class Compute>([=]() [[intel::kernel_args_restrict]] {
      for (int i = 0; i < array_size; ++i) {
        uint wt = weight[i];
        uint idx = feature[i];

        idx_load_pipe::write({idx, primes[idx]});
        uint hist = val_load_pipe::read();

        auto new_hist = hist + wt;
        store_pipe::write({idx, new_hist, primes[idx]});
      }
        // ext::oneapi::experimental::printf("Done calc\n");
    });
  });



  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}

