import csv
import os
import re
from pathlib import Path
from subprocess import Popen, PIPE, STDOUT
from time import sleep, time

# Keep parameters synced
from run_exp import EXP_DATA_DIR, Q_SIZES_DYNAMIC, Q_SIZES_DYNAMIC_NO_FORWARD

POWER_MEASUREMENT_RESOLUTION_SECS = 0.1

FPGA_POWER_CMD = 'fpgainfo power'

# Increase sizes to get a >1 min runtime for more accurate power measurement.
KERNEL_ASIZE_PAIRS = {
    'histogram' : 400000000,
    'histogram_if' : 400000000,
    'spmv' : 16000,
    'maximal_matching' : 400000000,
}

def get_fpga_watts_now():
    out = os.popen(FPGA_POWER_CMD).read()
    watts_reg = re.findall(r"Total Input Power\s*:\s*([\d\.]+) Watts", out)
    return float(watts_reg[0])

def run_bin_power(bin, asize):
    p_bin = Popen([bin, f'{asize}'], stdout=PIPE) 

    times_measured = 0
    watts_total = 0
    start = time()
    while p_bin.poll() is None:
        watts_total += get_fpga_watts_now()
        times_measured += 1
        sleep(POWER_MEASUREMENT_RESOLUTION_SECS)
    end = time()
    print(f'Ran {bin} for {int(end - start)} s')

    avg_watts = float(watts_total) / float(times_measured)
    return avg_watts


if __name__ == "__main__":
    SUB_DIR = 'hardware'

    for kernel, a_size in KERNEL_ASIZE_PAIRS.items():
        print('Running kernel:', kernel)

        BINS_STATIC = [f'{kernel}/bin/{kernel}_static.fpga']
        BINS_DYNAMIC = [f'{kernel}/bin/{kernel}_dynamic_{s}qsize.fpga' for s in Q_SIZES_DYNAMIC]
        BINS_DYNAMIC_NO_FORWARD = [f'{kernel}/bin/{kernel}_dynamic_no_forward_{s}qsize.fpga' 
                                   for s in Q_SIZES_DYNAMIC_NO_FORWARD]

        # Ensure dir structure exists
        Path(f'{EXP_DATA_DIR}/{kernel}/{SUB_DIR}').mkdir(parents=True, exist_ok=True)

        with open(f'{EXP_DATA_DIR}/{kernel}/{SUB_DIR}/power.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['q_size', 'static', 'dynamic', 'dynamic_no_forward'])

            static_power = run_bin_power(BINS_STATIC[0], a_size)

            for i, q_size in enumerate(Q_SIZES_DYNAMIC_NO_FORWARD):
                dyn_power = 0
                dyn_no_forward_power = 0
                if len(BINS_DYNAMIC) > i:
                    dyn_power = run_bin_power(BINS_DYNAMIC[i], a_size)
                if len(BINS_DYNAMIC_NO_FORWARD) > i:
                    dyn_no_forward_power = run_bin_power(BINS_DYNAMIC_NO_FORWARD[i], a_size)

                new_row = []
                new_row.append(q_size)
                new_row.append(static_power)
                new_row.append(dyn_power)
                new_row.append(dyn_no_forward_power)
                writer.writerow(new_row)


