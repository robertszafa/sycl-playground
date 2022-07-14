import sys
import os
import re
import csv
from pathlib import Path


EXP_DATA_DIR = 'exp_data/'

KERNEL_ASIZE_PAIRS = {
    'histogram' : 1000000,
    'histogram_if' : 1000000,
    'spmv' : 400,
}
# Decrease domain sizes when running in simulation.
KERNEL_ASIZE_PAIRS_SIM = {
    'histogram' : 1000,
    'histogram_if' : 1000,
    'spmv' : 20,
}

DATA_DISTRIBUTIONS = {
    0: 'forwarding_friendly',
    1: 'no_dependencies',
    2: 'random'
}

Q_SIZES_DYNAMIC = [1, 2, 4, 8]
Q_SIZES_DYNAMIC_NO_FORWARD = [1, 2, 4, 8, 16, 32, 64]

SIM_CYCLES_FILE = 'simulation_raw.json'
TMP_FILE = '.tmp_run_exp.txt'

# BRAM_STATIC_PARTITION = 492
# ALMS_STATIC_PARTITION = 89975   # In report this is 'Logic utilization'
# REGISTERS_STATIC_PARTITION = 98940
# DSP_STATIC_PARTITION = 0


def run_bin(bin, a_size, distr=0):
    os.system(f'{bin} {a_size} {distr} > {TMP_FILE}')

    stdout = ''
    with open(TMP_FILE, 'r') as f:
        stdout = str(f.read())
    if not 'Passed' in stdout:
        print(f' - Fail in {bin} {a_size} {distr}')
    
    if 'fpga_sim' in bin: 
        # Get cycle count
        with open(SIM_CYCLES_FILE, 'r') as f:
            match = re.search(r'"time":"(\d+)"', f.read())
        if (match):
            return int(match.group(1))
    else: 
        # Get time
        match = re.search(r'Kernel time \(ms\): (\d+\.\d+|\d+)', stdout)
        if (match):
            return float(match.group(1))


if __name__ == '__main__':
    is_sim = any('sim' in arg for arg in sys.argv[1:])

    BIN_EXTENSION = 'fpga_sim' if is_sim else 'fpga'
    SUB_DIR = 'simulation' if is_sim else 'hardware'
    KERNEL_ASIZE_PAIRS = KERNEL_ASIZE_PAIRS_SIM if is_sim else KERNEL_ASIZE_PAIRS

    for kernel, a_size in KERNEL_ASIZE_PAIRS.items():
        print('Running kernel:', kernel)

        BINS_STATIC = [f'{kernel}/bin/{kernel}_static.{BIN_EXTENSION}']
        BINS_DYNAMIC = [f'{kernel}/bin/{kernel}_dynamic_{s}qsize.{BIN_EXTENSION}' 
                        for s in Q_SIZES_DYNAMIC]
        BINS_DYNAMIC_NO_FORWARD = [f'{kernel}/bin/{kernel}_dynamic_no_forward_{s}qsize.{BIN_EXTENSION}' 
                                   for s in Q_SIZES_DYNAMIC_NO_FORWARD]

        for distr_idx, distr_name in DATA_DISTRIBUTIONS.items():
            print('Running with data distribution:', distr_name)

            # Ensure dir structure exists
            Path(f'{EXP_DATA_DIR}/{kernel}/{SUB_DIR}').mkdir(parents=True, exist_ok=True)

            with open(f'{EXP_DATA_DIR}/{kernel}/{SUB_DIR}/{distr_name}_{a_size}.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['q_size', 'static', 'dynamic', 'dynamic_no_forward'])

                static_time = run_bin(BINS_STATIC[0], a_size, distr=distr_idx)

                for i, q_size in enumerate(Q_SIZES_DYNAMIC_NO_FORWARD):
                    dyn_time = 0
                    dyn_no_forward_time = 0
                    if len(BINS_DYNAMIC) > i:
                        dyn_time = run_bin(BINS_DYNAMIC[i], a_size, distr=distr_idx)
                    if len(BINS_DYNAMIC_NO_FORWARD) > i:
                        dyn_no_forward_time = run_bin(BINS_DYNAMIC_NO_FORWARD[i], a_size, distr=distr_idx)

                    new_row = []
                    new_row.append(q_size)
                    new_row.append(static_time)
                    new_row.append(dyn_time)
                    new_row.append(dyn_no_forward_time)
                    writer.writerow(new_row)

    os.system(f'rm {TMP_FILE}')


