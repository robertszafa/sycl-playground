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
    'maximal_matching' : 1000000,
    'get_tanh' : 1000000,
}
# Decrease domain sizes when running in simulation.
KERNEL_ASIZE_PAIRS_SIM = {
    'histogram' : 1000,
    'histogram_if' : 1000,
    'spmv' : 20,
    # 'maximal_matching' : 1000,
    # 'get_tanh' : 1000,
}

DATA_DISTRIBUTIONS = {
    0: 'all_wait',
    1: 'no_wait',
    # 2: 'percentage_wait' # run_exp_all_percentages runs for different % of data hazards.
}

Q_SIZES_DYNAMIC = [2, 4, 8, 16]
Q_SIZES_DYNAMIC_NO_FORWARD = [2, 4, 8, 16, 32, 64]

PERCENTAGE_WAIT = 5

SIM_CYCLES_FILE = 'simulation_raw.json'
TMP_FILE = '.tmp_run_exp.txt'


def run_bin(bin, a_size, distr=0, percentage=0):
    print(f'> {bin} : ', end='')
    os.system(f'{bin} {a_size} {distr} {percentage} > {TMP_FILE}')

    stdout = ''
    with open(TMP_FILE, 'r') as f:
        stdout = str(f.read())
    if not 'Passed' in stdout:
        print(f' - Fail in {bin} {a_size} {distr} {percentage}')
    
    if 'fpga_sim' in bin: 
        # Get cycle count
        with open(SIM_CYCLES_FILE, 'r') as f:
            match = re.search(r'"time":"(\d+)"', f.read())
        if (match):
            print(f'{int(match.group(1))}')
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

            percentage_suffix = ''
            if distr_idx == 2:
                percentage_suffix = f'_{PERCENTAGE_WAIT}'
            
            with open(f'{EXP_DATA_DIR}/{kernel}/{SUB_DIR}/{distr_name}{percentage_suffix}.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['q_size', 'static', 'dynamic', 'dynamic_no_forward'])

                static_time = run_bin(BINS_STATIC[0], a_size, distr=distr_idx, percentage=PERCENTAGE_WAIT)

                for i, q_size in enumerate(Q_SIZES_DYNAMIC_NO_FORWARD):
                    dyn_time = 0
                    dyn_no_forward_time = 0
                    if len(BINS_DYNAMIC) > i:
                        dyn_time = run_bin(BINS_DYNAMIC[i], a_size, distr=distr_idx, percentage=PERCENTAGE_WAIT)
                    if len(BINS_DYNAMIC_NO_FORWARD) > i:
                        dyn_no_forward_time = run_bin(BINS_DYNAMIC_NO_FORWARD[i], a_size, distr=distr_idx, percentage=PERCENTAGE_WAIT)

                    new_row = []
                    new_row.append(q_size)
                    new_row.append(static_time)
                    new_row.append(dyn_time)
                    new_row.append(dyn_no_forward_time)
                    writer.writerow(new_row)
        

    os.system(f'rm {TMP_FILE}')


