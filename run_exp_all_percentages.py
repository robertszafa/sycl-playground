import sys
import os
import re
import csv
from pathlib import Path
from run_exp import (EXP_DATA_DIR, KERNEL_ASIZE_PAIRS, KERNEL_ASIZE_PAIRS_SIM, 
                     SIM_CYCLES_FILE, TMP_FILE)


# Only run with the best Q_SIZE, otherwise there will be a lot of runs.
BEST_Q_SIZES_DYNAMIC = {
    'histogram' : 16,
    'histogram_if' : 16,
    'spmv' : 16,
    'maximal_matching' : 4,
    'get_tanh' : 2,
}
BEST_Q_SIZES_DYNAMIC_NO_FORWARD  = {
    'histogram' : 32,   # 64 is actually better
    'histogram_if' : 32,
    'spmv' : 32,
    'maximal_matching' : 4,
    'get_tanh' : 4,
}

DATA_DISTRIBUTION_KEY = 2
DATA_DISTRIBUTION_NAME = 'percentage_wait'

PERCENTAGES_WAIT = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

CSV_PERCENTAGES_RES_FILE = f'{DATA_DISTRIBUTION_NAME}.csv'


def run_bin(bin, a_size, distr=2, percentage=0):
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
            print(f'{float(match.group(1))}')
            return float(match.group(1))


if __name__ == '__main__':
    is_sim = any('sim' in arg for arg in sys.argv[1:])

    BIN_EXTENSION = 'fpga_sim' if is_sim else 'fpga'
    SUB_DIR = 'simulation' if is_sim else 'hardware'
    KERNEL_ASIZE_PAIRS = KERNEL_ASIZE_PAIRS_SIM if is_sim else KERNEL_ASIZE_PAIRS

    for kernel, a_size in KERNEL_ASIZE_PAIRS.items():
        print('Running kernel:', kernel)

        BIN_STATIC = f'{kernel}/bin/{kernel}_static.{BIN_EXTENSION}'
        BIN_DYNAMIC = f'{kernel}/bin/{kernel}_dynamic_{BEST_Q_SIZES_DYNAMIC[kernel]}qsize.{BIN_EXTENSION}' 
        BIN_DYNAMIC_NO_FORWARD = f'{kernel}/bin/{kernel}_dynamic_no_forward_{BEST_Q_SIZES_DYNAMIC_NO_FORWARD[kernel]}qsize.{BIN_EXTENSION}' 

        # Ensure dir structure exists
        Path(f'{EXP_DATA_DIR}/{kernel}/{SUB_DIR}').mkdir(parents=True, exist_ok=True)

        with open(f'{EXP_DATA_DIR}/{kernel}/{SUB_DIR}/{CSV_PERCENTAGES_RES_FILE}', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['percentage', 'static', 
                                f'dynamic (q_size {BEST_Q_SIZES_DYNAMIC[kernel]})', 
                                f'dynamic_no_forward (q_size {BEST_Q_SIZES_DYNAMIC_NO_FORWARD[kernel]})'])

            for percentage in PERCENTAGES_WAIT:
                print(f'Running with percentage_wait {percentage} %')

                static_time = run_bin(BIN_STATIC, a_size, percentage=percentage)
                dyn_time = run_bin(BIN_DYNAMIC, a_size, percentage=percentage)
                dyn_no_forward_time = run_bin(BIN_DYNAMIC_NO_FORWARD, a_size, percentage=percentage)

                new_row = []
                new_row.append(percentage)
                new_row.append(static_time)
                new_row.append(dyn_time)
                new_row.append(dyn_no_forward_time)

                writer.writerow(new_row)


    os.system(f'rm {TMP_FILE}')


