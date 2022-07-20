import sys
import os
import re
import csv
from pathlib import Path
from run_exp import (EXP_DATA_DIR, KERNEL_ASIZE_PAIRS, KERNEL_ASIZE_PAIRS_SIM, 
                     SIM_CYCLES_FILE, TMP_FILE)


Q_SIZE_DYNAMIC = 16
Q_SIZE_DYNAMIC_NO_FORWARD = 32

PERCENTAGES_WAIT = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

CSV_PERCENTAGES_RES_FILE = 'percentage_wait_all.csv'


def run_bin(bin, a_size, distr=2, percentage=0):
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

        BIN_STATIC = f'{kernel}/bin/{kernel}_static.{BIN_EXTENSION}'
        BIN_DYNAMIC = f'{kernel}/bin/{kernel}_dynamic_{Q_SIZE_DYNAMIC}qsize.{BIN_EXTENSION}' 
        BIN_DYNAMIC_NO_FORWARD = f'{kernel}/bin/{kernel}_dynamic_no_forward_{Q_SIZE_DYNAMIC_NO_FORWARD}qsize.{BIN_EXTENSION}' 

        # Ensure dir structure exists
        Path(f'{EXP_DATA_DIR}/{kernel}/{SUB_DIR}').mkdir(parents=True, exist_ok=True)

        with open(f'{EXP_DATA_DIR}/{kernel}/{SUB_DIR}/{CSV_PERCENTAGES_RES_FILE}', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['percentage', 'static', 
                                f'dynamic (q_size {Q_SIZE_DYNAMIC})', 
                                f'dynamic_no_forward (q_size {Q_SIZE_DYNAMIC_NO_FORWARD})'])

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


