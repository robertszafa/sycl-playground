import sys
import os
import re
import csv
import time
from pathlib import Path
from build_all import Q_SIZES_DYNAMIC, KERNELS


EXP_DATA_DIR = 'exp_data/'
# The kernels that will actually be run are in build_all.KERNELS
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
    'maximal_matching' : 1000,
    'get_tanh' : 1000,
}
DATA_DISTRIBUTIONS = {
    0: 'all_wait',
    1: 'no_wait',
}
SIM_CYCLES_FILE = 'simulation_raw.json'
TMP_FILE = f'.tmp_run_exp{str(time.time())[-5:]}.txt'


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
        if match:
            if not  not 'emu' in bin:
                print(f'{float(match.group(1))}')
            return float(match.group(1))


if __name__ == '__main__':
    if sys.argv[1] not in ['emu', 'sim', 'hw']:
        exit("ERROR: No extension provided\nUSAGE: ./build_all.py [emu, sim, hw]\n")

    is_sim = 'sim' == sys.argv[1]
    is_emu = 'emu' == sys.argv[1]
    bin_ext = 'fpga_' + sys.argv[1]
    SUB_DIR = sys.argv[1]
    if is_sim or is_emu:
        KERNEL_ASIZE_PAIRS = KERNEL_ASIZE_PAIRS_SIM

    for kernel in KERNELS:
        if not kernel in KERNEL_ASIZE_PAIRS.keys():
            exit(f"ERROR: {kernel} not in KERNEL_ASIZE_PAIRS")
        a_size = KERNEL_ASIZE_PAIRS[kernel]

        print('\n--Running kernel:', kernel)
        for distr_idx, distr_name in DATA_DISTRIBUTIONS.items():
            print('\n--Running with data distribution:', distr_name)

            # Ensure dir structure exists
            Path(f'{EXP_DATA_DIR}/{kernel}/{SUB_DIR}').mkdir(parents=True, exist_ok=True)

            with open(f'{EXP_DATA_DIR}/{kernel}/{SUB_DIR}/{distr_name}.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['q_size', 'static', 'dynamic'])
                
                static_time = 0
                dyn_time = 0

                static_time = run_bin(f'{kernel}/bin/{kernel}_static.fpga', 
                                      a_size, distr=distr_idx)

                for i, q_size in enumerate(Q_SIZES_DYNAMIC):
                    dyn_time = run_bin(f'{kernel}/bin/{kernel}_dynamic_{q_size}qsize.{bin_ext}', 
                                        a_size, distr=distr_idx)

                    new_row = []
                    new_row.append(q_size)
                    new_row.append(static_time)
                    new_row.append(dyn_time)
                    writer.writerow(new_row)
            
            # Emulation times are meaningless.
            if is_emu:
                os.system(f'rm -r {EXP_DATA_DIR}/{kernel}/{SUB_DIR}')
        

    os.system(f'rm {TMP_FILE}')
