import os
import sys
import re
import subprocess
import csv


RUN_DATA_DIR = 'run_data/'

BIN_EXTENSION = 'fpga_sim'
Q_SIZES_DYNAMIC = [1, 2, 4, 8]
Q_SIZES_DYNAMIC_NO_FORWARD = [1, 2, 4, 8, 16, 32, 64]
BINS_STATIC = [f'bin/histogram_static.{BIN_EXTENSION}']
BINS_DYNAMIC = [f'bin/histogram_dynamic_{s}qsize.{BIN_EXTENSION}' for s in Q_SIZES_DYNAMIC]
BINS_DYNAMIC_NO_FORWARD = [
    f'bin/histogram_dynamic_no_forward_{s}qsize.{BIN_EXTENSION}' for s in Q_SIZES_DYNAMIC_NO_FORWARD]

SIM_CYCLES_FILE = 'simulation_raw.json'

DATA_DISTRIBUTIONS = {
    0: 'forwarding_friendly',
    1: 'no_dependencies',
    2: 'random'
}

A_SIZE = 12

BRAM_STATIC_PARTITION = 492
ALMS_STATIC_PARTITION = 89975   # In report this is 'Logic utilization'
REGISTERS_STATIC_PARTITION = 98940
DSP_STATIC_PARTITION = 0


def run_bin(bin, a_size=A_SIZE, distr=0):

    result = 0
    if 'sim' in bin: 
        stdout = os.system(f'{bin} {a_size} {distr} > /dev/null')
        # Get cycle count
        with open(SIM_CYCLES_FILE, 'r') as f:
            match = re.search(r'"time":"(\d+)"', f.read())
        if (match):
            result = int(match.group(1)) / 1000 
    else: 
        stdout = str(subprocess.check_output([str(bin), str(a_size), str(distr)]))
        # Get time
        match = re.search(r'Kernel time \(ms\): (\d+\.\d+|\d+)', stdout)
        if (match):
            result = float(match.group(1))

    return result


if __name__ == '__main__':

    SUB_DIR = 'simulation' if 'sim' in BIN_EXTENSION else 'hardware'

    for k, v in DATA_DISTRIBUTIONS.items():
        with open(f'{RUN_DATA_DIR}/{SUB_DIR}/{v}.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['q_size', 'static', 'dynamic', 'dynamic_no_forward'])

            static_time = run_bin(BINS_STATIC[0], distr=k)

            for i, q_size in enumerate(Q_SIZES_DYNAMIC_NO_FORWARD):
                print('Running q_size ', q_size)

                dyn_time = 0
                dyn_no_forward_time = 0
                if len(BINS_DYNAMIC) > i:
                    dyn_time = run_bin(BINS_DYNAMIC[i], distr=k)
                if len(BINS_DYNAMIC_NO_FORWARD) > i:
                    dyn_no_forward_time = run_bin(BINS_DYNAMIC_NO_FORWARD[i], distr=k)

                new_row = []
                new_row.append(q_size)
                new_row.append(static_time)
                new_row.append(dyn_time)
                new_row.append(dyn_no_forward_time)
                writer.writerow(new_row)


        print('Finished distr ', v)

