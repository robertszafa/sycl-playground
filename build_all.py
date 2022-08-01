import sys
import os
import re
import csv
from pathlib import Path


KERNELS = [
    'histogram',
    'histogram_if',
    'spmv',
    'maximal_matching',
    'get_tanh',
]

Q_SIZES_DYNAMIC = [1, 2, 4, 8, 16, 32, 64]
Q_SIZES_DYNAMIC_NO_FORWARD = [1, 2, 4, 8, 16, 32, 64]


def build_make_string(target='fpga_sim', kernel='dynamic', q_size=2):
  return f'make {target} Q_SIZE={q_size} KERNEL={kernel}'


def run_make(kernel, make_string):
    cmd = f'cd {kernel} && {make_string} && cd ..'
    # print(cmd)
    os.system(cmd)


if __name__ == '__main__':
    is_sim = any('sim' in arg for arg in sys.argv[1:])

    TARGET = 'fpga_sim' if is_sim else 'fpga'

    for kernel in KERNELS:
        print('Building for kernel:', kernel)

        run_make(kernel, build_make_string(TARGET, kernel='static'))

        for q_size in Q_SIZES_DYNAMIC:
          run_make(kernel, build_make_string(TARGET, kernel='dynamic', q_size=q_size))

        for q_size in Q_SIZES_DYNAMIC_NO_FORWARD:
          run_make(kernel, build_make_string(TARGET, kernel='dynamic_no_forward', q_size=q_size))


