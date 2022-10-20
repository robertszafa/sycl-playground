import sys
import os


KERNELS = [
    'spmv',
    'histogram',
    'histogram_if',
    'maximal_matching',
    'get_tanh',
]

Q_SIZES = [2, 4, 8, 16]


def build_make_string(target='fpga_sim', kernel='dynamic', q_size=2):
  return f'make {target} Q_SIZE={q_size} KERNEL={kernel}'

def run_make(kernel, make_string):
    cmd = f'cd {kernel} && {make_string} && cd ..'
    os.system(cmd)


if __name__ == '__main__':
    bin_ext = sys.argv[1]
    if bin_ext not in ['emu', 'sim', 'hw']:
        exit("ERROR: No extension provided\nUSAGE: ./build_all.py [emu, sim, hw]\n")
    target = 'fpga_' + bin_ext

    for kernel in KERNELS:
        print('Building for kernel:', kernel)

        run_make(kernel, build_make_string(target, kernel='static'))

        for q_size in Q_SIZES:
          run_make(kernel, build_make_string(target, kernel='dynamic', q_size=q_size))
