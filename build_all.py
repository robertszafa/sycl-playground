import sys
import os


KERNELS = [
    'histogram',
    'histogram_if',
    'spmv',
    'maximal_matching',
    'get_tanh',
]

Q_SIZES_DYNAMIC = [2, 4, 8, 16]

def build_make_string(target='fpga_sim', kernel='dynamic', q_size=2):
  return f'make {target} Q_SIZE={q_size} KERNEL={kernel}'


def run_make(kernel, make_string):
    cmd = f'cd {kernel} && {make_string} && cd ..'
    # print(cmd)
    os.system(cmd)


if __name__ == '__main__':
    bin_ext = sys.argv[1]
    if bin_ext not in ['emu', 'sim', 'hw']:
        exit("ERROR: No extension provided\nUSAGE: ./build_all.py [emu, sim, hw]\n")
    TARGET = 'fpga_' + bin_ext

    for kernel in KERNELS:
        print('Building for kernel:', kernel)

        run_make(kernel, build_make_string(TARGET, kernel='static'))

        for q_size in Q_SIZES_DYNAMIC:
          run_make(kernel, build_make_string(TARGET, kernel='dynamic', q_size=q_size))
