import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

# Keep parameters synced
from run_exp import *

plt.rcParams['font.size'] = 14
# colors = seaborn.color_palette("rocket", 3)
colors = ['#c3121e', '#0348a1', '#ffb01c', '#027608',
          '#0193b0', '#9c5300', '#949c01', '#7104b5']
# plt.figure(unique_id) ensures that different invocations of make_plot() don't interfere
fig_id = 0


def get_freq(bin):
    hw_prj = bin.replace('_sim', '') + '.prj'
    try:
        with open(f'{hw_prj}/acl_quartus_report.txt', 'r') as f:
            contents = f.read()
            m = re.search(r'Kernel fmax: (\d+)', contents)
            if m:
                return int(m.group(1))

            return 0
    except:
        return 0


def make_plot(filename, relative=True, y_label='Execution time (ms)', title=''):
    global fig_id
    global BINS_STATIC
    global BINS_DYNAMIC
    global BINS_DYNAMIC_NO_FORWARD

    fig_id = fig_id + 1

    df = pd.read_csv(filename)
    x = df['q_size']
    y = df['static'].replace({0: np.nan})

    x2 = df['q_size']
    y2 = df['dynamic'].replace({0: np.nan})
    x3 = df['q_size']
    y3 = df['dynamic_no_forward'].replace({0: np.nan})

    if relative:
        static_baseline = y[0]
        y = [1 for _ in df['static']]
        y2 = [val/static_baseline for val in y2]
        y3 = [val/static_baseline for val in y3]

    # plot
    fig = plt.figure(fig_id, figsize=(8, 8))
    plt.semilogx(x3, y3, linestyle='-', marker='^', label='dynamic (no forwarding)',
                 color=colors[2], mfc='w', markersize=8)
    plt.semilogx(x2, y2, linestyle='-', marker='s', label='dynamic (forwarding)',
                 color=colors[1], mfc='w', markersize=8)
    plt.semilogx(x, y, linestyle='-', marker='o', label='static',
                 color=colors[0], mfc='w', markersize=8)

    # Add frequencies as
    if not 'sim' in filename:
        plt.text(x[len(x) - 1], y[len(y) - 1], '283 MHz', fontsize='x-small', fontstyle='italic')

        for i in range(len(BINS_DYNAMIC)):
            freq = get_freq(BINS_DYNAMIC[i])
            if not np.isnan(y2[i]):
                plt.text(x2[i], y2[i], f'{freq} MHz', fontsize='x-small', fontstyle='italic', verticalalignment='top')

        for i in range(len(BINS_DYNAMIC_NO_FORWARD)):
            freq = get_freq(BINS_DYNAMIC_NO_FORWARD[i])
            if not np.isnan(y3[i]):
                plt.text(x3[i], y3[i], f'{freq} MHz', fontsize='x-small', fontstyle='italic', verticalalignment='bottom')

    xticks = Q_SIZES_DYNAMIC_NO_FORWARD
    plt.xticks(ticks=xticks, labels=xticks)

    plt.xlabel(r'Queue size', fontsize=14)
    plt.ylabel(y_label, fontsize=14)  # label the y axis

    # add the legend (will default to 'best' location)
    plt.legend(fontsize=14)
    plt.title(title)

    plt.savefig(filename.replace('csv', 'png'), dpi=300, bbox_inches="tight")


if __name__ == '__main__':
    is_sim = any('sim' in arg for arg in sys.argv[1:])

    SUB_DIR = 'simulation' if is_sim else 'hardware'
    BIN_EXTENSION = 'fpga_sim' if is_sim else 'fpga_emu'
    Y_LABEL = 'Cycles' if is_sim else 'Time (ms)'
    Y_LABEL += ' (normalised)'

    for kernel in KERNELS:
        BINS_STATIC = [f'{kernel}/bin/{kernel}_static.{BIN_EXTENSION}']
        BINS_DYNAMIC = [f'{kernel}/bin/{kernel}_dynamic_{s}qsize.{BIN_EXTENSION}' 
                        for s in Q_SIZES_DYNAMIC]
        BINS_DYNAMIC_NO_FORWARD = [f'{kernel}/bin/{kernel}_dynamic_no_forward_{s}qsize.{BIN_EXTENSION}' 
                                   for s in Q_SIZES_DYNAMIC_NO_FORWARD]

        A_SIZE = A_SIZES_KERNELS[kernel]


        # Ensure dir structure exists
        Path(f'{EXP_DATA_DIR}/{kernel}/{SUB_DIR}').mkdir(parents=True, exist_ok=True)
        for distr_idx, distr_name in DATA_DISTRIBUTIONS.items():
            squared = '^2' if kernel == 'spmv' else ''
            title=f'Array of {A_SIZES_KERNELS[kernel]}{squared} elements with data distribution {distr_name}'

            make_plot(f'{EXP_DATA_DIR}/{kernel}/{SUB_DIR}/{DATA_DISTRIBUTIONS[distr_idx]}_{A_SIZES_KERNELS[kernel]}.csv',
                      y_label=Y_LABEL, title=title)
