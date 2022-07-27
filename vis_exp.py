import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Keep parameters synced
from run_exp import (EXP_DATA_DIR, Q_SIZES_DYNAMIC, Q_SIZES_DYNAMIC_NO_FORWARD, DATA_DISTRIBUTIONS,
                     KERNEL_ASIZE_PAIRS, KERNEL_ASIZE_PAIRS_SIM, PERCENTAGE_WAIT)
from run_exp_all_percentages import (CSV_PERCENTAGES_RES_FILE, PERCENTAGES_WAIT, Q_SIZE_DYNAMIC, 
                                     Q_SIZE_DYNAMIC_NO_FORWARD)

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

def get_ii(bin):
    hw_prj = bin.replace('_sim', '') + '.prj'
    try:
        with open(f'{hw_prj}/reports/lib/json/loops.json', 'r') as f:
            contents = f.read()
            m = re.findall(r'\["Yes", "~?(\d+)"', contents)
            max = 1
            for s in m:
                max = max if max > int(s) else int(s)
            
            return max
    except:
        print(bin, ' no ii')
        return 1

def make_plot(filename, relative=True, y_label='Speedup (normalised)', title=''):
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
        y2 = [static_baseline/val for val in y2]
        y3 = [static_baseline/val for val in y3]

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
        freq = get_freq(BINS_STATIC[0])
        ii = get_ii(BINS_STATIC[0])
        plt.text(x[len(x) - 1], y[len(y) - 1], f'{freq} MHz\nII={ii}', fontsize='x-small', fontstyle='italic')

        for i in range(len(BINS_DYNAMIC)):
            freq = get_freq(BINS_DYNAMIC[i])
            ii = get_ii(BINS_DYNAMIC[i])
            if not np.isnan(y2[i]):
                plt.text(x2[i], y2[i], f'{freq} MHz\nII={ii}', fontsize='x-small', fontstyle='italic')

        for i in range(len(BINS_DYNAMIC_NO_FORWARD)):
            freq = get_freq(BINS_DYNAMIC_NO_FORWARD[i])
            ii = get_ii(BINS_DYNAMIC_NO_FORWARD[i])
            if not np.isnan(y3[i]):
                plt.text(x3[i], y3[i], f'{freq} MHz\nII={ii}', fontsize='x-small', fontstyle='italic') 

    xticks = Q_SIZES_DYNAMIC_NO_FORWARD
    plt.xticks(ticks=xticks, labels=xticks)

    plt.xlabel(r'Queue size', fontsize=14)
    plt.ylabel(y_label, fontsize=14)  # label the y axis

    # add the legend (will default to 'best' location)
    plt.legend(fontsize=14)
    plt.title(title)

    plt.savefig(filename.replace('csv', 'png'), dpi=300, bbox_inches="tight")


def make_plot_all_percentages(filename, relative=False, y_label='Speedup (normalised)', title=''):
    global fig_id
    global BINS_STATIC
    global BINS_DYNAMIC
    global BINS_DYNAMIC_NO_FORWARD

    fig_id = fig_id + 1

    df = pd.read_csv(filename)
    x = df['percentage']
    y = df['static'].replace({0: np.nan})

    x2 = df['percentage']
    y2 = df[f'dynamic (q_size {Q_SIZE_DYNAMIC})'].replace({0: np.nan})
    x3 = df['percentage']
    y3 = df[f'dynamic_no_forward (q_size {Q_SIZE_DYNAMIC_NO_FORWARD})'].replace({0: np.nan})

    if relative:
        static_baseline = y[0]
        y = [1 for _ in df['static']]
        y2 = [static_baseline/val for val in y2]
        y3 = [static_baseline/val for val in y3]

    # plot
    fig = plt.figure(fig_id, figsize=(8, 8))
    plt.plot(x3, y3, linestyle='-', marker='^', 
                 label=f'dynamic (no forwarding, q_size {Q_SIZE_DYNAMIC_NO_FORWARD})',
                 color=colors[2], mfc='w', markersize=8)
    plt.plot(x2, y2, linestyle='-', marker='s', 
                 label=f'dynamic (forwarding, q_size {Q_SIZE_DYNAMIC})',
                 color=colors[1], mfc='w', markersize=8)
    plt.plot(x, y, linestyle='-', marker='o', label='static',
                 color=colors[0], mfc='w', markersize=8)

    # Add frequencies as
    if not 'sim' in filename:
        freq = get_freq(BINS_STATIC[0])
        ii = get_ii(BINS_STATIC[0])
        plt.text(x[len(x) - 1], y[len(y) - 1], f'{freq} MHz\nII={ii}', fontsize='x-small', fontstyle='italic')

        for i in range(len(BINS_DYNAMIC)):
            if Q_SIZES_DYNAMIC[i] == Q_SIZE_DYNAMIC:
                freq = get_freq(BINS_DYNAMIC[i])
                ii = get_ii(BINS_DYNAMIC[i])
                plt.text(x2[len(x2) - 1], y2[len(y2) - 1], f'{freq} MHz\nII={ii}', fontsize='x-small', fontstyle='italic')

        for i in range(len(BINS_DYNAMIC_NO_FORWARD)):
            if Q_SIZES_DYNAMIC_NO_FORWARD[i] == Q_SIZE_DYNAMIC_NO_FORWARD:
                freq = get_freq(BINS_DYNAMIC_NO_FORWARD[i])
                ii = get_ii(BINS_DYNAMIC_NO_FORWARD[i])
                plt.text(x3[len(x3) - 1], y3[len(y3) - 1], f'{freq} MHz\nII={ii}', fontsize='x-small', fontstyle='italic') 

    xticks = PERCENTAGES_WAIT
    plt.xticks(ticks=xticks, labels=xticks)

    plt.xlabel(r'% of data dependencies', fontsize=14)
    plt.ylabel(y_label, fontsize=14)  # label the y axis

    # add the legend (will default to 'best' location)
    plt.legend(fontsize=14)
    plt.title(title)

    if relative:
        plt.savefig(filename.replace('.csv', '_normalised.png'), dpi=300, bbox_inches="tight")
    else:
        plt.savefig(filename.replace('csv', 'png'), dpi=300, bbox_inches="tight")



if __name__ == '__main__':
    is_sim = any('sim' in arg for arg in sys.argv[1:])

    SUB_DIR = 'simulation' if is_sim else 'hardware'
    BIN_EXTENSION = 'fpga_sim' if is_sim else 'fpga'
    KERNEL_ASIZE_PAIRS = KERNEL_ASIZE_PAIRS_SIM if is_sim else KERNEL_ASIZE_PAIRS
    Y_LABEL = 'Cycles' if is_sim else 'Time'

    for kernel in KERNEL_ASIZE_PAIRS.keys():
        BINS_STATIC = [f'{kernel}/bin/{kernel}_static.{BIN_EXTENSION}']
        BINS_DYNAMIC = [f'{kernel}/bin/{kernel}_dynamic_{s}qsize.{BIN_EXTENSION}' 
                        for s in Q_SIZES_DYNAMIC]
        BINS_DYNAMIC_NO_FORWARD = [f'{kernel}/bin/{kernel}_dynamic_no_forward_{s}qsize.{BIN_EXTENSION}' 
                                   for s in Q_SIZES_DYNAMIC_NO_FORWARD]

        # Ensure dir structure exists
        Path(f'{EXP_DATA_DIR}/{kernel}/{SUB_DIR}').mkdir(parents=True, exist_ok=True)
        for distr_idx, distr_name in DATA_DISTRIBUTIONS.items():
            percentage_suffix = ''
            percentage_suffix_title = ''
            if distr_idx == 2:
                percentage_suffix = f'_{PERCENTAGE_WAIT}'
                percentage_suffix_title = f'({PERCENTAGE_WAIT} %)'

            csv_fname = f'{EXP_DATA_DIR}/{kernel}/{SUB_DIR}/{distr_name}{percentage_suffix}.csv'

            make_plot(csv_fname, title=f'Data distribution {distr_name} {percentage_suffix_title}')


        ### Plot speedup vs. % data hazards
        csv_file_all_percentages = f'{EXP_DATA_DIR}/{kernel}/{SUB_DIR}/{CSV_PERCENTAGES_RES_FILE}'
        if Path(csv_file_all_percentages).is_file():
            make_plot_all_percentages(csv_file_all_percentages, y_label=Y_LABEL,
                                    title='Performance at various levels of data hazards')

            make_plot_all_percentages(csv_file_all_percentages, relative=True,
                                    title='Speedup of Store Queue at various levels of data hazards')


