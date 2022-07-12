import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    with open(f'{hw_prj}/acl_quartus_report.txt', 'r') as f:
        contents = f.read()
        m = re.search(r'Kernel fmax: (\d+)', contents)
        if m:
            return int(m.group(1))

        return 0


def make_plot(filename, relative=True, y_label='Execution time (ms)', title=''):
    global fig_id

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

    array_size = 1000  # 2**20
    relative = True

    is_sim = any('sim' in arg for arg in sys.argv[1:])
    SUB_DIR = 'simulation' if is_sim else 'hardware'
    y_label = 'Cycles' if is_sim else 'Time (ms)'
    if relative:
        y_label += ' relative to static'

    for k, v in DATA_DISTRIBUTIONS.items():
        make_plot(f'{RUN_DATA_DIR}/{SUB_DIR}/{DATA_DISTRIBUTIONS[k]}_{array_size}.csv',
                  relative=relative,
                  y_label=y_label,
                  title=f'Array of {array_size} elements with data distribution {v}')
