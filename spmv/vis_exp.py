import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Keep parameters synced
from run_exp import *

plt.rcParams['font.size'] = 14
# colors = seaborn.color_palette("rocket", 3)
colors = ['#c3121e', '#0348a1', '#ffb01c', '#027608', '#0193b0', '#9c5300', '#949c01', '#7104b5']
# plt.figure(unique_id) ensures that different invocations of make_plot() don't interfere
fig_id = 0


def make_plot(filename, y_label='Execution time (ms)', title=''):
  global fig_id 

  fig_id = fig_id + 1

  df = pd.read_csv(filename)
  x = df['q_size'].dropna()
  y = df['static'].dropna()
  x2 = df['q_size'].dropna()
  y2 = df['dynamic'].dropna().replace({0:np.nan})
  x3 = df['q_size'].dropna()
  y3 = df['dynamic_no_forward'].dropna()

  # plot
  fig = plt.figure(fig_id, figsize=(8, 8))
  plt.semilogx(x3, y3, linestyle='-', marker='^', label='dynamic (no forwarding)',
              color=colors[2], mfc='w', markersize=8)  
  plt.semilogx(x2, y2, linestyle='-', marker='s', label='dynamic (forwarding)',
              color=colors[1], mfc='w', markersize=8)  
  plt.semilogx(x, y, linestyle='-', marker='o', label='static',
              color=colors[0], mfc='w', markersize=8)  

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
  y_label = 'Cycles' if is_sim else 'Time (ms)'

  for k, v in DATA_DISTRIBUTIONS.items():
    make_plot(f'{RUN_DATA_DIR}/{SUB_DIR}/{DATA_DISTRIBUTIONS[k]}.csv', 
              y_label=y_label,
              title=f'Array of {M_SIZE} elements with data distribution {v}')
