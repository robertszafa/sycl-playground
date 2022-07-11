import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Keep parameters synced
from run_exp import *


def make_plot(filename, y_label='Execution time (ms)'):
  plt.rcParams['font.size'] = 14
  df = pd.read_csv(filename)

  x = df['q_size'].dropna()
  y = df['static'].dropna()

  x2 = df['q_size'].dropna()
  y2 = df['dynamic'].dropna()

  x3 = df['q_size'].dropna()
  y3 = df['dynamic_no_forward'].dropna()

  # bodacious colors
  # colors = sns.color_palette("rocket", 3)
  colors = ['#c3121e', '#0348a1', '#ffb01c', '#027608', '#0193b0', '#9c5300', '#949c01', '#7104b5']

  # plot
  fig = plt.figure(1, figsize=(8, 6))
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

  plt.savefig(filename.replace('csv', 'png'), dpi=300, bbox_inches="tight")


if __name__ == '__main__':
  make_plot(f'{RUN_DATA_DIR}{DATA_DISTRIBUTIONS[0]}.csv', y_label='Cycles')