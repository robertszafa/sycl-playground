import re
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Keep parameters synced
from run_exp import (EXP_DATA_DIR, Q_SIZES_DYNAMIC, Q_SIZES_DYNAMIC_NO_FORWARD, DATA_DISTRIBUTIONS,
                     KERNEL_ASIZE_PAIRS)


POWER_CS_FILE = 'power.csv'


# BRAM_STATIC_PARTITION = 492
# ALUTS_STATIC_PARTITION = 179950
# REGISTERS_STATIC_PARTITION = 98940
# DSP_STATIC_PARTITION = 0


# return ALUTs, REGs, RAMs, DSPs, Fmax
def get_resources(bin):
    hw_prj = bin.replace('_sim', '') + '.prj'

    res = {'aluts' : 0, 'regs' : 0, 'rams' : 0, 'dsps' : 0, 'freq' : 0}
    try:
        with open(f'{hw_prj}/acl_quartus_report.txt', 'r') as f:
            report_str = f.read()

            alut_usage_str = re.findall("ALUTs: (\d+)", report_str)
            reg_usage_str = re.findall("Registers: ([,\d]+)", report_str)
            ram_usage_str = re.findall("RAM blocks: (.*?)/", report_str)
            dsp_usage_str = re.findall("DSP blocks: (.*?)/", report_str)
            freq_str = re.findall("Actual clock freq: (\d+)", report_str)

            res['aluts'] = int(re.sub("[^0-9]", "", alut_usage_str[0]))
            res['regs'] = int(re.sub("[^0-9]", "", reg_usage_str[0]))
            res['rams'] = int(re.sub("[^0-9]", "", ram_usage_str[0]))
            res['dsps'] = int(re.sub("[^0-9]", "", dsp_usage_str[0]))
            res['freq'] = freq_str[0]
    except:
        pass
    
    return res

def get_power(kernel, approach, q_size_idx):
    filename = f'{EXP_DATA_DIR}/{kernel}/hardware/power.csv'
    try:
        df = pd.read_csv(filename)
        res = df.loc[q_size_idx][approach]
        return res
    except:
        return 0

def get_min_max_runtime(kernel, approach, q_size_idx):
    min = np.inf
    max = 0
    for distr_idx, distr_name in DATA_DISTRIBUTIONS.items():
        distr_name = f'{distr_name}_5' if distr_idx == 2 else distr_name
        filename = f'{EXP_DATA_DIR}/{kernel}/hardware/{distr_name}.csv'
        df = pd.read_csv(filename)
        time = df.loc[q_size_idx][approach]
        min = time if time < min else min
        max = time if time > max else max

    return min, max

def annotate_difference(resources_base, resources_new):
    res = {}

    for key in resources_new.keys():
        diff = int(resources_new[key]) -  int(resources_base[key])
        if diff >= 0:
            diff = f'+{diff}'

        res[key] = f'{resources_new[key]} ({diff})'
    
    return res


if __name__ == '__main__':
    BIN_EXTENSION = 'fpga_hw'
    SUB_DIR = 'hardware'

    for kernel in KERNEL_ASIZE_PAIRS.keys():
        print('Running kernel:', kernel)

        BINS_STATIC = [f'{kernel}/bin/{kernel}_static.{BIN_EXTENSION}']
        BINS_DYNAMIC = [f'{kernel}/bin/{kernel}_dynamic_{s}qsize.{BIN_EXTENSION}' 
                        for s in Q_SIZES_DYNAMIC]
        BINS_DYNAMIC_NO_FORWARD = [f'{kernel}/bin/{kernel}_dynamic_no_forward_{s}qsize.{BIN_EXTENSION}' 
                                   for s in Q_SIZES_DYNAMIC_NO_FORWARD]
        

        Path(f'{EXP_DATA_DIR}/{kernel}/{SUB_DIR}').mkdir(parents=True, exist_ok=True)

        with open(f'{EXP_DATA_DIR}/{kernel}/{SUB_DIR}/resources.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['approach', 'ALUTs', 'REGs', 'RAMs', 'DSPs', 'Fmax (MHz)', 'Watts', 'Exec time min-max (ms)'])

            min_static, max_static = get_min_max_runtime(kernel, 'static', 0)
            min_max_static = f'{min_static} - {max_static} (1x)'
            power_static = get_power(kernel, 'static', 0)
            resources_static = get_resources(BINS_STATIC[0])

            new_row = ['static'] + list(resources_static.values()) + [power_static, min_max_static]
            writer.writerow(new_row)
            
            for i, q_size in enumerate(Q_SIZES_DYNAMIC_NO_FORWARD):
                if i < len(BINS_DYNAMIC):
                    min_dynamic, max_dynamic = get_min_max_runtime(kernel, 'dynamic', i)
                    relative_speedup_1 = round(min_static/min_dynamic, 2)
                    relative_speedup_2 = round(max_static/max_dynamic, 2)
                    relative_str = f'{min(relative_speedup_1, relative_speedup_2)}-{max(relative_speedup_1, relative_speedup_2)}x'
                    min_max_dynamic = f'{min_dynamic} - {max_dynamic} ({relative_str})'

                    power_dynamic = get_power(kernel, 'dynamic', i)
                    resources_dynamic = get_resources(BINS_DYNAMIC[i])
                    resources_dynamic = annotate_difference(resources_static, resources_dynamic)

                    new_row = [f'dynamic_{q_size}qsize'] + list(resources_dynamic.values()) + [power_dynamic, min_max_dynamic]
                    writer.writerow(new_row)

                if i < len(BINS_DYNAMIC_NO_FORWARD):
                    min_dynamic_no_frwd, max_dynamic_no_frwd = get_min_max_runtime(kernel, 'dynamic_no_forward', i)
                    relative_speedup_1 = round(min_static/min_dynamic_no_frwd, 2)
                    relative_speedup_2 = round(max_static/max_dynamic_no_frwd, 2)
                    relative_str = f'{min(relative_speedup_1, relative_speedup_2)}-{max(relative_speedup_1, relative_speedup_2)}x'
                    min_max_dynamic_no_frwd = f'{min_dynamic_no_frwd} - {max_dynamic_no_frwd} ({relative_str})'

                    power_dynamic_no_frwd = get_power(kernel, 'dynamic_no_forward', i)
                    resources_dynamic_no_frwd = get_resources(BINS_DYNAMIC_NO_FORWARD[i])
                    resources_dynamic_no_frwd = annotate_difference(resources_static, resources_dynamic_no_frwd)

                    new_row = [f'dynamic_no_forward_{q_size}qsize'] + list(resources_dynamic_no_frwd.values()) + [power_dynamic_no_frwd, min_max_dynamic_no_frwd]
                    writer.writerow(new_row)

