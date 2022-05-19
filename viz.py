import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm


def visualize_and_compare_poses():
    
    # old_ts_df = pd.read_csv('/home/minh/duy/out1604/raw.txt', sep= ' ')
    old_ts_df = pd.read_csv('/home/minh/duy/tien/converted_to_map_frame.txt', sep= ' ')
    old_ts = old_ts_df['timestamps']
    y_old = old_ts_df['ty']
    z_old = old_ts_df['tx']


    new_ts_df = pd.read_csv('/home/minh/duy/tien/converted.txt', sep= ' ')
    new_ts = old_ts_df['timestamps']
    y_new = new_ts_df['ty']
    z_new = new_ts_df['tx']


    plt.figure(0)
    plt.plot(y_old, z_old, 'bo')
    plt.plot(y_new, z_new, 'rx')
    print(len(y_old))
    print(len(y_new))
    plt.show()
            

# postprocess()
visualize_and_compare_poses()