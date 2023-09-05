import numpy as np
import pandas as pd
import os
from glob import glob
import cv2
from utils.vis_utils import data_for_cylinder_along_z, data_for_sphere_along_z
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
from scipy import spatial

# Define the parameters
pc_name = os.getlogin()
leds = 'rrrgggbbb'
gel = 'clear'
start_sample_time = 2
indenter = ['sphere3']
save = False
save_name = '/home/{pc_name}/allsight_sim/experiments/osher'

sim_prefix = f'/home/{pc_name}/allsight_sim/experiments/'
real_prefix = f'/home/{pc_name}/catkin_ws/src/allsight'

# paths of the real and sim dataframes
sim_paths = [f"{sim_prefix}/allsight_sim_dataset/{gel}/{leds}/data/{ind}" for ind in indenter]
real_paths = [f"{real_prefix}/dataset/{gel}/{leds}/data/{ind}" for ind in indenter]


buffer_sim_paths, buffer_real_paths = [], []

# List df paths
for p in sim_paths:
    buffer_sim_paths += [y for x in os.walk(p) for y in glob(os.path.join(x[0], '*.json'))]
for p in real_paths:
    buffer_real_paths += [y for x in os.walk(p) for y in glob(os.path.join(x[0], '*.json'))]
buffer_real_paths = [p for p in buffer_real_paths if ('transformed_annotated' in p)]

# Load real_paths to single df
for idx, p in enumerate(buffer_real_paths):
    if idx == 0:
        df_data_real = pd.read_json(p).transpose()
    else:
        df_data_real = pd.concat([df_data_real, pd.read_json(p).transpose()], axis=0)

# Load sim_paths to single df
for idx, p in enumerate(buffer_sim_paths):
    if idx == 0:
        df_data_sim = pd.read_json(p).transpose()
    else:
        df_data_sim = pd.concat([df_data_sim, pd.read_json(p).transpose()], axis=0)

# Sample for the real_dataset
# df_data_real = df_data_real[df_data_real.time > start_sample_time]
# df_data_real = df_data_real[df_data_real.num > 1]
n_samples = len(df_data_sim)
df_data_real = df_data_real.sample(n=n_samples)

# convert to arrays
pose_real = np.array([df_data_real.iloc[idx].pose_transformed[0][:3] for idx in range(df_data_real.shape[0])])
pose_sim = np.array([df_data_sim.iloc[idx].pose_transformed[0] for idx in range(df_data_sim.shape[0])])

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

Xc, Yc, Zc = data_for_cylinder_along_z(0., 0., 0.012, 0.016)
ax.plot_surface(Xc, Yc, Zc, alpha=0.1, color='grey')
Xc, Yc, Zc = data_for_sphere_along_z(0., 0., 0.012, 0.016)
ax.plot_surface(Xc, Yc, Zc, alpha=0.1, color='grey')

# REAL TO SIM MATCHING
tree = spatial.KDTree(pose_sim)

ind_to_keep = []
for sample in range(len(pose_real)):

    real_xyz = pose_real[sample]
    real_image = (cv2.imread(df_data_real['frame'][sample].replace("osher", pc_name))).astype(np.uint8)

    _, ind = tree.query(real_xyz)
    ind_to_keep.append(ind)
    sim_xyz = pose_sim[ind]
    sim_image = (cv2.imread(sim_prefix + df_data_sim['frame'][ind])).astype(np.uint8)

    ax.scatter(sim_xyz[0], sim_xyz[1], sim_xyz[2], c='black', label='sim')

    cv2.imshow('sim\t\t\treal', np.concatenate((sim_image, real_image), axis=1))
    # ax.scatter(real_xyz[0], real_xyz[1], real_xyz[2], c='red', label='true')

    # print(f'real {real_xyz}\nsim {sim_xyz}\nnorm {np.linalg.norm((real_xyz, sim_xyz)) * 1000}')

    ax.set_xlim((-0.014, 0.014))
    ax.set_ylim((-0.014, 0.014))
    ax.set_zlim((0.0, 0.03))

    # plt.pause(0.001)
    wait = 1 if sample == 0 else 1
    cv2.waitKey(wait) & 0xff

df_data_sim = df_data_sim.iloc[ind_to_keep]

if save:
    import json
    to_dict = {}
    for index, row in list(df_data_sim.iterrows()):
        to_dict[index] = dict(row)
    with open(r'{}_aligned.json'.format(save_name), 'w') as json_file:
        json.dump(to_dict, json_file, indent=3)

# SIM TO REAL MATCHING
# tree = spatial.KDTree(pose_real)
# for i in range(len(pose_sim)):
#
#     sample = i
#     sim_xyz = pose_sim[sample]
#     sim_image = (cv2.imread(sim_prefix + df_data_sim['frame'][sample])).astype(np.uint8)
#
#     ax.scatter(sim_xyz[0], sim_xyz[1], sim_xyz[2], c='black', label='sim')
#
#     _, ind = tree.query(sim_xyz)
#     real_xyz = pose_real[ind]
#
#     real_image = (cv2.imread(df_data_real['frame'][ind].replace("osher", pc_name))).astype(np.uint8)
#     cp = df_data_real['contact_px'][ind]
#
#     cv2.imshow('sim\t\t\treal', np.concatenate((sim_image, real_image), axis=1))
#
#     ax.scatter(real_xyz[0], real_xyz[1], real_xyz[2], c='red', label='true')
#
#     print(f'real {real_xyz}\nsim {sim_xyz}\nnorm {np.linalg.norm((real_xyz, sim_xyz)) * 1000}')
#
#     ax.set_xlim((-0.014, 0.014))
#     ax.set_ylim((-0.014, 0.014))
#     ax.set_zlim((0.0, 0.03))
#
#     plt.pause(0.001)
#     wait = 1000 if i == 0 else 1000
#     cv2.waitKey(wait) & 0xff