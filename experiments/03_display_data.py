import numpy as np
import pandas as pd
from transformations import translation_matrix, rotation_matrix, translation_from_matrix, rotation_from_matrix, \
    concatenate_matrices, quaternion_matrix, quaternion_from_matrix
from utils.geometry import convert_quat_wxyz_to_xyzw, convert_quat_xyzw_to_wxyz
import os
from glob import glob
from utils.vis_utils import set_axes_equal
import json

# np.set_printoptions(formatter=dict(int=lambda x: f'{x:4}'))  # to widen the printed array

# TODO SUPER NOTE !! in the transformations package, Quaternions w+ix+jy+kz are represented as [w, x, y, z].
origin, xaxis, yaxis, zaxis = (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)

# change depend on the data you use
pc_name = os.getlogin()
leds = 'rrrgggbbb'

# indenter = '20'
# data_name = 'data_2023_04_27-01:36:57'
# JSON_FILE = f"/home/{pc_name}/catkin_ws/src/allsight/dataset/{leds}/data/{indenter}/{data_name}/{data_name}.json"
# buffer_paths = [JSON_FILE]

indenter = ['30', '40', '20']
paths = [f"{os.path.dirname(os.path.abspath(__file__))}/dataset/{leds}/data/{ind}" for ind in indenter]
buffer_paths = []
for p in paths:
    buffer_paths += [y for x in os.walk(p) for y in glob(os.path.join(x[0], '*.json'))]

buffer_paths = [p for p in buffer_paths if ('transformed' not in p) and
                                           ('final' not in p) and
                                           ('summary' not in p)
                ]

j_i = 0
for JSON_FILE in buffer_paths:

    j_i += 1
    print(f'Loading dataset: {JSON_FILE[-58:]} \t {j_i}/{len(buffer_paths)}')

    df_data = pd.read_json(JSON_FILE).transpose()

    print(f'df length: {df_data.shape}')

    save = True

    # do some plotting
    import matplotlib.pyplot as plt
    from pytransform3d import rotations as pr
    from pytransform3d import transformations as pt
    from pytransform3d.transform_manager import TransformManager
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(df_data.pose)):
        pxyz = df_data['pose'][i][0]
        ax.scatter(pxyz[0], pxyz[1], pxyz[2], c='black')

    ax.view_init(90, 90)
    set_axes_equal(ax)
    fig.savefig(JSON_FILE[:-5] + '_test', dpi=200, bbox_inches='tight')

    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d

    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            FancyArrowPatch.draw(self, renderer)


    tm = TransformManager()

    amnt = 1 if len(df_data) < 6000 else 25
    for i in range(0, len(df_data['pose']), amnt):
        object2cam = pt.transform_from_pq(np.hstack((df_data['pose'][i][0],
                                                     pr.quaternion_wxyz_from_xyzw(
                                                         df_data['pose'][i][1]))))

        tm.add_transform("object" + str(i), "camera", object2cam)

    ax = tm.plot_frames_in("camera", s=0.0015, show_name=False)

    scale = 2000
    for i in range(0, len(df_data['pose']), 5):
        pxyz = df_data['pose'][i][0]
        nf_xyz = df_data['ft'][i]

        # a = Arrow3D([pxyz[0], pxyz[0] + nf_xyz[0] / scale], [pxyz[1], pxyz[1] + nf_xyz[1] / scale],
        #             [pxyz[2], pxyz[2] + nf_xyz[2] / scale], mutation_scale=5,
        #             lw=0.3, arrowstyle="-|>", color="k")

        # ax.add_artist(a)

    ax.set_xlim((-0.014, 0.014))
    ax.set_ylim((-0.014, 0.014))
    ax.set_zlim((0.0, 0.03))
    set_axes_equal(ax)

    fig.savefig(JSON_FILE[:-5] + '_pose', dpi=200, bbox_inches='tight')
    ax.view_init(90, 90)
    set_axes_equal(ax)

    fig.savefig(JSON_FILE[:-5] + '_pose_top', dpi=200, bbox_inches='tight')
    ax.view_init(0, 0)
    set_axes_equal(ax)

    fig.savefig(JSON_FILE[:-5] + '_pose_side', dpi=200, bbox_inches='tight')



    plt.close('all')
    print('Finished to transform the data.\n')

# buffer_paths = [p for p in buffer_paths if 'summary' not in p]
# j_i = 0
# for JSON_FILE in buffer_paths:
#     print(f'transforming dataset: {JSON_FILE[-58:]} \t {j_i}/{len(buffer_paths)}')
#     df_data = pd.read_json(JSON_FILE).transpose()
#     j_i += 1
#     for i in df_data.index:
#         df_data.loc[i].frame = df_data.loc[i].frame.replace(f'/{leds}', f'/markers/{leds}')
#         df_data.loc[i].ref_frame = df_data.loc[i].ref_frame.replace(f'/{leds}', f'/markers/{leds}')
#
#     save = True
#     if save:
#         import json
#
#         to_dict = {}
#         for index, row in list(df_data.iterrows()):
#             to_dict[index] = dict(row)
#         with open(JSON_FILE, 'w') as json_file:
#             json.dump(to_dict, json_file, indent=3)

# buffer_paths = [p for p in buffer_paths if 'summary' in p]
# j_i = 0
# for JSON_FILE in buffer_paths:
#     print(f'transforming dataset: {JSON_FILE} \t {j_i}/{len(buffer_paths)}')
#     with open(JSON_FILE, 'rb') as handle:
#         summ = json.load(handle)
#
#     j_i += 1
#
#     summ["sensor_id"] = 9
#     save = True
#     if save:
#         import json
#
#         with open(JSON_FILE, 'w') as json_file:
#             json.dump(summ, json_file, indent=3)