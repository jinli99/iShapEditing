import copy
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import open3d as o3d

from scipy.spatial import cKDTree

# path = './datas/layer-guidance/4'
# for i in ['8']:
#     mesh = o3d.io.read_triangle_mesh(os.path.join(path, 'layer'+i+'.obj')).filter_smooth_simple(number_of_iterations=1)
#     o3d.io.write_triangle_mesh(i+'_smooth.obj', mesh)


# mesh = o3d.io.read_triangle_mesh('datas/real-shape-edit/chairs/test8/edit03.obj').subdivide_loop(number_of_iterations=2)
# o3d.io.write_triangle_mesh('mesh.obj', mesh)

mesh = o3d.io.read_triangle_mesh('edit03.obj').filter_smooth_simple(number_of_iterations=20).remove_degenerate_triangles()
o3d.io.write_triangle_mesh('1.obj', mesh)

# from meshProcess import calc_chamfer
# path = '/home/jing/Desktop/Ours'
# for name in os.listdir(path):
#     if not name.startswith('test'):
#         continue
#     for edit in os.listdir(os.path.join(path, name)):
#         if not edit.startswith('edit'):
#             continue
#         s = calc_chamfer(filename_gt=os.path.join(path, name, 'test.obj'), filename_pre=os.path.join(path, name, edit), point_num=200000)
#         print(name, edit, s)

"""Save kd_tree into .pkl"""
# path = '/media/jing/I/datas/airplanes'
# from scipy.spatial import cKDTree
# import pickle
# point_num = 200000
# i = 0
# for test in os.listdir(path):
#     if os.path.exists(os.path.join(path, test, 'mesh_scale_smooth.obj')):
#         mesh = o3d.io.read_triangle_mesh(os.path.join(path, test, 'mesh_scale_smooth.obj'))
#         mesh.translate(-mesh.get_center())
#         vtx = mesh.vertices
#         bbx = mesh.get_max_bound() - mesh.get_min_bound()
#         mesh.scale(2. / (bbx.max() + 0.01), center=mesh.get_center())
#         points = np.asarray(mesh.sample_points_uniformly(point_num).points, dtype=np.float32)
#         np.save(os.path.join(path, test, 'pcd.npy'), points)
#         tree = cKDTree(points)
#         with open(os.path.join(path, test, 'kdtree.pkl'), 'wb') as f:
#             pickle.dump(tree, f)
#         i += 1
#         print(i, test)
#     elif os.path.exists(os.path.join(path, test, 'model.obj')):
#         mesh = o3d.io.read_triangle_mesh(os.path.join(path, test, 'model.obj'))
#         mesh.translate(-mesh.get_center())
#         vtx = mesh.vertices
#         bbx = mesh.get_max_bound() - mesh.get_min_bound()
#         mesh.scale(2. / (bbx.max() + 0.01), center=mesh.get_center())
#         points = np.asarray(mesh.sample_points_uniformly(point_num).points, dtype=np.float32)
#         np.save(os.path.join(path, test, 'pcd.npy'), points)
#         tree = cKDTree(points)
#         with open(os.path.join(path, test, 'kdtree.pkl'), 'wb') as f:
#             pickle.dump(tree, f)
#         i += 1
#         print(i, test)


# path = 'datas/comparison/airplanes'
# path1 = os.path.join('/media/jing/J/SLIDE/pointnet2/deformation/airplanes')
# case = 2
# if case == 1:
#     for name in os.listdir(path):
#         if not name.startswith('test'):
#             continue
#         os.makedirs(os.path.join(path1, name), exist_ok=True)
#         shutil.copy(os.path.join(path, name, 'test.npz'), os.path.join(path1, name, 'test.npz'))
# elif case == 2:
#     for name in os.listdir(path1):
#         if not name.startswith('test'):
#             continue
#         shutil.copy(os.path.join(path1, name, 'reconstructed_pcd.npz'), os.path.join(path, name, 'reconstructed_pcd.npz'))

