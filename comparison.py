import os.path
import shutil
import open3d as o3d
import numpy as np
from meshProcess import calc_mesh_points_normals, calc_local_distance
import torch as th


"""Drag Information of DragBench3D"""
drag_info_car = {
    'test1':{
            'edit01':{
                'keypoint number': [10], 'delta': [[0.0, 0.1, 0.0]]
            },
            'edit11': {
                'keypoint number': [15, 13], 'delta': [[0.0, -0.05, 0.0]]
            },
            'edit02': {
                'keypoint number': [7], 'delta': [[-0.05, 0.05, 0.0]]
            },
            'edit12': {
                'keypoint number': [12, 2], 'delta': [[0.05, 0.0, 0.0]]
            }},
    'test2':{
            'edit11':{
                'keypoint number': [8, 5, 9], 'delta': [[0.0, 0.05, 0.0]]
            },
            'edit12': {
                'keypoint number': [6, 10, 11], 'delta': [[0.0, 0.06, 0.0]]
            },
            'edit13': {
                'keypoint number': [11, 10, 6], 'delta': [[0.0, -0.06, 0.0]]
            },
            'edit14': {
                'keypoint number': [1, 13], 'delta': [[-0.06, 0.0, 0.0]]
            }},
    'test3':{
            'edit11':{
                'keypoint number': [11, 9, 6, 3, 11, 6, 3, 9], 'delta': [[0.0, 0.07, 0.0]]
            }},
    'test4':{
            'edit11':{
                'keypoint number': [12, 7], 'delta': [[0.0, 0.07, 0.0]]
            },
            'edit12': {
                'keypoint number': [11, 2], 'delta': [[0.06, 0.0, 0.0]]
            },
            'edit01': {
                'keypoint number': [1], 'delta': [[-0.06, 0.0, 0.0]]
            },
            'edit13': {
                'keypoint number': [7], 'delta': [[0.05, -0.03, 0.0]]
            }},
    'test5': {
        'edit11': {
            'keypoint number': [8, 12, 3], 'delta': [[0.0, -0.06, 0.0]]
        },
        'edit12': {
            'keypoint number': [15, 2, 13], 'delta': [[0.07, 0.0, 0.0]]
        }},
    'test6': {
        'edit01': {
            'keypoint number': [15], 'delta': [[0.0, -0.05, 0.0]]
        },
        'edit11': {
            'keypoint number': [8, 11, 5, 12, 13], 'delta': [[0.0, 0.0, 0.05], [0.0, 0.0, 0.05], [0.0, 0.0, 0.05], [0.0, 0.0, -0.05], [0.0, 0.0, -0.05]]
        },
        'edit12': {
            'keypoint number': [1, 2], 'delta': [[-0.06, 0.0, 0.0], [0.06, 0.0, 0.0]]
        },
        'edit13': {
            'keypoint number': [14, 7, 9], 'delta': [[0.0, 0.06, 0.0]]
        }},
    'test7': {
        'edit01': {
            'keypoint number': [4], 'delta': [[0.0, -0.06, 0.0]]
        }},
    'test10': {
        'edit11': {
            'keypoint number': [9, 8, 13, 5], 'delta': [[0.0, -0.06, 0.0]]
        },
        'edit12': {
            'keypoint number': [14, 12], 'delta': [[0.0, -0.06, 0.0]]
        },
        'edit13': {
            'keypoint number': [2, 15], 'delta': [[0.07, 0.0, 0.0]]
        }},
    }

drag_info_chair = {
        'test1':{
            'edit11':{
                'keypoint number': [12, 9], 'delta': [0.0, 0.1, 0.0]
            },
            'edit12': {
                'keypoint number': [12, 9], 'delta': [0.15, -0.05, 0.0]
            },
            'edit13': {
                'keypoint number': [8, 11, 5, 2, 3, 10, 6, 13], 'delta': [-0.1, 0.0, 0.0]
            }
        },
        'test2': {
            'edit01': {
                'keypoint number': [10], 'delta': [0.0, -0.1, 0.0]
            },
            'edit02': {
                'keypoint number': [10], 'delta': [-0.1, 0.0, 0.0]
            },
            'edit03': {
                'keypoint number': [9], 'delta': [0.1, 0.0, 0.0]
            },
            'edit11': {
                'keypoint number': [7, 10], 'delta': [0.0, 0.1, 0.0]
            },
            'edit12': {
                'keypoint number': [7, 10], 'delta': [0.0, -0.1, 0.0]
            },
            'edit13': {
                'keypoint number': [7, 10], 'delta': [0.15, -0.05, 0.0]
            },
            'edit14': {
                'keypoint number': [7, 10], 'delta': [-0.1, 0.0, 0.0]
            },
            'edit15': {
                'keypoint number': [3, 9, 6, 15, 4, 8, 2], 'delta': [0.1, 0.0, 0.0]
            }
        },
        'test3': {
            'edit11': {
                'keypoint number': [12, 0], 'delta': [0.0, 0.1, 0.0]
            },
            'edit12': {
                'keypoint number': [4, 6, 14, 15], 'delta': [-0.1, -0.1, 0.0]
            }
        },
        'test4': {
            'edit01': {
                'keypoint number': [12], 'delta': [0.0, -0.1, 0.0]
            },
            'edit11': {
                'keypoint number': [8, 7, 14], 'delta': [0.0, 0.15, 0.0]
            },
            'edit12': {
                'keypoint number': [3, 9, 6, 15, 2, 10, 5], 'delta': [-0.1, 0.0, 0.0]
            },
        },
        'test6': {
            'edit01': {
                'keypoint number': [11], 'delta': [0.0, 0.1, 0.0]
            },
            'edit02': {
                'keypoint number': [15], 'delta': [0.1, 0.0, 0.0]
            },
            'edit11': {
                'keypoint number': [2, 15, 8, 5, 13], 'delta': [0.1, 0.0, 0.0]
            },
        },
        'test8': {
            'edit01': {
                'keypoint number': [12], 'delta': [0.15, 0.0, 0.0]
            },
            'edit11': {
                'keypoint number': [12, 9], 'delta': [0.1, -0.05, 0.0]
            },
            'edit12': {
                'keypoint number': [7, 11, 5, 2, 15, 6, 13, 10, 3], 'delta': [-0.1, 0.0, 0.0]
            }
        },
        'test9': {
            'edit01': {
                'keypoint number': [8], 'delta': [0.1, 0.0, 0.0]
            },
            'edit11': {
                'keypoint number': [8, 5, 3, 11, 9, 4], 'delta': [0.1, 0.0, 0.0]
            }
        }
    }

drag_info_airplane = {
    'test1':{
            'edit01':{
                'keypoint number': [6], 'delta': [[0.0, -0.05, 0.0]]
            },
            'edit11': {
                'keypoint number': [14, 11], 'delta': [[0.05, 0.0, 0.0]]
            },
            'edit02': {
                'keypoint number': [1], 'delta': [[-0.07, 0.0, 0.0]]
            }},
    'test2': {
        'edit01': {
            'keypoint number': [4], 'delta': [[-0.07, 0.0, 0.0]]
        },
        'edit11': {
            'keypoint number': [2, 1], 'delta': [[0.07, 0.0, 0.0]]
        },
        'edit02': {
            'keypoint number': [4], 'delta': [[0.07, 0.0, 0.0]]
        },
        'edit12': {
            'keypoint number': [5, 6], 'delta': [[0.07, 0.0, 0.0]]
        },
        'edit03': {
            'keypoint number': [3], 'delta': [[0.07, 0.0, 0.0]]
        }},
    'test3': {
        'edit01': {
            'keypoint number': [4], 'delta': [[0.08, 0.0, 0.0]]
        },
        'edit11': {
            'keypoint number': [6, 5], 'delta': [[0.08, 0.0, 0.0]]
        },
        'edit02': {
            'keypoint number': [1], 'delta': [[0.0, 0.0, 0.09]]
        },
        'edit12': {
            'keypoint number': [7, 8], 'delta': [[0.0, 0.05, 0.0]]
        }},
    'test4': {
        'edit01': {
            'keypoint number': [4], 'delta': [[-0.08, 0.0, 0.0]]
        },
        'edit02': {
            'keypoint number': [3], 'delta': [[0.0, 0.0, -0.08]]
        }},
    'test5': {
        'edit11': {
            'keypoint number': [7, 12], 'delta': [[0.07, 0.0, 0.0]]
        }},
    'test7': {
        'edit01': {
            'keypoint number': [5], 'delta': [[0.07, 0.0, 0.0]]
        },
        'edit11': {
            'keypoint number': [11, 7, 12, 6, 8], 'delta': [[0.06, 0.0, 0.0]]
        },
        'edit12': {
            'keypoint number': [2, 1], 'delta': [[0.0, 0.0, 0.07], [0.0, 0.0, -0.07]]
        }},
    'test8': {
        'edit11': {
            'keypoint number': [5, 6], 'delta': [[-0.07, 0.0, 0.0]]
        },
        'edit12': {
            'keypoint number': [3, 4], 'delta': [[0.0, 0.0, -0.06], [0.0, 0.0, 0.06]]
        }},
    'test9': {
        'edit11': {
            'keypoint number': [10, 9], 'delta': [[0.06, 0.0, 0.0]]
        }},
}


def save_encode_info(path_, label='chairs'):
    dic_info = calc_mesh_points_normals(path_)
    if label == 'chairs':
        info_ = {'points': np.expand_dims(dic_info["points"], axis=0),
                 'normals': np.expand_dims(dic_info["normals"], axis=0),
                 'label': th.tensor([4]), 'model': ['test0'], 'category_name': ['chair'], 'category': ['03001627']}
    elif label == 'cars':
        info_ = {'points': np.expand_dims(dic_info["points"], axis=0),
                 'normals': np.expand_dims(dic_info["normals"], axis=0),
                 'label': th.tensor([3]), 'model': ['test0'], 'category_name': ['car'], 'category': ['02958343']}
    elif label == 'airplanes':
        info_ = {'points': np.expand_dims(dic_info["points"], axis=0),
                 'normals': np.expand_dims(dic_info["normals"], axis=0),
                 'label': th.tensor([0]), 'model': ['test0'], 'category_name': ['airplane'], 'category': ['02691156']}
    else:
        raise NotImplementedError("Can't handle this type!")
    return info_


def copy_files():
    path = 'datas/comparison/airplanes'
    path1 = os.path.join('/media/jing/J/SLIDE/pointnet2/deformation/airplanes')
    case = 2
    if case == 1:    # copy test.npz to SLIDE
        for name in os.listdir(path):
            if not name.startswith('test'):
                continue
            os.makedirs(os.path.join(path1, name), exist_ok=True)
            shutil.copy(os.path.join(path, name, 'test.npz'), os.path.join(path1, name, 'test.npz'))
    elif case == 2:  # copy reconstructed_pcd.npz from SLIDE
        for name in os.listdir(path1):
            if not name.startswith('test'):
                continue
            shutil.copy(os.path.join(path1, name, 'reconstructed_pcd.npz'),
                        os.path.join(path, name, 'reconstructed_pcd.npz'))


def save_feature_info():
    label_jing = 'airplanes'
    path = os.path.join('datas/comparison', label_jing)
    path_ = os.path.join('/media/jing/J/SLIDE/pointnet2/deformation', label_jing)
    if label_jing == "chairs":
        drag_info = drag_info_chair
    elif label_jing == "cars":
        drag_info = drag_info_car
    elif label_jing == "airplanes":
        drag_info = drag_info_airplane
    else:
        raise NotImplementedError("Unknown type!")
    for name in os.listdir(path):
        if not name.startswith('test'):
            continue
        for edit in os.listdir(os.path.join(path, name)):
            if not edit.startswith('edit'):
                continue
            number = drag_info[name][edit[:-4]]['keypoint number']
            delta = drag_info[name][edit[:-4]]['delta']
            encode = np.load(os.path.join(path, name, 'reconstructed_pcd.npz'))
            keypoint = encode['keypoint']
            if len(delta) == 1:
                keypoint[0, number, :] += np.array(delta)
            else:
                assert len(delta) == len(number)
                keypoint[0, number, :] += np.array(delta)
            mask = np.zeros((keypoint.shape[0], keypoint.shape[1]))
            mask[:, number] = 1
            info_ = {'points': keypoint, 'label': encode['label'], 'category_name': encode['category_name'],
                     'category': encode['category'], 'timing': np.ones(keypoint.shape[0]),
                     'keypoint_feature': encode['keypoint_feature'], 'keypoint_mask': mask}
            os.makedirs(os.path.join(path_, name, edit[:-4]), exist_ok=True)
            np.savez(os.path.join(os.path.join(path_, name, edit[:-4]), 'shapenet_psr_generated_data_16_pts.npz'), **info_)


def remove_color_mls():
    path = '/media/jing/J/NeuralMLS/log/deformation/airplanes'
    for name in os.listdir(path):
        test_name = os.path.join(path, name)
        for edit_name in os.listdir(test_name):
            if not edit_name.startswith('edit'):
                continue
            for mesh_name in os.listdir(os.path.join(test_name, edit_name)):
                if not mesh_name.startswith('test'):
                    continue
                mesh = o3d.io.read_triangle_mesh(os.path.join(
                    test_name, edit_name, mesh_name)).remove_degenerate_triangles()
                o3d.io.write_triangle_mesh(os.path.join(
                    test_name, edit_name, mesh_name), mesh, write_vertex_colors=False)


def ply2obj():
    path = '/media/jing/J/SLIDE/pointnet2/deformation/airplanes'
    save_name = 'airplane_00000'
    for name in os.listdir(path):
        test_name = os.path.join(path, name)
        for edit_name in os.listdir(test_name):
            if edit_name.startswith('edit'):
                save_path = os.path.join(test_name, edit_name, 'fix', 'mesh_reconstruction',
                                         'shapenet_psr_generated_data_2048_pts',
                                         'visualization_results_at_iteration_00000000_epoch_0000', 'reconstructed_mesh')
                mesh = o3d.io.read_triangle_mesh(os.path.join(save_path, save_name+'.ply'))
                new_mesh = o3d.geometry.TriangleMesh()
                new_mesh.vertices = mesh.vertices
                new_mesh.triangles = mesh.triangles
                o3d.io.write_triangle_mesh(os.path.join(save_path, save_name+'.obj'), new_mesh.subdivide_loop(
                    number_of_iterations=2).filter_smooth_simple(number_of_iterations=2).remove_degenerate_triangles())
                save_path = os.path.join(test_name, edit_name, 'nofix', 'mesh_reconstruction',
                                         'shapenet_psr_generated_data_2048_pts',
                                         'visualization_results_at_iteration_00000000_epoch_0000', 'reconstructed_mesh')
                mesh = o3d.io.read_triangle_mesh(os.path.join(save_path, save_name+'.ply'))
                new_mesh = o3d.geometry.TriangleMesh()
                new_mesh.vertices = mesh.vertices
                new_mesh.triangles = mesh.triangles
                o3d.io.write_triangle_mesh(os.path.join(save_path, save_name+'.obj'), new_mesh.subdivide_loop(
                    number_of_iterations=2).filter_smooth_simple(number_of_iterations=2).remove_degenerate_triangles())


def comparison_local_iou():
    label_jing = 'airplanes'
    path = os.path.join('datas/comparison', label_jing)
    if label_jing == 'chairs':
        drag_info = drag_info_chair
    elif label_jing == 'cars':
        drag_info = drag_info_car
    elif label_jing == 'airplanes':
        drag_info = drag_info_airplane
    else:
        raise NotImplementedError('Unknown type!')

    """Ours"""
    print('*******Ours Methods******')
    for name in os.listdir(path):
        if not name.startswith('test'):
            continue
        keypoints = np.load(os.path.join(path, name, 'reconstructed_pcd.npz'))['keypoint']  # 1*16*3
        for edit in os.listdir(os.path.join(path, name)):
            if not edit.startswith('edit'):
                continue
            s = calc_local_distance(mesh_a=os.path.join(path, name, 'test.obj'),
                                    mesh_b=os.path.join(path, name, edit),
                                    points_a=keypoints[0, drag_info[name][edit[:-4]]['keypoint number'], :].astype(
                                        np.float32),
                                    points_b=(keypoints[0, drag_info[name][edit[:-4]]['keypoint number'], :] + np.array(
                                        drag_info[name][edit[:-4]]['delta'])).astype(np.float32),
                                    r=0.1, point_num=200000)
            print(s)

    """Slide"""
    path1 = os.path.join('/media/jing/J/SLIDE/pointnet2/deformation', label_jing)
    if label_jing == 'chairs':
        mesh_name = 'chair_00000.obj'
    elif label_jing == 'cars':
        mesh_name = 'car_00000.obj'
    elif label_jing == 'airplanes':
        mesh_name = 'airplane_00000.obj'
    else:
        raise NotImplementedError('Unknown type!')
    s1, s2 = [], []
    for name in os.listdir(path1):
        if not name.startswith('test'):
            continue
        keypoints = np.load(os.path.join(path, name, 'reconstructed_pcd.npz'))['keypoint']  # 1*16*3
        for edit in os.listdir(os.path.join(path1, name)):
            if not edit.startswith('edit'):
                continue
            s1.append(calc_local_distance(mesh_a=os.path.join(path, name, 'test.obj'), mesh_b=os.path.join(path1, name, edit, 'fix/mesh_reconstruction/shapenet_psr_generated_data_2048_pts/visualization_results_at_iteration_00000000_epoch_0000/reconstructed_mesh', mesh_name),
                                          points_a=keypoints[0, drag_info[name][edit]['keypoint number'], :].astype(np.float32),
                                          points_b=(keypoints[0, drag_info[name][edit]['keypoint number'], :] + np.array(drag_info[name][edit]['delta'])).astype(np.float32),
                                          r=0.1, point_num=200000))
            s2.append(calc_local_distance(mesh_a=os.path.join(path, name, 'test.obj'),
                                          mesh_b=os.path.join(path1, name, edit, 'nofix/mesh_reconstruction/shapenet_psr_generated_data_2048_pts/visualization_results_at_iteration_00000000_epoch_0000/reconstructed_mesh', mesh_name),
                                          points_a=keypoints[0, drag_info[name][edit]['keypoint number'], :].astype(np.float32),
                                          points_b=(keypoints[0, drag_info[name][edit]['keypoint number'], :] + np.array(drag_info[name][edit]['delta'])).astype(np.float32),
                                          r=0.1, point_num=200000))
    print('*******Slide******')
    for i in s2:
        print(i)
    print('*******Slide(fix)******')
    for i in s1:
        print(i)

    """MLS"""
    print('*******NeuralMLS******')
    path2 = os.path.join('/media/jing/J/NeuralMLS/log/deformation', label_jing)
    for name in os.listdir(path2):
        if not name.startswith('test'):
            continue
        keypoints = np.load(os.path.join(path, name, 'reconstructed_pcd.npz'))['keypoint']  # 1*16*3
        for edit in os.listdir(os.path.join(path2, name)):
            if not edit.startswith('edit'):
                continue
            s = calc_local_distance(mesh_a=os.path.join(path, name, 'test.obj'),
                                    mesh_b=os.path.join(path2, name, edit, name+'-Sab.obj'),
                                    points_a=keypoints[0, drag_info[name][edit]['keypoint number'], :].astype(
                                        np.float32),
                                    points_b=(keypoints[0, drag_info[name][edit]['keypoint number'],
                                              :] + np.array(drag_info[name][edit]['delta'])).astype(np.float32),
                                    r=0.1, point_num=200000)
            print(s)


def gather_edit_results():
    label_jing = 'cars'

    """ours"""
    path = '/home/jing/Desktop/Ours'
    os.makedirs(path, exist_ok=True)
    path1 = os.path.join('/media/jing/J/DragShape/datas/comparison', label_jing)
    for name in os.listdir(path1):
        if not name.startswith('test'):
            continue
        for edit in os.listdir(os.path.join(path1, name)):
            if not edit.startswith('edit'):
                continue
            os.makedirs(os.path.join(path, name), exist_ok=True)
            shutil.copy(os.path.join(path1, name, edit), os.path.join(path, name, edit))
            shutil.copy(os.path.join(path1, name, 'test.obj'), os.path.join(path, name, 'test.obj'))

    """Slide"""
    path = '/home/jing/Desktop/Slide'
    os.makedirs(path, exist_ok=True)
    path1 = os.path.join('/media/jing/J/SLIDE/pointnet2/deformation', label_jing)
    if label_jing == 'chairs':
        mesh_name = 'chair_00000.obj'
    elif label_jing == 'cars':
        mesh_name = 'car_00000.obj'
    elif label_jing == 'airplanes':
        mesh_name = 'airplane_00000.obj'
    else:
        raise NotImplementedError('Unknown type')
    for name in os.listdir(path1):
        if not name.startswith('test'):
            continue
        for edit in os.listdir(os.path.join(path1, name)):
            if not edit.startswith('edit'):
                continue
            os.makedirs(os.path.join(path, name), exist_ok=True)
            shutil.copy(os.path.join(path1, name, edit,
                                     'fix/mesh_reconstruction/shapenet_psr_generated_data_2048_pts/visualization_results_at_iteration_00000000_epoch_0000/reconstructed_mesh', mesh_name),
                        os.path.join(path, name, 'fix' + edit + '.obj'))
            shutil.copy(os.path.join(path1, name, edit,
                                     'nofix/mesh_reconstruction/shapenet_psr_generated_data_2048_pts/visualization_results_at_iteration_00000000_epoch_0000/reconstructed_mesh', mesh_name),
                        os.path.join(path, name, 'nofix' + edit + '.obj'))

    """NeuralMLS"""
    path = '/home/jing/Desktop/NeuralMLS'
    os.makedirs(path, exist_ok=True)
    path1 = os.path.join('/media/jing/J/NeuralMLS/log/deformation', label_jing)
    for name in os.listdir(path1):
        if not name.startswith('test'):
            continue
        for edit in os.listdir(os.path.join(path1, name)):
            if not edit.startswith('edit'):
                continue
            os.makedirs(os.path.join(path, name), exist_ok=True)
            shutil.copy(os.path.join(path1, name, edit, name + '-Sab.obj'), os.path.join(path, name, edit + '.obj'))


def gather_bench3d():
    path = 'datas/comparison/chairs'
    bench_path = '/home/jing/Desktop/SupplementaryMaterials/DragBench3D/chairs'
    for name in os.listdir(path):
        if not name.startswith('test'):
            continue
        os.makedirs(os.path.join(bench_path, name), exist_ok=True)
        for drag in os.listdir(os.path.join(path, name)):
            if not drag.startswith('drag'):
                continue
            shutil.copy(os.path.join(path, name, drag), os.path.join(bench_path, name, drag))
        shutil.copy(os.path.join(path, name, 'test.obj'), os.path.join(bench_path, name, 'test.obj'))
        shutil.copy(os.path.join(path, name, 'reconstructed_pcd'), os.path.join(bench_path, name, 'reconstructed_pcd'))


def main():
    # 1: save encode information;  First Step
    # 2: cop files, test.npz, reconstructed_pcd.npz;  Second Step
    # 3: save feature;   Third Step
    # 4: remove vertices color for MLS method;
    # 5: convert ply format to obj format for SLIDE method;
    # 6: calculate comparison local IoU metric
    # 7: gather results which are send to Linux
    # 8:gather bench3D
    case = 8
    if case == 1:
        path = 'datas/comparison/airplanes'
        for i in range(1, 10):
            info = save_encode_info(path_=os.path.join(path, 'test%d' % i, 'test.obj'), label='airplanes')
            np.savez(os.path.join(path, 'test%d' % i, 'test.npz'), **info)
    elif case == 2:
        copy_files()
    elif case == 3:
        save_feature_info()
    elif case == 4:
        remove_color_mls()
    elif case == 5:
        ply2obj()
    elif case == 6:
        comparison_local_iou()
    elif case == 7:
        gather_edit_results()
    elif case == 8:
        gather_bench3d()


if __name__ == "__main__":
    main()
