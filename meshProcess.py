import os.path
from scipy.spatial import cKDTree
import open3d as o3d
import numpy as np


def calc_implicit_field(mesh, points, sdf=True):
    scene = o3d.t.geometry.RaycastingScene()
    mesh = o3d.t.geometry.TriangleMesh().from_legacy(mesh_legacy=mesh)
    _ = scene.add_triangles(mesh)
    if sdf:
        return (scene.compute_signed_distance(points)).numpy()
    else:
        return (scene.compute_occupancy(points)).numpy()


# calculate the chamfer distance between two meshes
def calc_chamfer(filename_gt, filename_pre, point_num):

    mesh_a = o3d.io.read_triangle_mesh(filename_gt)
    points_a = np.asarray(mesh_a.sample_points_uniformly(point_num).points, dtype=np.float32)
    mesh_b = o3d.io.read_triangle_mesh(filename_pre)
    points_b = np.asarray(mesh_b.sample_points_uniformly(point_num).points, dtype=np.float32)

    kdtree_a = cKDTree(points_a)
    dist_a, _ = kdtree_a.query(points_b)
    chamfer_a = np.mean(np.square(dist_a))

    kdtree_b = cKDTree(points_b)
    dist_b, _ = kdtree_b.query(points_a)
    chamfer_b = np.mean(np.square(dist_b))

    return chamfer_a+chamfer_b


# calculate the hausdorff distance between two meshes
def calc_hausdorff(filename_gt, filename_pre, point_num):
    # scale = 1.0e5  # scale the result for better display

    mesh_a = o3d.io.read_triangle_mesh(filename_gt)
    points_a = np.asarray(mesh_a.sample_points_uniformly(point_num).points, dtype=np.float32)
    mesh_b = o3d.io.read_triangle_mesh(filename_pre)
    points_b = np.asarray(mesh_b.sample_points_uniformly(point_num).points, dtype=np.float32)

    kdtree_a = cKDTree(points_a)
    dist_a, _ = kdtree_a.query(points_b)
    hausdorff_a = np.max(dist_a)

    kdtree_b = cKDTree(points_b)
    dist_b, _ = kdtree_b.query(points_a)
    hausdorff_b = np.max(dist_b)

    return max(hausdorff_a, hausdorff_b)


def calc_iou(filename_gt, filename_pre, point_num):
    mesh_a = o3d.io.read_triangle_mesh(filename_gt)
    mesh_b = o3d.io.read_triangle_mesh(filename_pre)
    uniform_points = (np.random.rand(int(point_num*0.2), 3) * 2 - 1).astype(dtype=np.float32)
    mesh_points_a = np.asarray(mesh_a.sample_points_uniformly(int(point_num*0.4)).points, dtype=np.float32)
    mesh_points_a += 0.01 * np.random.randn(mesh_points_a.shape[0], 3)
    mesh_points_b = np.asarray(mesh_b.sample_points_uniformly(int(point_num * 0.4)).points, dtype=np.float32)
    mesh_points_b += 0.01 * np.random.randn(mesh_points_b.shape[0], 3)
    iou_points = np.concatenate([uniform_points, mesh_points_a, mesh_points_b], axis=0)
    occ_a = calc_implicit_field(mesh_a, points=iou_points).reshape(-1)
    occ_b = calc_implicit_field(mesh_b, points=iou_points).reshape(-1)
    occ_a = (occ_a < 0)
    occ_b = (occ_b < 0)
    return (occ_a & occ_b).astype(np.float32).sum(axis=-1) / (occ_a | occ_b).astype(np.float32).sum(axis=-1)


def calc_distance_error(filename_source, filename_target, pnt_source, pnt_target, r, point_num):
    mesh_source = o3d.io.read_triangle_mesh(filename_source)
    mesh_target = o3d.io.read_triangle_mesh(filename_target)
    points = (np.random.rand(point_num, 3) * 2 - 1).astype(dtype=np.float32) * r  # num*3, [-r, r]^3
    distance_iou, distance_chamfer = 0, 0
    for i in range(pnt_source.shape[0]):
        source = points+pnt_source[i]
        target = points+pnt_target[i]
        occ_source = calc_implicit_field(mesh_source, points=source).reshape(-1)
        occ_target = calc_implicit_field(mesh_target, points=target).reshape(-1)

        # L2 distance
        #distance += ((occ_target - occ_source)**2).mean()

        # IoU metric
        occ_source = occ_source < 0
        occ_target = occ_target < 0
        distance_iou += (occ_source & occ_target).astype(np.float32).sum(axis=-1) / (occ_source | occ_target).astype(np.float32).sum(axis=-1)

        # Chamfer Distance
        # source = source[occ_source]
        # target = target[occ_target]
        # kdtree_a = cKDTree(source)
        # dist_a, _ = kdtree_a.query(target)
        # chamfer_a = np.mean(np.square(dist_a))
        # kdtree_b = cKDTree(target)
        # dist_b, _ = kdtree_b.query(source)
        # chamfer_b = np.mean(np.square(dist_b))
        # distance_chamfer += chamfer_a + chamfer_b

    return distance_iou/pnt_source.shape[0]


def calc_mesh_points_normals(mesh, pcd=None):
    if isinstance(mesh, str):
        mesh = o3d.io.read_triangle_mesh(mesh)
    if pcd is None:
        pcd = mesh.sample_points_uniformly(number_of_points=2048)
    pnt = np.asarray(pcd.points).astype(np.float32)
    mesh.compute_triangle_normals()
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    ans = scene.compute_closest_points(o3d.core.Tensor(pnt, dtype=o3d.core.Dtype.Float32))
    return {"points": pnt, "normals": ans['primitive_normals'].numpy().astype(np.float32)}


def cloud2mesh():
    path = '/media/jing/I/ShapeNet_Preprocess'
    for name in os.listdir(path):
        for file in os.listdir(os.path.join(path, name)):
            file_path = os.path.join(path, name, file)
            if os.path.exists(os.path.join(file_path, "mesh_origin.obj")):
                continue
            try:
                pointcloud = np.load(os.path.join(file_path, "pointcloud.npz"))
            except FileNotFoundError:
                print("***************************No such file or directory:", file_path)
                continue
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointcloud["points"])
            pcd.normals = o3d.utility.Vector3dVector(pointcloud["normals"])
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)[0]  # poisson reconstruction

            # translate and scale
            new_mesh = o3d.geometry.TriangleMesh()
            new_mesh.vertices = mesh.vertices
            new_mesh.triangles = mesh.triangles
            o3d.io.write_triangle_mesh(os.path.join(file_path, "mesh_origin.obj"), new_mesh)
            max_bound, min_bound = new_mesh.get_max_bound(), new_mesh.get_min_bound()
            axis_extent = max_bound - min_bound
            new_mesh.translate(-new_mesh.get_center())
            new_mesh.scale(2. / (axis_extent.max() + 0.01), center=new_mesh.get_center())
            v = np.asarray(new_mesh.vertices)
            v -= (v.max(0) + v.min(0)) * 0.5
            new_mesh.vertices = o3d.utility.Vector3dVector(v)
            o3d.io.write_triangle_mesh(os.path.join(file_path, "mesh_scale.obj"), new_mesh)

            # post process
            new_mesh = (new_mesh.filter_smooth_simple(number_of_iterations=10)).remove_degenerate_triangles()
            o3d.io.write_triangle_mesh(os.path.join(file_path, "mesh_scale_smooth.obj"), new_mesh)
            print(file_path, "Done!")
            max_bound, min_bound = new_mesh.get_max_bound(), new_mesh.get_min_bound()
            if np.any(min_bound > 1) or np.any(min_bound < -1) or np.any(max_bound > 1) or np.any(max_bound < -1):
                print("****************************", file_path)


def crop_images_rgb(path):
    # from skimage import io
    # image_origin = np.stack([io.imread(image) for image in path], axis=0)
    # image = image_origin.mean(-1)
    # print(image_origin.shape, image.shape)
    # image[image < 255] = 1
    # image[image > 1] = 0
    # image = image.max(0)
    # axis_x = np.nonzero(image.max(0))
    # left_bound, right_bound = axis_x[0][0], axis_x[0][-1]
    # axis_y = np.nonzero(image.max(1))
    # top_bound, bottom_bound = axis_y[0][0], axis_y[0][-1]
    # crop_image = image_origin[:, top_bound:bottom_bound+1, left_bound:right_bound+1, :]
    # for i in range(crop_image.shape[0]):
    #     io.imsave('%d.png'%i, crop_image[i])

    from skimage import io
    path_list = os.listdir(path)
    image_origin = np.stack([io.imread(os.path.join(path, image)) for image in path_list], axis=0)  # n*h*w*3
    top_bound, bottom_bound, left_bound, right_bound = 86, 996, 148, 824
    crop_image = image_origin[:, top_bound:bottom_bound+1, left_bound:right_bound+1, :]
    os.makedirs(os.path.join(path, 'Resize'))
    for i in range(crop_image.shape[0]):
        io.imsave(os.path.join(path, 'Resize', path_list[i]), crop_image[i])


def crop_images_rgba(path):
    from skimage import io
    path_list = os.listdir(path)
    image_origin = np.stack([io.imread(os.path.join(path, image)) for image in path_list], axis=0)  # n*h*w*4
    image_alpha = image_origin[..., -1].copy()  # n*h*w
    image_alpha[image_alpha > 0] = 1
    image_alpha = image_alpha.max(0)  # h*w
    axis_x = np.nonzero(image_alpha.max(0))
    left_bound, right_bound = axis_x[0][0], axis_x[0][-1]
    axis_y = np.nonzero(image_alpha.max(1))
    top_bound, bottom_bound = axis_y[0][0], axis_y[0][-1]
    # print(top_bound, bottom_bound, left_bound, right_bound)
    # return
    crop_image = image_origin[:, top_bound:bottom_bound+1, left_bound:right_bound+1, :]
    os.makedirs(os.path.join(path, 'Resize'))
    for i in range(crop_image.shape[0]):
        io.imsave(os.path.join(path, 'Resize', path_list[i]), crop_image[i])


def crop_images_rgba_each(path):
    # from skimage import io
    # path_list = os.listdir(path)
    # image_origin = np.stack([io.imread(os.path.join(path, image)) for image in path_list], axis=0)  # n*h*w*4
    # image_alpha = image_origin[..., -1].copy()  # n*h*w
    # image_alpha[image_alpha > 0] = 1
    # image_alpha_x = image_alpha.max(1).astype(np.bool_)  # n*w
    # left_bound = np.argmax(image_alpha_x, axis=1)  # (n, )
    # right_bound = image_alpha_x.shape[1] - 1 - np.argmax(image_alpha_x[::-1], axis=1)  # (n, )
    # image_alpha_y = image_alpha.max(2).astype(np.bool_)  # n*h
    # top_bound = np.argmax(image_alpha_y, axis=1)  # (n, )
    # bottom_bound = image_alpha_y.shape[1] - 1 - np.argmax(image_alpha_y[::-1], axis=1)  # (n, )
    # os.makedirs(os.path.join(path, 'Resize1'))
    # print(left_bound, right_bound, top_bound, bottom_bound)
    # for i in range(image_alpha.shape[0]):
    #     io.imsave(os.path.join(path, 'Resize1', path_list[i]), image_origin[i][top_bound[i]:bottom_bound[i]+1, left_bound[i]:right_bound[i]+1, :])

    from skimage import io
    for name in os.listdir(path):
        image_origin = io.imread(os.path.join(path, name))  # h*w*4
        image_alpha = image_origin[..., -1].copy()  # h*w
        image_alpha[image_alpha > 0] = 1
        image_alpha_x = image_alpha.max(0).astype(np.bool_)  # w
        left_bound = np.argmax(image_alpha_x)  # (n, )
        right_bound = image_alpha_x.shape[0] - 1 - np.argmax(image_alpha_x[::-1])
        image_alpha_y = image_alpha.max(1).astype(np.bool_)  # h
        top_bound = np.argmax(image_alpha_y)  # (n, )
        bottom_bound = image_alpha_y.shape[0] - 1 - np.argmax(image_alpha_y[::-1])
        os.makedirs(os.path.join(path, 'Resize'), exist_ok=True)
        io.imsave(os.path.join(path, 'Resize', name), image_origin[top_bound:bottom_bound + 1, left_bound:right_bound + 1, :])


def crop_rgba_jpg(path):
    from skimage import io
    from PIL import Image
    for name in os.listdir(path):
        # if not name.startswith('test7-12'):
        #     continue
        image_origin = io.imread(os.path.join(path, name))  # h*w*4
        image_alpha = image_origin[..., -1].copy()  # h*w
        image_alpha[image_alpha > 0] = 1
        image_alpha_x = image_alpha.max(0).astype(np.bool_)  # w
        left_bound = np.argmax(image_alpha_x)  # (n, )
        right_bound = image_alpha_x.shape[0] - 1 - np.argmax(image_alpha_x[::-1])
        image_alpha_y = image_alpha.max(1).astype(np.bool_)  # h
        top_bound = np.argmax(image_alpha_y)  # (n, )
        bottom_bound = image_alpha_y.shape[0] - 1 - np.argmax(image_alpha_y[::-1])
        os.makedirs(os.path.join(path, 'jpg'), exist_ok=True)
        image = Image.fromarray(image_origin[top_bound:bottom_bound + 1, left_bound:right_bound + 1])
        white_background = Image.new('RGB', image.size, (255, 255, 255))
        white_background.paste(image, mask=image.split()[3])
        white_background.convert('RGB').save(os.path.join(path, 'jpg', name).replace('.png', '.jpg'), 'JPEG')


def down_sample(path):
    from PIL import Image
    for i in os.listdir(path):
        image = Image.open(os.path.join(path, i))
        size = image.size
        resized_image = image.resize((int(size[0]*0.5), int(size[1]*0.5)))
        os.makedirs(os.path.join(path, 'DownSample'), exist_ok=True)
        resized_image.save(os.path.join(path, 'DownSample', i))


def calc_sphere_center():

    # x = [0, 0.132107, 0.1101, 0.143729, 0.128163]
    # y = [0, -0.191867, -0.189824, -0.204831, -0.217933]
    # z = [0, 0.707387, 0.715432, 0.688176, 0.698755]

    x = [0, 0.112461, 0.124516, 0.136056, 0.0868335]
    y = [0, -0.307764, -0.308694, -0.318525, -0.296157]
    z = [0, 0.725442, 0.720488, 0.713705, 0.709968]

    a, b, c = x[1] - x[2], y[1] - y[2], z[1] - z[2]
    a1, b1, c1 = x[3] - x[4], y[3] - y[4], z[3] - z[4]
    a2, b2, c2 = x[2] - x[3], y[2] - y[3], z[2] - z[3]
    P = 0.5*(x[1]**2 - x[2]**2 + y[1]**2 - y[2]**2 + z[1]**2 - z[2]**2)
    Q = 0.5*(x[3]**2 - x[4]**2 + y[3]**2 - y[4]**2 + z[3]**2 - z[4]**2)
    R = 0.5*(x[2]**2 - x[3]**2 + y[2]**2 - y[3]**2 + z[2]**2 - z[3]**2)
    D = np.linalg.det(np.array([[a, b, c], [a1, b1, c1], [a2, b2, c2]]))
    cx = np.linalg.det(np.array([[P, b, c], [Q, b1, c1], [R, b2, c2]])) / D
    cy = np.linalg.det(np.array([[a, P, c], [a1, Q, c1], [a2, R, c2]])) / D
    cz = np.linalg.det(np.array([[a, b, P], [a1, b1, Q], [a2, b2, R]])) / D
    print(cx, cy, cz)


def arap():
    mesh = o3d.io.read_triangle_mesh('./datas/generated-shape-edit/chairs/test3.obj')

    vertices = np.asarray(mesh.vertices)
    static_ids = [151956, 139344, 35623, 16874]
    static_pos = []
    for id in static_ids:
        static_pos.append(vertices[id])
    handle_ids = [np.argmin(((vertices - np.array([0.5774046641377827, -0.19996981418962428, 0.025786472652074535]))**2).sum(-1))]
    handle_pos = [np.array([0.5867314028280973, -0.049477389598448385, 0.018280856473317588])]
    constraint_ids = o3d.utility.IntVector(static_ids + handle_ids)
    constraint_pos = o3d.utility.Vector3dVector(static_pos + handle_pos)

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh_prime = mesh.deform_as_rigid_as_possible(constraint_ids,
                                                      constraint_pos,
                                                      max_iter=50)
        o3d.io.write_triangle_mesh('1.obj', mesh_prime)


def png_to_jpg(path):
    from PIL import Image
    for name in os.listdir(path):
        if not name.endswith('.png'):
            continue
        image = Image.open(os.path.join(path, name))
        # rgb_image = image.convert('RGB')
        # rgb_image.save(os.path.join(path, name[:-4]+'.jpg'), 'JPEG')  #

        white_background = Image.new('RGB', image.size, (255, 255, 255))
        # 将PNG图像粘贴到白色背景图像上
        white_background.paste(image, mask=image.split()[3])
        # 将图像转换为JPEG格式并保存
        white_background.convert('RGB').save(os.path.join(path, name).replace('.png', '.jpg'), 'JPEG')


def vis_tri_feat():
    from skimage import io
    name = ['tri_feat', 'tri_feat_opt', 'img120']
    path = 'datas/ablation/opt-based/test1'
    rgb = np.array([0.8, 0.386, 0.606])
    for i in name:
        x = np.load(os.path.join(path, i + '.npy')).reshape(3, 32, 128, 128)  # 1*96*128*128
        x = x.mean(1)  # 3*128*128
        x = (x - x.min()) / (x.max() - x.min()) * 255
        for j in range(3):
            y = np.around(np.stack([x[j] * rgb[0], x[j] * rgb[1], x[j] * rgb[2]], axis=-1)).astype(np.uint8)
            io.imsave(os.path.join(path, i + '_%d_rgb.png'%j), y)


def main():
    # crop_images_rgba_each(path='Supp/Generated Shape Edit/cars')
    # png_to_jpg('Supp/Generated Shape Edit/chairs/Resize')
    crop_rgba_jpg(path='Supp/Comparison/airplanes')
    # down_sample('pics/ablation/limitation/Resize')
    # crop_images_rgba(path='pics/ablation/opt-based')
    # crop_images_rgb(path='pics/ablation/jpg')

    # path = "./datas/layer-guidance/cars"
    # err = np.random.rand(10, 4)
    # for i in range(10, 20):
    #     gt_filename = os.path.join(path, str(i), 'origin.obj')
    #     for j in range(7, 11):
    #         err[i-10, j-7] = calc_iou(gt_filename, filename_pre=os.path.join(path, str(i), 'layer%d.obj'%j), point_num=200000)
    # for j in range(4):
    #     print(j+7, err[:, j].tolist())
    # calc_sphere_center()
    # arap()
    # vis_tri_feat()


if __name__ == "__main__":
    main()