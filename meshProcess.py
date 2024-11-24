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
def calc_chamfer(mesh_a, mesh_b, point_num):

    if isinstance(mesh_a, str):
        mesh_a = o3d.io.read_triangle_mesh(mesh_a)
    if isinstance(mesh_b, str):
        mesh_b = o3d.io.read_triangle_mesh(mesh_b)
    points_a = np.asarray(mesh_a.sample_points_uniformly(point_num).points, dtype=np.float32)
    points_b = np.asarray(mesh_b.sample_points_uniformly(point_num).points, dtype=np.float32)

    kdtree_a = cKDTree(points_a)
    dist_a, _ = kdtree_a.query(points_b)
    chamfer_a = np.mean(np.square(dist_a))

    kdtree_b = cKDTree(points_b)
    dist_b, _ = kdtree_b.query(points_a)
    chamfer_b = np.mean(np.square(dist_b))

    return chamfer_a+chamfer_b


# calculate the hausdorff distance between two meshes
def calc_hausdorff(mesh_a, mesh_b, point_num):

    if isinstance(mesh_a, str):
        mesh_a = o3d.io.read_triangle_mesh(mesh_a)
    if isinstance(mesh_b, str):
        mesh_b = o3d.io.read_triangle_mesh(mesh_b)
    points_a = np.asarray(mesh_a.sample_points_uniformly(point_num).points, dtype=np.float32)
    points_b = np.asarray(mesh_b.sample_points_uniformly(point_num).points, dtype=np.float32)

    kdtree_a = cKDTree(points_a)
    dist_a, _ = kdtree_a.query(points_b)
    hausdorff_a = np.max(dist_a)

    kdtree_b = cKDTree(points_b)
    dist_b, _ = kdtree_b.query(points_a)
    hausdorff_b = np.max(dist_b)

    return max(hausdorff_a, hausdorff_b)


def calc_iou(mesh_a, mesh_b, point_num):

    if isinstance(mesh_a, str):
        mesh_a = o3d.io.read_triangle_mesh(mesh_a)
    if isinstance(mesh_b, str):
        mesh_b = o3d.io.read_triangle_mesh(mesh_b)

    uniform_points = (np.random.rand(int(point_num*0.2), 3) * 2 - 1).astype(dtype=np.float32)
    points_a = np.asarray(mesh_a.sample_points_uniformly(int(point_num*0.4)).points, dtype=np.float32)
    points_a += 0.01 * np.random.randn(points_a.shape[0], 3)
    points_b = np.asarray(mesh_b.sample_points_uniformly(int(point_num * 0.4)).points, dtype=np.float32)
    points_b += 0.01 * np.random.randn(points_b.shape[0], 3)
    iou_points = np.concatenate([uniform_points, points_a, points_b], axis=0)
    occ_a = calc_implicit_field(mesh_a, points=iou_points).reshape(-1)
    occ_b = calc_implicit_field(mesh_b, points=iou_points).reshape(-1)
    occ_a = (occ_a < 0)
    occ_b = (occ_b < 0)

    return (occ_a & occ_b).astype(np.float32).sum(axis=-1) / (occ_a | occ_b).astype(np.float32).sum(axis=-1)


def calc_local_distance(mesh_a, mesh_b, points_a, points_b, r, point_num, metric='IoU'):

    if points_a.shape != points_b.shape:    # expect to be num*3
        raise ArithmeticError("The 'points_a' and 'points_b' should have the same shape!")

    if isinstance(mesh_a, str):
        mesh_a = o3d.io.read_triangle_mesh(mesh_a)
    if isinstance(mesh_b, str):
        mesh_b = o3d.io.read_triangle_mesh(mesh_b)
    points = (np.random.rand(point_num, 3) * 2 - 1).astype(dtype=np.float32) * r  # num*3, [-r, r]^3
    distance = 0

    for i in range(points_a.shape[0]):
        occ_source = calc_implicit_field(mesh_a, points=points+points_a[i]).reshape(-1)
        occ_target = calc_implicit_field(mesh_b, points=points+points_b[i]).reshape(-1)
        if metric == 'IoU':
            occ_source = occ_source < 0
            occ_target = occ_target < 0
            distance += (occ_source & occ_target).astype(np.float32).sum(axis=-1) / (
                    occ_source | occ_target).astype(np.float32).sum(axis=-1)
        elif metric == 'L2':
            distance += ((occ_target - occ_source) ** 2).mean()
        elif metric == 'CD':
            pass

    return distance/points_a.shape[0]


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


def cloud2mesh(path):
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


# crop images with the same region
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
    crop_image = image_origin[:, top_bound:bottom_bound+1, left_bound:right_bound+1, :]
    os.makedirs(os.path.join(path, 'Resize'))

    for i in range(crop_image.shape[0]):
        io.imsave(os.path.join(path, 'Resize', path_list[i]), crop_image[i])


# crop each single image by bbox
def crop_images_rgba_each(path, save_format='png'):
    from skimage import io

    for name in os.listdir(path):
        if not name.endswith(('.png', '.jpg')):
            continue
        image_origin = io.imread(os.path.join(path, name))  # h*w*4
        image_alpha = image_origin[..., -1].copy()  # h*w
        image_alpha[image_alpha > 0] = 1
        image_alpha_x = image_alpha.max(0).astype(np.bool_)  # w
        left_bound = np.argmax(image_alpha_x)  # (n, )
        right_bound = image_alpha_x.shape[0] - 1 - np.argmax(image_alpha_x[::-1])
        image_alpha_y = image_alpha.max(1).astype(np.bool_)  # h
        top_bound = np.argmax(image_alpha_y)  # (n, )
        bottom_bound = image_alpha_y.shape[0] - 1 - np.argmax(image_alpha_y[::-1])

        if save_format == 'png':
            os.makedirs(os.path.join(path, 'png'), exist_ok=True)
            io.imsave(os.path.join(
                path, 'png', name), image_origin[top_bound:bottom_bound + 1, left_bound:right_bound + 1, :])
        elif save_format == 'jpg':
            from PIL import Image
            os.makedirs(os.path.join(path, 'jpg'), exist_ok=True)
            image = Image.fromarray(image_origin[top_bound:bottom_bound + 1, left_bound:right_bound + 1])
            white_background = Image.new('RGB', image.size, (255, 255, 255))
            white_background.paste(image, mask=image.split()[3])
            white_background.convert('RGB').save(os.path.join(path, 'jpg', name).replace('.png', '.jpg'), 'JPEG')
        else:
            raise NotImplementedError("Unsupported image format")


def down_sample(path):
    from PIL import Image
    for i in os.listdir(path):
        image = Image.open(os.path.join(path, i))
        size = image.size
        resized_image = image.resize((int(size[0]*0.5), int(size[1]*0.5)))
        os.makedirs(os.path.join(path, 'DownSample'), exist_ok=True)
        resized_image.save(os.path.join(path, 'DownSample', i))


def arap(mesh, static_ids, handle_ids, handle_pos):

    vertices = np.asarray(mesh.vertices)
    static_pos = []
    for id in static_ids:
        static_pos.append(vertices[id])
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

        white_background.paste(image, mask=image.split()[3])

        white_background.convert('RGB').save(os.path.join(path, name).replace('.png', '.jpg'), 'JPEG')

