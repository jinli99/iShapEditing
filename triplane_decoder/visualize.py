import argparse
import mcubes
from tqdm import tqdm
from .axisnetworks import *
from .dataset_3d import *
from matplotlib import pyplot as plt
device = torch.device('cuda')


def cross_section(model, obj_idx, res=512, max_batch_size=50000, axis='z'):
    # Output a cross section of the fitted volume

    xx = torch.linspace(-1, 1, res)
    yy = torch.linspace(-1, 1, res)
    # zz = torch.linspace(-1, 1, res)
    (x_coords, y_coords) = torch.meshgrid([xx, yy])
    z_coords = torch.zeros_like(x_coords)
    coords = torch.cat([x_coords.unsqueeze(-1), y_coords.unsqueeze(-1), z_coords.unsqueeze(-1)], -1)
    # coords = torch.cat([x_coords.unsqueeze(-1), z_coords.unsqueeze(-1), y_coords.unsqueeze(-1)], -1)

    coords = coords.reshape(res*res, 3)
    prediction = torch.zeros(coords.shape[0], 1)

    with torch.no_grad():
        head = 0
        while head < coords.shape[0]:
            prediction[head:head+max_batch_size] = model(obj_idx, coords[head:head+max_batch_size].to(device).unsqueeze(0)).cpu()
            head += max_batch_size
            
    prediction = (prediction > 0).cpu().numpy().astype(np.uint8)
    prediction = prediction.reshape(res, res)
    plt.figure(figsize=(16, 16))
    plt.imshow(prediction)


def create_obj(model, obj_idx, res=128, max_batch_size=50000, output_path='output.obj'):
    # Output a res * res * res * 1 volume prediction. Download ChimeraX to open the files.
    # Set the threshold in ChimeraX to 0.5 if mrc_mode=0, 0 else

    model.eval()
    xx = torch.linspace(-1, 1, res)
    yy = torch.linspace(-1, 1, res)
    zz = torch.linspace(-1, 1, res)

    (x_coords, y_coords, z_coords) = torch.meshgrid([xx, yy, zz], indexing='ij')
    coords = torch.cat([x_coords.unsqueeze(-1), y_coords.unsqueeze(-1), z_coords.unsqueeze(-1)], -1)

    coords = coords.reshape(res*res*res, 3)
    prediction = torch.zeros(coords.shape[0], 1)
    
    # with tqdm(total=coords.shape[0]) as pbar:
    #     with torch.no_grad():
    #         head = 0
    #         while head < coords.shape[0]:
    #             prediction[head:head+max_batch_size] = model(obj_idx, coords[head:head+max_batch_size].to(device).unsqueeze(0)).cpu()
    #             head += max_batch_size
    #             pbar.update(min(max_batch_size, coords.shape[0] - head))

    with torch.no_grad():
        head = 0
        while head < coords.shape[0]:
            prediction[head:head + max_batch_size] = model(obj_idx,
                                                           coords[head:head + max_batch_size].to(device).unsqueeze(
                                                               0)).cpu()
            head += max_batch_size
    
    prediction = prediction.reshape(res, res, res).cpu().detach().numpy()
    
    smoothed_prediction = prediction
    # np.save('occupancy.npy', smoothed_prediction)
    vertices, triangles = mcubes.marching_cubes(smoothed_prediction, 0)
    vertices = vertices / 255. * 2 - 1
    mcubes.export_obj(vertices, triangles, output_path)


def create_obj_o3d(model, obj_idx, res=128, max_batch_size=50000):
    import open3d as o3d
    model.eval()
    xx = torch.linspace(-1, 1, res)
    yy = torch.linspace(-1, 1, res)
    zz = torch.linspace(-1, 1, res)

    (x_coord, y_coord, z_coord) = torch.meshgrid([xx, yy, zz], indexing='ij')
    coord = torch.cat([x_coord.unsqueeze(-1), y_coord.unsqueeze(-1), z_coord.unsqueeze(-1)], -1)

    coord = coord.reshape(res * res * res, 3)
    prediction = torch.zeros(coord.shape[0], 1)

    with torch.no_grad():
        head = 0
        while head < coord.shape[0]:
            prediction[head:head + max_batch_size] = model(obj_idx,
                                                           coord[head:head + max_batch_size].to(device).unsqueeze(
                                                               0)).cpu()
            head += max_batch_size

    prediction = prediction.reshape(res, res, res).cpu().detach().numpy()

    smoothed_prediction = prediction
    vertices, triangles = mcubes.marching_cubes(smoothed_prediction, 0)
    vertices = vertices / res * 2 - 1
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh


def main(args=None, feature=None):
    if args is not None:
        args = args
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--input', type=str)
        parser.add_argument('--output', type=str, required=True)
        parser.add_argument('--model_path', type=str, default='models/epoch_24_decoder_loss=25.37570571899414.pt', required=False)
        parser.add_argument('--res', type=int, default='128', required=False)

        args = parser.parse_args()
    model = MultiTriplane(1, input_dim=3, output_dim=1).to(device)
    model.net.load_state_dict(torch.load(args.model_path))
    model.eval()
    triplanes = np.load(args.input).reshape(3, 32, 128, 128) if feature is None else feature

    with torch.no_grad():
        for i in range(3):
            model.embeddings[i] = torch.tensor(triplanes[[i]]).to(device)

    create_obj(model, 0, res=args.res, output_path=args.output)  # res = 256
    

