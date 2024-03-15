import argparse
import copy
import time
from argparse import Namespace
import os

from triplane_decoder.visualize import create_obj_o3d
from neural_field_diffusion.guided_diffusion import dist_util
from neural_field_diffusion.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)
import torch as th
from torch.utils.data import Dataset
import torch.nn.functional as ff
from tqdm import tqdm
# from occupancy_field.intersector import check_mesh_contains
from neural_field_diffusion.guided_diffusion.gaussian_diffusion import _extract_into_tensor
from meshProcess import calc_implicit_field
import open3d as o3d
import numpy as np
from triplane_decoder.axisnetworks import MultiTriplane


def get_args():
    parser = argparse.ArgumentParser(description='Generate a set of triplane and their corresponding meshes')
    parser.add_argument('--resolution', type=str, default=128, required=False,
                        help='Triplane resolution.')
    parser.add_argument('--num_steps', type=int, default=200,
                        help='Number of steps to take in denoise process.', required=False)
    parser.add_argument('--shape_resolution', type=int, default=256,
                        help='Resolution at which to decode shapes.', required=False)
    parser.add_argument('--w_time', type=int, default=150,
                        help='Start time at optimization.', required=False)
    parser.add_argument('--feat_layer', type=int, default=8,
                        help='Index of features layer(7-9).', required=False)
    parser.add_argument('--loss_type', type=str, default='l2')
    parser.add_argument('--points_size', type=int, default=200000,
                        help='Size of training points.')
    parser.add_argument('--points_uniform_ratio', type=float, default=0.5,
                        help='Ratio of points to sample uniformly in bounding box.')

    args = parser.parse_args()

    # Generate triplane samples using DDPM with default arguments
    ddpm_args = Namespace(
        clip_denoised=True, num_samples=1, batch_size=1, use_ddim=False,
        model_path=None, stats_dir=None, num_steps=args.num_steps,
        explicit_normalization=True, save_dir=None, save_intermediate=False, save_timestep_interval=20,
        image_size=args.resolution, num_channels=256,
        num_res_blocks=2, num_heads=4, num_heads_upsample=-1, num_head_channels=64, attention_resolutions='32,16,8',
        channel_mult='', dropout=0.1, class_cond=False, shape_resolution=args.shape_resolution,
        use_checkpoint=False, use_scale_shift_norm=True, resblock_updown=True, use_fp16=True,
        use_new_attention_order=False, in_out_channels=96, learn_sigma=True,
        diffusion_steps=1000, noise_schedule='linear', timestep_respacing=str(args.num_steps),
        w_time=args.w_time, feat_layer=args.feat_layer, points_size=args.points_size,
        points_uniform_ratio=args.points_uniform_ratio, loss_type=args.loss_type, use_kl=False, predict_xstart=False,
        rescale_timesteps=False, decoder_ckpt=None, rescale_learned_sigmas=False
    )
    return ddpm_args


def synthesize_latent(model, diffusion, args=None, t1=None, t2=0, inter_latent_idx=None, inter_feat_idx=None,
                      img=None, calc_grad=False, **kwargs):
    """

    :param inter_feat_idx:
    :param inter_latent_idx:
    :param model:
    :param diffusion:
    :param args:
    :param t1:
    :param t2:
    :param img: intermediate noise after t1 steps
    :param calc_grad:
    :param kwargs:
    :return:
    """

    if args is None:
        args = get_args()
    shape = (args.batch_size, 96, args.image_size, args.image_size)
    if img is None:
        img = th.randn(shape, device=next(model.parameters()).device)
        if calc_grad:
            img.requires_grad_(True)
    assert img.shape == shape
    if t1 is None:
        t1 = args.num_steps
    elif t1 == 0:
        return {"img": img[:args.num_samples], "inter_latent": [], "inter_feat": [], "pred_xstart": [],
                "model_output": None}
    model_kwargs = {}
    sample_fun = diffusion.ddim_sample if args.use_ddim else diffusion.p_sample_guidance
    inter_latent = []
    inter_feat = []
    predict_x0 = []
    model_output = None
    variance = []
    noise = []
    if calc_grad:
        for i in range(t1 - 1, t2 - 1, -1):
            t = th.tensor([i] * shape[0], device=next(model.parameters()).device)
            out = sample_fun(model, img, t, model_kwargs=model_kwargs, **kwargs)
            if inter_feat_idx is not None and i in inter_feat_idx:
                img, inter_feat_ = out["sample"], out["inter_feat"]
                inter_feat.append(inter_feat_)
                noise.append(out["noise"].cpu())
                variance.append(out["variance"].cpu())
            else:
                img = out["sample"]
            if inter_latent_idx is not None and i in inter_latent_idx:
                inter_latent.append(img)
                predict_x0.append(out["pred_xstart"])
            model_output = out["model_output"]
    else:
        with th.no_grad():
            for i in range(t1 - 1, t2 - 1, -1):
                t = th.tensor([i] * shape[0], device=next(model.parameters()).device)
                out = sample_fun(model, img, t, model_kwargs=model_kwargs, **kwargs)
                if inter_feat_idx is not None and i in inter_feat_idx:
                    img, inter_feat_ = out["sample"], out["inter_feat"]
                    inter_feat.append(inter_feat_)
                    # noise.append(out["noise"].cpu())
                    # variance.append(out["variance"].cpu())
                else:
                    img = out["sample"]
                if inter_latent_idx is not None and i in inter_latent_idx:
                    inter_latent.append(img)
                    predict_x0.append(out["pred_xstart"])
                model_output = out["model_output"]
    return {"img": img[:args.num_samples], "inter_latent": inter_latent, "inter_feat": inter_feat,
            "pred_xstart": predict_x0, "model_output": model_output, "variance": variance, "noise": noise}


def make_offsets(r, device):
    p = th.arange(-r, r + 1, device=device)
    px, py, pz = th.meshgrid(p, p, p, indexing='ij')
    offsets = th.stack([px.reshape(-1), py.reshape(-1), pz.reshape(-1)], dim=-1)
    return offsets


def resize_feat_align(feature, cat_var=True):
    batch_num, channel_num = feature.shape[:2]
    assert not channel_num % 2
    channel_num = int(channel_num / 2)
    feature_mean, feature_var = th.split(feature, channel_num, dim=1)
    if channel_num % 3:
        expect_num = channel_num - channel_num % 3
        feature_mean = feature_mean.permute(2, 3, 0, 1)
        feature_mean = th.nn.functional.interpolate(feature_mean, (batch_num, expect_num)).permute(2, 3, 0, 1)
        feature_var = feature_var.permute(2, 3, 0, 1)
        feature_var = th.nn.functional.interpolate(feature_var, (batch_num, expect_num)).permute(2, 3, 0, 1)

    # 3*channel*img_size*img_size
    if cat_var:
        return th.cat((
            feature_mean.reshape(3, -1, feature_mean.shape[2], feature_mean.shape[3]),
            feature_var.reshape(3, -1, feature_mean.shape[2], feature_mean.shape[3])), dim=1).type(th.float32)
    else:
        return feature_mean.reshape(3, -1, feature_mean.shape[2], feature_mean.shape[3]).type(th.float32)


class OccupancyDatas(Dataset):
    def __init__(self, points, occupancies):
        self.data = np.concatenate([points, occupancies], axis=-1).astype(np.float32)

    def __getitem__(self, index):
        return self.data[index, :-1], self.data[index, [-1]]

    def __len__(self):
        return self.data.shape[0]


# classifier guidance
class DragStuff:

    args = get_args()

    def __init__(self):

        # create model and diffusion
        dist_util.setup_dist()
        self.device = dist_util.dev()
        self.model, self.diffusion = create_model_and_diffusion(
            **args_to_dict(self.args, model_and_diffusion_defaults().keys())
        )
        self.model.to(self.device)
        self.model.eval()
        self.decoder = MultiTriplane(1, input_dim=3, output_dim=1).to(self.device)
        self.decoder.eval()
        self.range = 1.
        self.middle = 0.

        # latent space parameters
        self.latent_code = None
        self.w0 = None
        self.w = None
        self.r1 = 12  # motion loss
        self.offset1 = make_offsets(self.r1, self.device)
        self.voxel_size = 2. / self.args.shape_resolution
        self.train_flag = True
        self.targets = None
        self.sources = None
        self.mesh = None
        self.mesh0 = None
        self.tri_feat0 = None
        self.tri_feat = None
        self.noise = []
        self.variance = []
        self.variance_noise = []
        self.feature_guidance0 = []
        self.feature_guidance = []

    def update_model_params(self, main_path):

        # load diffusion model
        for files in os.listdir(main_path):
            if files.startswith('ddpm'):
                ddpm_path = os.path.join(main_path, files)
                for sub_file in os.listdir(ddpm_path):
                    if sub_file.startswith('ema'):
                        self.args.model_path = os.path.join(ddpm_path, sub_file)
                        break
            elif files.endswith('.pt'):
                self.args.decoder_ckpt = os.path.join(main_path, files)
        stat_path = os.path.join(main_path, 'statistics')
        self.args.stats_dir = os.path.join(stat_path, os.listdir(stat_path)[0])
        self.args.save_dir = os.path.join('samples', main_path[9:]+'_samples')
        os.makedirs(self.args.save_dir, exist_ok=True)
        self.model.load_state_dict(
            dist_util.load_state_dict(self.args.model_path, map_location=self.device), strict=True)
        if self.args.use_fp16:
            self.model.convert_to_fp16()
        self.model.eval()

        # load decoder model
        if self.args.explicit_normalization:
            min_values = np.load(f'{self.args.stats_dir}/lower_bound.npy').astype(np.float32).reshape(1, -1, 1, 1)
            max_values = np.load(f'{self.args.stats_dir}/upper_bound.npy').astype(np.float32).reshape(1, -1, 1, 1)
            _range = th.Tensor((max_values - min_values)).to(self.device)
            middle = th.Tensor(((min_values + max_values) / 2)).to(self.device)
            self.range = (_range / 2).clone().detach()
            self.middle = middle.clone().detach()
        else:
            self.range = 1.
            self.middle = 0.
        self.decoder.net.load_state_dict(th.load(self.args.decoder_ckpt))
        self.decoder.eval()
        for param in self.decoder.net.parameters():
            param.requires_grad = False

    # get w and feature from self.latent_code,  latent_code --> w --> feature --> tri_feat --> mesh
    def update_latent_params(self, img=None, **kwargs):

        # save guidance features
        if img is not None:
            if th.is_tensor(img):
                img = img.type(next(iter(self.model.parameters())).dtype).to(self.device)
            elif type(img) is np.ndarray:
                img = th.tensor(img, dtype=next(iter(self.model.parameters())).dtype, device=self.device)
            else:
                raise NotImplementedError("Unknown data type!")
        else:
            img = th.randn((1, 96, self.args.image_size, self.args.image_size), dtype=next(
                iter(self.model.parameters())).dtype, device=self.device)
        self.latent_code = img.clone().detach()
        # np.save('noise.npy', img.cpu().numpy())
        with (th.no_grad()):
            for i in range(self.args.num_steps-1, -1, -1):
                t = th.tensor([i], device=img.device)
                outs = self.diffusion.p_sample_guidance(self.model, img, t, feat_layer=self.args.feat_layer,
                                                        clip_denoised=self.args.clip_denoised, **kwargs)
                img = outs["sample"]
                if i == self.args.w_time:
                    self.w = img.clone().detach()
                    self.w0 = self.w.clone().detach()
                if i < self.args.w_time:
                    self.feature_guidance0.append(resize_feat_align(outs["inter_feat"]).clone().detach().cpu())
            assert len(self.feature_guidance0) == self.args.w_time
            self.feature_guidance = copy.deepcopy(self.feature_guidance0)
            self.mesh0 = self.get_mesh(tri_feat=img)
            self.mesh = copy.deepcopy(self.mesh0)
            self.tri_feat0 = img.clone().detach().cpu()
            self.tri_feat = copy.deepcopy(self.tri_feat0)
            return img

    def get_mesh(self, tri_feat=None, img=None, t=None):
        with th.no_grad():
            if tri_feat is None:
                img = img if img is not None else (
                    th.randn((1, 96, self.args.image_size, self.args.image_size)).to(self.device))
                tri_feat = synthesize_latent(
                    self.model, self.diffusion, self.args, t1=t, img=img, clip_denoised=self.args.clip_denoised)["img"]
            tri_feat = (tri_feat * self.range + self.middle).reshape(3, 32, self.args.image_size, self.args.image_size)
            # tri_feat = tri_feat.reshape(3, 32, self.args.image_size, self.args.image_size)
            for i in range(3):
                self.decoder.embeddings[i] = tri_feat[[i]]
            mesh = create_obj_o3d(self.decoder, 0, res=self.args.shape_resolution)
            # o3d.io.write_triangle_mesh('1.obj', mesh)
            return mesh.filter_smooth_simple(number_of_iterations=10)

    def training(self, sources=None, targets=None, scale=600, cof=0.2):
        if self.args.num_samples > 1:
            raise NotImplementedError('We can handle only one shape at each time!')
        self.sources = th.tensor(sources, device=self.device, dtype=th.float32)  # B*3
        self.targets = th.tensor(targets, device=self.device, dtype=th.float32)  # B*3
        assert self.sources.shape[0] == self.targets.shape[0]

        num_piece = int(th.ceil(th.norm((self.sources - self.targets), dim=1).max() / 15))
        print("total piece number:", num_piece, th.norm((self.sources - self.targets), dim=1).max())
        sources_ = self.sources.clone().detach()
        mesh = None
        for piece in range(1, num_piece+1):
            targets_ = piece / num_piece * self.targets + (1 - piece / num_piece) * self.sources
            img = self.w.clone().detach()
            stop_time = 0
            self.train_flag = True

            # mask regularization
            img_width = self.feature_guidance[0].shape[-1]
            img_channel = self.feature_guidance[0].shape[1]
            patch_pnt = sources_.unsqueeze(1) + self.voxel_size * self.offset1.unsqueeze(0)  # B*N1*3
            shift_pnt = targets_.unsqueeze(1) + self.voxel_size * self.offset1.unsqueeze(0)  # B*N1*3
            patch_grid = th.cat((patch_pnt[..., :2].unsqueeze(0), patch_pnt[..., 1:].unsqueeze(0),
                                 patch_pnt[..., :3:2].unsqueeze(0)), dim=0)  # 3*B*N1*2
            shift_grid = th.cat((shift_pnt[..., :2].unsqueeze(0), shift_pnt[..., 1:].unsqueeze(0),
                                 shift_pnt[..., :3:2].unsqueeze(0)), dim=0)  # 3*B*N1*2
            patch_pnt_int = th.round((patch_pnt + 1) * (img_width - 1) / 2).type(dtype=th.int16).reshape(-1, 3)  # (B*N1)*3
            shift_pnt_int = th.round((shift_pnt + 1) * (img_width - 1) / 2).type(dtype=th.int16).reshape(-1, 3)  # (B*N1)*3
            content_idx = th.cat((patch_pnt_int, shift_pnt_int), dim=0)  # (2*B*N1)*3
            p = th.arange(0, img_width).type(th.int16)
            px, py = th.meshgrid(p, p, indexing='ij')
            total_idx = th.stack([px.reshape(-1), py.reshape(-1)], dim=-1).tolist()
            total_idx = set([tuple(i) for i in total_idx])
            xy_idx = content_idx[:, [1, 0]].cpu().tolist()
            xy_idx_mask = th.tensor(list(total_idx - set([tuple(i) for i in xy_idx])), device=self.device)
            yz_idx = content_idx[:, [2, 1]].cpu().tolist()
            yz_idx_mask = th.tensor(list(total_idx - set([tuple(i) for i in yz_idx])), device=self.device)
            xz_idx = content_idx[:, [2, 0]].cpu().tolist()
            xz_idx_mask = th.tensor(list(total_idx - set([tuple(i) for i in xz_idx])), device=self.device)
            for i in range(self.args.w_time-1, -1, -1):
                if not self.train_flag:
                    stop_time = i + 1
                    break
                img.requires_grad_(True)
                # # case2: Don't fix variance_noise
                outs = self.diffusion.p_sample_guidance(
                    self.model, img, th.tensor([i], device=self.device), feat_layer=self.args.feat_layer)

                edit_feature = resize_feat_align(outs["inter_feat"])  # 3*c*s*s
                origin_feature = self.feature_guidance[self.args.w_time-1-i].to(self.device)

                # motion loss
                patch_feature = th.nn.functional.grid_sample(
                    origin_feature, patch_grid, mode='bilinear', padding_mode='zeros', align_corners=True)  # 3*c*B*N1
                shift_feature = th.nn.functional.grid_sample(edit_feature, shift_grid, mode='bilinear',
                                                             padding_mode='zeros', align_corners=True)  # 3*c*B*N1
                if self.args.loss_type == 'l1':
                    if cof <= 0:
                        mask_loss = 0.
                    else:
                        mask_loss = ((edit_feature[0, :, xy_idx_mask[:, 0], xy_idx_mask[:, 1]] -
                                      origin_feature[0, :, xy_idx_mask[:, 0], xy_idx_mask[:, 1]]).abs().sum() +
                                     (edit_feature[1, :, yz_idx_mask[:, 0], yz_idx_mask[:, 1]] -
                                      origin_feature[1, :, yz_idx_mask[:, 0], yz_idx_mask[:, 1]]).abs().sum() +
                                     (edit_feature[2, :, xz_idx_mask[:, 0], xz_idx_mask[:, 1]] -
                                      origin_feature[2, :, xz_idx_mask[:, 0], xz_idx_mask[:, 1]]).abs().sum())/(
                                img_channel*(xy_idx_mask.shape[0]+yz_idx_mask.shape[0]+xz_idx_mask.shape[0]))
                    loss = -ff.l1_loss(shift_feature, patch_feature.detach()) - cof * mask_loss  # l1 loss
                else:  # l2 loss
                    if cof <= 0:
                        mask_loss = 0.
                    else:
                        mask_loss = (((edit_feature[0, :, xy_idx_mask[:, 0], xy_idx_mask[:, 1]] -
                                      origin_feature[0, :, xy_idx_mask[:, 0], xy_idx_mask[:, 1]])**2).sum() +
                                     ((edit_feature[1, :, yz_idx_mask[:, 0], yz_idx_mask[:, 1]] -
                                      origin_feature[1, :, yz_idx_mask[:, 0], yz_idx_mask[:, 1]])**2).sum() +
                                     ((edit_feature[2, :, xz_idx_mask[:, 0], xz_idx_mask[:, 1]] -
                                      origin_feature[2, :, xz_idx_mask[:, 0], xz_idx_mask[:, 1]])**2).sum())/(
                                img_channel*(xy_idx_mask.shape[0]+yz_idx_mask.shape[0]+xz_idx_mask.shape[0]))
                    loss = -((shift_feature.reshape(-1)-patch_feature.detach().reshape(-1))**2).mean() - cof*mask_loss
                loss.backward()
                grads1 = img.grad.clone().detach()
                grads = scale*grads1
                with th.no_grad():

                    # case2: don't fix variance
                    img = (outs["sample"] + outs["variance"] * grads).clone().detach()
                    assert outs["variance"].shape == grads.shape

                if not i % 10:
                    print("Step%d done!" % i, grads1.max().item(), grads1.min().item(), grads1.norm().item(), loss.item())
                yield 1 - i / (self.args.w_time - 1.)
                self.feature_guidance[self.args.w_time - 1 - i] = edit_feature.clone().detach().cpu()
            sources_ = targets_
            mesh = self.get_mesh(tri_feat=img)
            o3d.io.write_triangle_mesh('%d.obj' % piece, mesh)

        self.mesh = mesh
        # self.tri_feat = img.clone().detach().cpu()
        # o3d.io.write_triangle_mesh('test11.obj', self.mesh)
        # np.save('tri_feat.npy', self.tri_feat.numpy())

    def train_triplane(self, tri_feat_path=None):

        with th.no_grad():
            img = th.tensor(np.load(tri_feat_path), device=self.device)
            self.clear_params()
            self.mesh = self.get_mesh(img)
            self.mesh0 = copy.deepcopy(self.mesh)
            self.tri_feat0 = img.clone().detach().cpu()
            self.tri_feat = copy.deepcopy(self.tri_feat0)
            self.latent_inversion(tri_feat=img)
            return

    def latent_inversion(self, tri_feat, name='noise.npy'):
        with th.no_grad():
            outs = self.diffusion.ddpm_inversion(
                self.model, tri_feat, self.args.w_time, clip_denoised=self.args.clip_denoised,
                feat_layer=self.args.feat_layer)
            print('inversion done!')
            noise = outs["latent"]
            # np.save(name, noise.cpu().numpy())
            x = noise.cpu().numpy()
            print(x.min(), x.max(), x.mean(), x.var())
        self.w = noise.clone().detach()
        self.w0 = self.w.clone().detach()
        self.feature_guidance0 = [
            resize_feat_align(inter_feat).clone().detach().cpu() for inter_feat in outs["inter_feat"]]
        self.feature_guidance = copy.deepcopy(self.feature_guidance0)

    def clear_params(self):
        self.mesh0 = None
        self.mesh = None
        self.latent_code = None
        self.w0 = None
        self.w = None
        self.feature_guidance0.clear()
        self.feature_guidance.clear()
        self.noise.clear()
        self.variance.clear()
        self.variance_noise.clear()

    def reset_params(self):
        if self.mesh is not None:
            self.mesh = copy.deepcopy(self.mesh0)
        if self.w0 is not None:
            self.w = (self.w0.clone().detach()).requires_grad_(True)
        self.tri_feat = copy.deepcopy(self.tri_feat0)
        self.feature_guidance.clear()
        self.feature_guidance = copy.deepcopy(self.feature_guidance0)


def main():

    drag = DragStuff()
    drag.update_model_params('./models/chairs')
    # drag.train_triplane(mesh_path='mesh_scale_smooth.obj', center_mesh=False)
    # drag.update_latent_params()
    o3d.io.write_triangle_mesh('mesh.obj', drag.get_mesh(tri_feat=th.tensor(np.load('tri_feat.npy'), device=drag.device)))
    # drag.latent_inversion_feat(tri_feat=th.tensor(np.load('tri_feat.npy'), device=drag.device))

    """Test for mesh reconstruction on real shape datasets"""
    # name_list = ["1b67a3a1101a9acb905477d2a8504646", "1b938d400e1a340b17b431cae0dd70ed", "1b05971a4373c7d2463600025db2266",
    #              "1b80175cc081f3e44e4975e87c20ce53", "1b81441b7e597235d61420a53a0cb96d", "1b92525f3945f486fe24b6f1cb4a9319"]   # chairs id

    # name_list = ["1a0bc9ab92c915167ae33d942430658c", "1a0c91c02ef35fbe68f60a737d94994a", "1a1dcd236a1e6133860800e6696b8284",
    #              "1a1de15e572e039df085b75b20c2db33", "1a3b35be7a0acb2d9f2366ce3e663402", "1a4ef4a2a639f172f13d1237e1429e9e",
    #              "1a7b9697be903334b99755e16c4a9d21", "1a7d2c9d3a084885afa2ee0adc62d22", "1a15dff06b073d7282a2d03a8c21b1d6",
    #              "1a48d03a977a6f0aeda0253452893d75", "1a56d596c77ad5936fa87a658faf1d26", "1a63a260180f11baafe717997470b28d",
    #              "1a64bf1e658652ddb11647ffa4306609", "1a83a525ae32afcf622ac2f50deaba1f", "1a87a329781b15763820d8f180caa23a",
    #              "1a759c0122847d985c08d463d5424c88", "1a4337da899da1936909632a947fc19b", "1a7125aefa9af6b6597505fd7d99b613"]   # cars id
    name_list = ["1a30678509a1fdfa7fb5267a071ae23a", "1a19271683597db4fe7e6f8a8e38f62d", "1ab80bc91a45b7d0a31091d2234b9f68",
                 "1acfbda4ce0ec524bedced414fad522f", "1ad321f067ffbe7e51a95aaa6caba1d3", "1ae184691a39e3d3e0e8bce75d28b114"]

    # name_list = ["1a04e3eab45ca15dd86060f189eb133", "1a6ad7a24bb89733f412783097373bdc", "1a9b552befd6306cc8f2d5fe7449af61",
    #              "1a32f10b20170883663e90eaf6b4ca52", "1a54a2319e87bd4071d03b466c72ce41", "1a74b169a76e651ebc0909d98a1ff2b4",
    #              "1a888c2c86248bbcf2b0736dd4d8afe0", "1a963a929d9b1332290d63dca780cfb6", "1a29042e20ab6f005e9e2656aff7dd5b",
    #              "1abe9524d3d38a54f49a51dc77a0dd59", "1ac29674746a0fc6b87697d3904b168b", "1adb40469ec3636c3d64e724106730cf",
    #              "1af4b32eafffb0f7ee60c37cbf99c1c", "1b0b1d2cb9f9d5c0575bd26acccafabd", "1b3c6b2fbcf834cf62b600da24e0965",
    #              "1b7ac690067010e26b7bd17e458d0dcb", "1b300fd9ad4050e6301fa0a0663ee996", "1b626fd06226b600adcbeb54f3d014e9"]    # planes
    # path = "/media/jing/I/datas/cars"
    # drag = DragStuff()
    # drag.update_model_params('./models/cars')
    # for name in name_list:
    #     mesh_path = os.path.join(path, name)
    #     drag.train_triplane(mesh_path=os.path.join(mesh_path, 'mesh_scale_smooth.obj'), center_mesh=False, path=mesh_path)
    #     print(mesh_path, "done!")


if __name__ == "__main__":
    main()

