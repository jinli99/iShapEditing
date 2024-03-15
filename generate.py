"""
Generate 3D samples of a given class using a pretrained model.
Will output triplanes as .npy files and meshes as .ply files.
"""
import time
from neural_field_diffusion.guided_diffusion import image_sample
from triplane_decoder import visualize
import argparse
from argparse import Namespace
import numpy as np
import os


def main():
    parser = argparse.ArgumentParser(description='Generate a set of triplanes and their corresponding meshes')
    parser.add_argument('--resolution', type=str, default=128, required=False,
                        help='Triplane resolution.')
    name = 0
    if name == 0:
        parser.add_argument('--ddpm_ckpt', type=str,
                            default='models/chairs/ddpm_chairs_ckpts/ema_0.9999_200000.pt',
                            help='DDPM checkpoint.', required=False)
        parser.add_argument('--decoder_ckpt', type=str, default='models/chairs/chair_decoder.pt',
                            help='Decoder checkpoint.', required=False)
        parser.add_argument('--stats_dir', type=str, default='models/chairs/statistics/chairs_triplanes_stats',
                            help='Normalization statistics to use.', required=False)
        parser.add_argument('--save_dir', type=str, default='samples/chairs_samples',
                            help='Where to save generated samples.', required=False)
    elif name == 1:
        parser.add_argument('--ddpm_ckpt', type=str,
                            default='models/cars/ddpm_cars_ckpts/ema_0.9999_405000.pt',
                            help='DDPM checkpoint.', required=False)
        parser.add_argument('--decoder_ckpt', type=str, default='models/cars/car_decoder.pt',
                            help='Decoder checkpoint.', required=False)
        parser.add_argument('--stats_dir', type=str, default='models/cars/statistics/cars_triplanes_stats',
                            help='Normalization statistics to use.', required=False)
        parser.add_argument('--save_dir', type=str, default='samples/cars_samples',
                            help='Where to save generated samples.', required=False)
    else:
        parser.add_argument('--ddpm_ckpt', type=str,
                            default='models/planes/ddpm_planes_ckpts/ema_0.9999_220000.pt',
                            help='DDPM checkpoint.', required=False)
        parser.add_argument('--decoder_ckpt', type=str, default='models/planes/plane_decoder.pt',
                            help='Decoder checkpoint.', required=False)
        parser.add_argument('--stats_dir', type=str, default='models/planes/statistics/planes_triplanes_stats',
                            help='Normalization statistics to use.', required=False)
        parser.add_argument('--save_dir', type=str, default='samples/planes_samples',
                            help='Where to save generated samples.', required=False)

    parser.add_argument('--num_samples', type=int, default=1,
                        help='How many samples to generate.', required=False)
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for the DDPM. Use a smaller batch size if GPU memory is low.', required=False)
    parser.add_argument('--num_steps', type=int, default=256,
                        help='Number of steps to take in deniosing process.', required=False)
    parser.add_argument('--shape_resolution', type=int, default=256,
                        help='Resolution at which to decode shapes.', required=False)

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Generate triplane samples using DDPM with default arguments
    ddpm_args = Namespace(
        clip_denoised=True, num_samples=args.num_samples, batch_size=args.batch_size, use_ddim=False, model_path=args.ddpm_ckpt, stats_dir=args.stats_dir,
        explicit_normalization=True, save_dir=args.save_dir, save_intermediate=False, save_timestep_interval=20, image_size=args.resolution, num_channels=256,
        num_res_blocks=2, num_heads=4, num_heads_upsample=-1, num_head_channels=64, attention_resolutions='32,16,8', channel_mult='', dropout=0.1, class_cond=False,
        use_checkpoint=False, use_scale_shift_norm=True, resblock_updown=True, use_fp16=True, use_new_attention_order=False, in_out_channels=96, learn_sigma=True,
        diffusion_steps=1000, noise_schedule='linear', timestep_respacing=str(args.num_steps), use_kl=False, predict_xstart=False, rescale_timesteps=False,
        rescale_learned_sigmas=False
    )
    t1 = time.time()
    samples = image_sample.noise2shape(args=ddpm_args)  # Run ddpm
    t2 = time.time()
    print('ddpm time:', t2 - t1, 4)

    # Convert samples to a directory of .npy triplanes
    os.makedirs(f'{args.save_dir}/triplanes', exist_ok=True)

    samples = np.transpose(samples, [0, 3, 1, 2])
    for idx, triplane in enumerate(samples):
        save_path = f'{args.save_dir}/triplanes/{idx}.npy'
        print(f'saving to {save_path}...')
        np.save(save_path, triplane)

    # Decode triplane samples
    os.makedirs(f'{args.save_dir}/objects', exist_ok=True)

    for idx, triplane in enumerate(samples):
        print(f'Decoding triplane {idx}...')
        decoder_args = Namespace(
            input=f'{args.save_dir}/triplanes/{idx}.npy', output=f'{args.save_dir}/objects/{idx}.obj',
            model_path=args.decoder_ckpt, res=args.shape_resolution
        )
        visualize.main(args=decoder_args)  # Run decoder

    print('Done!')
    print('decode time:', time.time() - t2)


if __name__ == "__main__":
    main()
