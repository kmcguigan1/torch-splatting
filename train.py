import os
import torch
import numpy as np
import random  # Import the random module
import gaussian_splatting.utils as utils
from gaussian_splatting.trainer import Trainer
import gaussian_splatting.utils.loss_utils as loss_utils
from gaussian_splatting.utils.data_utils import read_all
from gaussian_splatting.utils.camera_utils import to_viewpoint_camera
from gaussian_splatting.utils.point_utils import get_point_clouds
from gaussian_splatting.gauss_model import GaussModel
from gaussian_splatting.gauss_render import GaussRenderer

import contextlib

from torch.profiler import profile, ProfilerActivity

USE_GPU_PYTORCH = True
USE_PROFILE = False

# Set the seed for all random number generators
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    # For CUDA (if using GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_torch_gpu_memory_usage(device=0):
    """
    Returns the allocated and reserved memory by PyTorch on the specified GPU.

    :param device: Index of the GPU (default is 0)
    :return: Dictionary containing memory details in MB
    """
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check your PyTorch installation and GPU setup.")
        return None

    # Get allocated and reserved memory
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # Convert bytes to MB
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)    # Convert bytes to MB
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    max_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    # Get total memory from device properties
    total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)  # MB

    # Calculate free memory (approximation)
    free_memory = total_memory - reserved

    memory_info = {
        'Allocated Memory (MB)': allocated,
        'Reserved Memory (MB)': reserved,
        'Max Allocated Memory (MB)': max_allocated,
        'Max Reserved Memory (MB)': max_reserved,
        'Total Memory (MB)': total_memory,
        'Approx Free Memory (MB)': free_memory
    }

    # print(f"GPU {device} Memory Usage:")
    # for key, value in memory_info.items():
    #     print(f"  {key}: {value:.2f} MB")

    return memory_info

class GSSTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = kwargs.get('data')
        self.gaussRender = GaussRenderer(**kwargs.get('render_kwargs', {}))
        self.lambda_dssim = 0.2
        self.lambda_depth = 0.0
    
    def on_train_step(self):
        ind = np.random.choice(len(self.data['camera']))
        camera = self.data['camera'][ind]
        rgb = self.data['rgb'][ind]
        depth = self.data['depth'][ind]

        mask = (self.data['alpha'][ind] > 0.5)
        if USE_GPU_PYTORCH:
            camera = to_viewpoint_camera(camera)

        if USE_PROFILE:
            prof = profile(activities=[ProfilerActivity.CUDA], with_stack=True)
        else:
            prof = contextlib.nullcontext()

        with prof:
            out = self.gaussRender(pc=self.model, camera=camera)

        if USE_PROFILE:
            print(prof.key_averages(group_by_stack_n=True).table(sort_by='self_cuda_time_total', row_limit=20))


        l1_loss = loss_utils.l1_loss(out['render'], rgb)
        depth_loss = loss_utils.l1_loss(out['depth'][..., 0][mask], depth[mask])
        ssim_loss = 1.0-loss_utils.ssim(out['render'], rgb)

        total_loss = (1-self.lambda_dssim) * l1_loss + self.lambda_dssim * ssim_loss + depth_loss * self.lambda_depth
        psnr = utils.img2psnr(out['render'], rgb)

        memory_info = get_torch_gpu_memory_usage()

        percent_memory_usage = memory_info['Max Reserved Memory (MB)']/memory_info['Total Memory (MB)']*100

        log_dict = {'total': total_loss,'l1':l1_loss, 'ssim': ssim_loss, 'depth': depth_loss, 'psnr': psnr, 
                    'pos':out['tile_stats']['pos'], 'negs':out['tile_stats']['negs'], 'Memory Usage (%)':percent_memory_usage}

        return total_loss, log_dict
    
    def run_all_cameras(self):
        import matplotlib.pyplot as plt
        for ind in range(len(self.data['camera'])):
            print("writting camera: ", ind)
            camera = self.data['camera'][ind]

            if USE_GPU_PYTORCH:
                camera = to_viewpoint_camera(camera)

            out = self.gaussRender(pc=self.model, camera=camera)
            rgb_pd = out['render'].detach().cpu().numpy()
            utils.imwrite(str(self.results_folder / f'image-{ind}.png'), rgb_pd)

    def post_run_step(self, **kwargs):
        import matplotlib.pyplot as plt
        with torch.no_grad():
            for ind in range(len(self.data['camera'])):
                # print("writting camera: ", ind)
                camera = self.data['camera'][ind]
                rgb = self.data['rgb'][ind]
                depth = self.data['depth'][ind]
                mask = (self.data['alpha'][ind] > 0.5)

                if USE_GPU_PYTORCH:
                    camera = to_viewpoint_camera(camera)

                out = self.gaussRender(pc=self.model, camera=camera)

                l1_loss = loss_utils.l1_loss(out['render'], rgb)
                depth_loss = loss_utils.l1_loss(out['depth'][..., 0][mask], depth[mask])
                ssim_loss = 1.0-loss_utils.ssim(out['render'], rgb)

                total_loss = (1-self.lambda_dssim) * l1_loss + self.lambda_dssim * ssim_loss + depth_loss * self.lambda_depth
                psnr = utils.img2psnr(out['render'], rgb)

                log_dict = {'total': total_loss,'l1':l1_loss, 'ssim': ssim_loss, 'depth': depth_loss, 'psnr': psnr,
                            'pos': out['tile_stats']['pos'], 'negs':out['tile_stats']['negs']}
                print(f"Camera: {ind} -> {', '.join([f'{key}: {val:.4f}' for key, val in log_dict.items()])}")

                rgb = self.data['rgb'][ind].detach().cpu().numpy()
                depth = self.data['depth'][ind].detach().cpu().numpy()
                rgb_pd = out['render'].detach().cpu().numpy()
                depth_pd = out['depth'].detach().cpu().numpy()[..., 0]
                depth = np.concatenate([depth, depth_pd], axis=1)
                depth = (1 - depth / depth.max())
                depth = plt.get_cmap('jet')(depth)[..., :3]
                image = np.concatenate([rgb, rgb_pd], axis=1)
                image = np.concatenate([image, depth], axis=0)
                utils.imwrite(str(self.results_folder / f'image-{ind}-{self.step}.png'), image)

    def on_evaluate_step(self, **kwargs):
        import matplotlib.pyplot as plt
        ind = np.random.choice(len(self.data['camera']))
        camera = self.data['camera'][ind]

        if USE_GPU_PYTORCH:
            camera = to_viewpoint_camera(camera)

        rgb = self.data['rgb'][ind].detach().cpu().numpy()
        out = self.gaussRender(pc=self.model, camera=camera)
        rgb_pd = out['render'].detach().cpu().numpy()
        depth_pd = out['depth'].detach().cpu().numpy()[..., 0]
        depth = self.data['depth'][ind].detach().cpu().numpy()
        depth = np.concatenate([depth, depth_pd], axis=1)
        depth = (1 - depth / depth.max())
        depth = plt.get_cmap('jet')(depth)[..., :3]
        image = np.concatenate([rgb, rgb_pd], axis=1)
        image = np.concatenate([image, depth], axis=0)
        utils.imwrite(str(self.results_folder / f'image-{self.step}.png'), image)

def manual_debug():
    device = 'cuda'
    folder = './B075X65R3X'
    data = read_all(folder, resize_factor=0.25)
    data = {k: v.to(device) for k, v in data.items()}
    data['depth_range'] = torch.Tensor([[1,3]]*len(data['rgb'])).to(device)

    gaussModel = GaussModel(debug=False)
    gaussModel.create_manually()

    render_kwargs = {
        'white_bkgd': True,
    }

    results_folder = 'result/test'
    os.makedirs(results_folder, exist_ok=True)
    trainer = GSSTrainer(model=gaussModel, 
        data=data,
        train_batch_size=1, 
        train_num_steps=1,
        i_image=1,
        train_lr=1e-3, 
        amp=False,
        fp16=True,
        results_folder=results_folder,
        render_kwargs=render_kwargs,
        ignore_negatives=False
    )
    trainer.run_all_cameras()

def main():
    device = 'cuda'
    folder = './B075X65R3X'
    data = read_all(folder, resize_factor=0.25)
    data = {k: v.to(device) for k, v in data.items()}
    data['depth_range'] = torch.Tensor([[1,3]]*len(data['rgb'])).to(device)

    # Call the set_seed function with a fixed seed value
    SEED = 42  # You can choose any integer value
    set_seed(SEED)

    # Set the current device
    torch.cuda.set_device(0)


    points = get_point_clouds(data['camera'], data['depth'], data['alpha'], data['rgb'])
    raw_points = points.random_sample(2**14)
    # raw_points.write_ply(open('points.ply', 'wb'))

    gaussModel = GaussModel(debug=False)
    gaussModel.create_from_pcd(pcd=raw_points)
    
    render_kwargs = {
        'white_bkgd': True,
    }

    results_folder = 'result/test'
    os.makedirs(results_folder, exist_ok=True)
    trainer = GSSTrainer(model=gaussModel, 
        data=data,
        train_batch_size=1, 
        train_num_steps=1000,
        i_image =100,
        train_lr=1e-3, 
        amp=False,
        fp16=True,
        results_folder=results_folder,
        render_kwargs=render_kwargs,
    )

    # trainer.on_evaluate_step()
    trainer.train()
    trainer.post_run_step()


if __name__ == "__main__":
    # manual_debug()
    main()