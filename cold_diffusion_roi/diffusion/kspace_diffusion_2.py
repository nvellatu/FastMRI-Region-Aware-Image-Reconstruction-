import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
import numpy as np
from matplotlib import pyplot as plt
import fastmri
from help_func import print_var_detail


# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


def mask_sequence_sample(mask_fold):
    """
    calculate mask sequence given folded mask

    :param mask_fold: input folded mask tensor, [B, 1, H/num_patches_H, W/num_patches_W]
    :return: mask sequence tensor stores all masked patches position idx, [num of masked patches, 2]
    """
    mask_seqs = []
    for i in range(mask_fold.shape[0]):
        mask_idx = torch.where(mask_fold[i][0] == 0)
        mask_seq = torch.cat((mask_idx[0][:, None], mask_idx[1][:, None]), dim=1)
        r = torch.randperm(mask_seq.shape[0])
        mask_seq = mask_seq[r[:]]
        mask_seqs.append(mask_seq)
    return mask_seqs


class KspaceDiffusion(nn.Module):
    """
    diffusion module for kspace diffusing process

    Args:
    ----------
    denoise_fn : model
        actual training model for reverse diffusion process given image and time step t
    image_size : int
        a list containing elements like (file_name, index). A .h5 file contains multi slices.
    device_of_kernel : string
        device for training, usually 'cuda'
    channels : int
        number of channels for input image. default 3
    timesteps: int,
        training steps of reverse diffusion.
    loss_type: string
        type of loss function. default 'l1'
    blur_routine: string
        blur routine from one time step to next, default 'Incremental'
    train_routine : string
        the train routine to set if the training process is throughout to the end or not, default 'Final'
    sampling_routine : string
        the sample routine choose between 'default' or 'x0_step_down'
    discrete: bool,
        if the blur process is discrete or continuous, default 'False'
    alpha : float
        Weighting factor for ROI in loss function. Default 1.0.
    sobel_threshold : float
        Threshold for Sobel edge detection. Default 0.1.
    dilation_size : int
        Size of the dilation kernel for expanding the ROI. Default 5.
    """

    def __init__(
            self,
            denoise_fn,
            *,
            image_size,
            device_of_kernel,
            channels=3,
            timesteps=1000,
            loss_type='l1',
            blur_routine='Incremental',
            train_routine='Final',
            sampling_routine='default',
            discrete=False,
            alpha=1.0,
            sobel_threshold=0.1,
            dilation_size=5
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.device_of_kernel = device_of_kernel

        self.num_timesteps = timesteps
        self.loss_type = loss_type
        self.blur_routine = blur_routine

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.train_routine = train_routine
        self.sampling_routine = sampling_routine
        self.discrete = discrete


        ##### Upweighting adds #####
        self.alpha = alpha
        self.sobel_threshold = sobel_threshold
        self.dilation_size = dilation_size
        # Sobel kernels
        self.sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
        self.sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)
        # Dilation kernel
        self.dilation_kernel = torch.ones(1, 1, dilation_size, dilation_size)

    def q_sample(self, kspace, mask_seqs, mask, mask_fold, t):
        """
        sample a kspace given step t and masks

        :param kspace: fully sampled kspace tensor, [B, Nc, H, W, 2]
        :param mask_seqs: mask sequence tensor stores all masked patches position idx, [num of masked patches, 2]
        :param mask: input unfolded mask tensor, [B, 1, H, W]
        :param mask_fold: input folded mask tensor, [B, 1, H/num_patches_H, W/num_patches_W]
        :param t: time steps tensor, [B,]
        :return: image_part_masked: masked image tensor, [B, Nc, H, W, 2]; kspace_part_masked: masked kspace tensor,
            [B, Nc, H, W, 2]
        """

        # mask at t steps
        patch_size_H = mask.shape[-2] // mask_fold.shape[-2]
        patch_size_W = mask.shape[-1] // mask_fold.shape[-1]
        mask_part = torch.ones(mask_fold.shape)

        for i in range(mask_fold.shape[0]):
            mask_seq = mask_seqs[i]
            step = mask_seq.shape[0] / self.num_timesteps
            mask_part[i, ..., mask_seq[0:int(t[i] * step), 0], mask_seq[0:int(
                t[i] * step), 1]] = 0.0

        mask_expand = torch.nn.functional.interpolate(mask_part, scale_factor=[patch_size_H, patch_size_W],
                                                      mode='nearest').to(self.device_of_kernel)
        kspace_part_masked = kspace * mask_expand.unsqueeze(-1).repeat(1, kspace.shape[1], 1, 1, kspace.shape[-1])
        image_part_masked = fastmri.ifft2c(kspace_part_masked)  # [B,Nc,H,W,2]

        return image_part_masked, kspace_part_masked  # [B,Nc,H,W,2],  [B,Nc,H,W,2]

    @torch.no_grad()
    def sample(self, kspace, mask, mask_fold, t):
        """
        reverse diffusion sampling given unmasked kspace, masks and time step t

        :param kspace: fully sampled kspace tensor, [B, Nc, H, W, 2]
        :param mask: input unfolded mask tensor, [B, 1, H, W]
        :param mask_fold: input folded mask tensor, [B, 1, H/num_patches_H, W/num_patches_W]
        :param t: time steps tensor, [B,]
        :return: q_sample xt: masked image tensor, [B, Nc, H, W, 2]; direct_recons: direction reconstructed image from
            denoise_fn, [B, Nc, H, W, 2]; img: reverse sampled img [B, Nc, H, W, 2]
        """
        self.denoise_fn.eval()
        batch_size, Nc, H, W, C = kspace.shape

        if t is None:
            t = self.num_timesteps
        mask_seqs = mask_sequence_sample(mask_fold)
        # sample masked image and kspace given t
        with torch.no_grad():
            img, kspace = self.q_sample(kspace, mask_seqs, mask, mask_fold,
                                        torch.full((batch_size,), t, dtype=torch.long).cuda())  # [B,Nc,H,W,2]
        xt = img
        direct_recons = None
        while t:
            step = torch.full((batch_size,), t, dtype=torch.long).cuda()

            x = torch.zeros(xt.shape).to(self.device_of_kernel)
            for i in range(Nc):
                x[:, i, :] = self.denoise_fn(img[:, i, :].permute(0, 3, 1, 2), step).permute(0, 2, 3, 1)  # [B,Nc,H,W,2]
            k = fastmri.fft2c(x)

            if self.train_routine == 'Final':
                if direct_recons is None:
                    direct_recons = x
                if self.sampling_routine == 'default':
                    if self.blur_routine == 'Individual_Incremental':
                        x, _ = self.q_sample(k, mask_seqs, mask, mask_fold, step - 2)
                    else:
                        with torch.no_grad():
                            x, _ = self.q_sample(k, mask_seqs, mask, mask_fold, step)
                elif self.sampling_routine == 'x0_step_down':
                    k_times = k
                    with torch.no_grad():
                        x_times, _ = self.q_sample(k_times, mask_seqs, mask, mask_fold, step)
                    k_times_sub_1 = k
                    with torch.no_grad():
                        x_times_sub_1, _ = self.q_sample(k_times_sub_1, mask_seqs, mask, mask_fold, step - 1)

                    x = img - x_times + x_times_sub_1
            img = x
            t = t - 1
        self.denoise_fn.train()
        return xt, direct_recons, img  # [B,Nc,H,W,2]
    



    #######################################################################################
    ################# Adding roi mask code for the diffusion model ####################
    ################# Same as the one in roi_unet_module.py ####################
    #######################################################################################
    def create_roi_mask(self, image_shape, image, device=None) -> torch.Tensor:
        """
        Generate mask for the ROI region. Will fill mask with 1s and rest with 0s.

        inputs:
            image_shape: shape of the image (C, H, W)
            roi_size: size of the ROI region (H, W)
            device: device to create the mask on (default is None, which uses the default device)
        outputs:
            mask: mask with 1s in the ROI region and 0s elsewhere
        """
        device = image.device if device is None else device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)
        self.dilation_kernel = self.dilation_kernel.to(device)
        # image is (B, Nc, H, W, 2); we need (B, C, H, W) for Sobel
        # Take the magnitude of the complex image (combine real and imaginary parts)
        img_for_sobel = torch.sqrt(image[..., 0]**2 + image[..., 1]**2)  # (B, Nc, H, W)
        num_channels = img_for_sobel.shape[1]  # Nc
        # Use the first channel if Nc > 1 
        if num_channels > 1:
            img_for_sobel = img_for_sobel[:, :1, :, :]  # (B, 1, H, W)
            num_channels = 1
        # Apply filters
        grad_x = F.conv2d(img_for_sobel, self.sobel_x, padding=1)
        grad_y = F.conv2d(img_for_sobel, self.sobel_y, padding=1)
        # Compute gradient magnitude
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        # Normalize to [0, 1] for thresholding
        grad_max = torch.amax(grad_magnitude, dim=(2, 3), keepdim=True)
        grad_normalized = grad_magnitude / (grad_max + 1e-8)  # Avoid division by zero

        edge_map = (grad_normalized > self.sobel_threshold).to(torch.float32)  # (B, 1, H, W)

        dilated_map = F.conv2d(
            edge_map,
            self.dilation_kernel,
            padding=self.dilation_size // 2
        )
        mask = (dilated_map > 0).to(torch.float32)  # (B, 1, H, W)

        mask = mask.expand(-1, num_channels, -1, -1)  # (B, Nc, H, W)
        mask = mask.unsqueeze(-1).expand(-1, -1, -1, -1, 2)  # (B, Nc, H, W, 2)

        print(f"create_roi_mask: mask.shape={mask.shape}")
        return mask


    def p_losses(self, kspace, mask_seqs, mask, mask_fold, t):
        """
        loss function for denoise module given train_routine and loss_type

        :param kspace: fully sampled kspace tensor, [B, Nc, H, W, 2]
        :param mask_seqs: mask sequence tensor stores all masked patches position idx, [num of masked patches, 2]
        :param mask: input unfolded mask tensor, [B, 1, H, W]
        :param mask_fold: input folded mask tensor, [B, 1, H/num_patches_H, W/num_patches_W]
        :param t: time steps tensor, [B,]
        :return: loss: l1 or l2 loss
        """
        B, Nc, H, W, C = kspace.shape
        x_start = fastmri.ifft2c(kspace)  # [B,Nc,H,W,2]
        if self.train_routine == 'Final':
            x_blur, _ = self.q_sample(kspace, mask_seqs, mask, mask_fold, t)  # [B,Nc,H,W,2]

            x_recon = torch.zeros(x_blur.shape).to(self.device_of_kernel)
            for i in range(Nc):
                x_recon[:, i, :] = self.denoise_fn(x_blur[:, i, :].permute(0, 3, 1, 2), t).permute(0, 2, 3,
                                                                                                   1)  # [B,Nc,H,W,2]
            # adding reudction none here
            if self.loss_type == 'l1':
                loss = (x_start - x_recon).abs() #.mean()
            elif self.loss_type == 'l2':
                loss = F.mse_loss(x_start, x_recon, reduction='none')
            else:
                raise NotImplementedError()
            
            # loss is [B, Nc, H, W, 2] or torch.Size([6, 1, 320, 320, 2])
            # need to up the roi_mask dims to match the loss dims
            roi_mask = self.create_roi_mask(
                image_shape=x_start.shape,
                image=x_start,
                device=x_start.device
            )
            # create_roi_mask now handles this
            # roi_mask = roi_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1) # [1, 1, H, W, 1]
            # # now need to make it [B, Nc, H, W, C] shape
            # # below works but not the best way to do it imo
            # # roi_mask = roi_mask.expand(B, 1, -1, -1, 2) # [B, 1, H, W, 2]
            # roi_mask = roi_mask.expand(B, Nc, -1, -1, C)

            # default alpha 
            alpha = self.alpha
            roi_loss = (1 + (alpha - 1) * roi_mask) * loss
            loss = roi_loss.mean()

        return loss

    def forward(self, kspace, mask, mask_fold, *args, **kwargs):
        B, Nc, H, W, C, device, img_size, = *kspace.shape, kspace.device, self.image_size
        assert H == img_size and W == img_size, f'height and width of image must be {img_size}'

        # given random timestep and calculate p_losses
        t = torch.randint(0, self.num_timesteps + 1, (B,), device=device).long()
        mask_seqs = mask_sequence_sample(mask_fold)
        return self.p_losses(kspace, mask_seqs, mask, mask_fold, t, *args, **kwargs)