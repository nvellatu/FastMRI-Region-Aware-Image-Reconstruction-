from pl_modules.unet_module import UnetModule
import torch
import torch.nn.functional as F

class ROIUnetModule(UnetModule):
    
    def __init__(self, sobel_threshold=0.1, dilation_size=5, alpha=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sobel_threshold = sobel_threshold
        self.dilation_size = dilation_size
        self.alpha = alpha
        # Sobel kernels
        self.sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
        self.sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)
        # Dilation kernel
        self.dilation_kernel = torch.ones(1, 1, dilation_size, dilation_size)
    
    def create_roi_mask(self, image_shape, image, device=None, return_edge_map=False) -> torch.Tensor:
        """
        Generate mask for the ROI region using Sobel edge detection. Fills mask with 1s in the ROI and 0s elsewhere.

        inputs:
            image_shape: shape of the image (B, H, W) or (B, C, H, W)
            image: input image tensor (B, H, W) or (B, C, H, W)
            device: device to create the mask on (default is None, uses image device)
            return_edge_map: if True, return both edge_map and roi_mask
        outputs:
            mask: mask with 1s in the ROI region and 0s elsewhere, shape (B, C, H, W) where C=1 for single-coil
            edge_map: Sobel edge map, shape (B, 1, H, W) (if return_edge_map=True)
        """
        device = image.device if device is None else device
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)
        self.dilation_kernel = self.dilation_kernel.to(device)
        
        # Handle input dimensions
        if image.dim() == 3:
            img_for_sobel = image.unsqueeze(1)  # (B, 1, H, W)
            num_channels = 1  # Single-coil data has 1 channel
        else:
            img_for_sobel = image  # Assume (B, C, H, W)
            num_channels = image.shape[1]
            if num_channels > 1:
                img_for_sobel = img_for_sobel[:, :1, :, :]  # (B, 1, H, W)
                num_channels = 1
        
        # print(f"create_roi_mask: image_shape={image_shape}, image.dim()={image.dim()}, num_channels={num_channels}, img_for_sobel.shape={img_for_sobel.shape}")

        # Apply Sobel filters
        grad_x = F.conv2d(img_for_sobel, self.sobel_x, padding=1)
        grad_y = F.conv2d(img_for_sobel, self.sobel_y, padding=1)
        
        # Compute gradient magnitude
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        # print(f"create_roi_mask: grad_magnitude.shape={grad_magnitude.shape}")
        
        # Normalize to [0, 1] for thresholding
        grad_max = torch.amax(grad_magnitude, dim=(2, 3), keepdim=True)
        grad_normalized = grad_magnitude / (grad_max + 1e-8)  # Avoid division by zero
        # print(f"create_roi_mask: grad_normalized.shape={grad_normalized.shape}")
        
        # Apply threshold to get binary edge map
        edge_map = (grad_normalized > self.sobel_threshold).to(torch.float32)  # (B, 1, H, W)
        # print(f"create_roi_mask: edge_map.shape={edge_map.shape}")
        
        # Dilate edges to create a broader ROI
        dilated_map = F.conv2d(
            edge_map,
            self.dilation_kernel,
            padding=self.dilation_size // 2
        )
        # print(f"create_roi_mask: dilated_map.shape={dilated_map.shape}")
        mask = (dilated_map > 0).to(torch.float32)  # (B, 1, H, W)
        # print(f"create_roi_mask: mask before expansion={mask.shape}")
        
        # Ensure mask has correct shape for single-coil (B, 1, H, W)
        if mask.shape[1] != num_channels:
            mask = mask[:, :1, :, :]  # Ensure single channel
        mask = mask.expand(-1, num_channels, -1, -1)  # (B, 1, H, W)
        # print(f"create_roi_mask: mask after expansion={mask.shape}")
        
        if return_edge_map:
            return edge_map, mask
        return mask
    
    def training_step(self, batch, batch_idx):
        """
        Overriding training step to include ROI-based loss.

        inputs:
            batch: batch of data from dataloader
            batch_idx: index of the batch (not used)
        outputs:
            loss: loss value for the batch
        """
        # Log input shape for debugging
        # self.log("input_shape", str(batch.image.shape), on_step=True)
        
        # Forward pass
        output = self(batch.image)

        # Create ROI mask with edge detection
        roi_mask = self.create_roi_mask(
            image_shape=batch.image.shape,
            image=batch.image,
            device=batch.image.device
        )

        # L1 loss
        base_loss = F.l1_loss(output, batch.target, reduction='none')

        # Apply ROI weighting
        roi_loss = (1 + (self.alpha - 1) * roi_mask) * base_loss

        # Compute mean loss
        loss = roi_loss.mean()

        self.log("loss", loss.detach())

        return loss