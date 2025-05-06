from fastMRI.fastmri.pl_modules.unet_module import UnetModule
import torch
import torch.nn.functional as F

class ROIUnetModule(UnetModule):
    
    def __init__(self, roi_size=128, alpha=2.0, *args, **kwargs):
        # init parent class 
        super().__init__(*args, **kwargs)

        self.roi_size = (roi_size, roi_size) # (H,W) size of the ROI region
        self.alpha = alpha
    
    def create_roi_mask(self, image_shape, roi_size=(128,128), device=None) -> torch.Tensor:
        """
        Generate mask for the ROI region. Will fill mask with 1s and rest with 0s.

        inputs:
            image_shape: shape of the image (C, H, W)
            roi_size: size of the ROI region (H, W)
            device: device to create the mask on (default is None, which uses the default device)
        outputs:
            mask: mask with 1s in the ROI region and 0s elsewhere
        """
        h, w = image_shape[-2:] # (H,W) into individuals
        rh, rw = roi_size # (H,W) into individuals
        # empty mask 
        # do we need to set device?????
        mask = torch.zeros((h, w), device=device) # (H,W)
        start_h = (h - rh) // 2 # H
        start_w = (w- rw) // 2 # W

        mask[start_h:start_h + rh, start_w:start_w + rw] = 1.0
        # so whats the shape of the mask?
        print(mask.shape)       
        return mask # in THEORY (H,W) shape. lets see
    
    def training_step(self, batch, batch_idx):
        '''
        Overriding training step to include ROI based loss 

        inputs:
            batch: batch of data from dataloader
            batch_idx: index of the batch (not used)
        outputs:
            loss: loss value for the batch
        '''
        # so forward pass in module 
        output = self(batch.image)

        # create ROI mask 
        roi_mask = self.create_roi_mask(
            image_shape=batch.image.shape,
            roi_size=self.roi_size,
            device=batch.image.device
        )

        # lets make roi_mask future proof and same shape as output
        roi_mask = roi_mask.unsqueeze(0) # (1, H, W)
        roi_mask = roi_mask.expand_as(output) # (B, H, W)

        # do a manual L1 loss to get access to all pixels 
        # Nope I lied we can change reduction to none and get input shape 
        base_loss = F.l1_loss(output, batch.target, reduction='none') 


        # so this gives us per pixel loss 
        # 0s everywhere except 1s for ROI region 
        # now do ROI weighting 
        alpha = self.alpha 
        
        '''
        2 ROI approaches here:
            (1 + (alpha - 1) * roi_mask)
                so this gives us a weighted mask and we can modify the alpha value to change the weighting!!! 
                    this gives us more control!
                so inside the ROI region we have (1 + (alpha - 1) * 1) = alpha
                and outside the ROI region we have (1 + (alpha - 1) * 0) = 1
        
            base_loss * roi_mask
                this only focuses the ROI region and sets loss elsewhere to 0 
                could be useful to visualize but not sure if its useful for training

        '''
        print("implementing roi loss")
        roi_loss = (1 + (alpha - 1) * roi_mask) * base_loss

        # default reduction is mean
        loss = roi_loss.mean()

        self.log("loss", loss.detach())

        return loss