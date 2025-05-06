import os
import pathlib
import yaml
import torch
import matplotlib.pyplot as plt
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import UnetDataTransform
from fastmri.pl_modules import FastMriDataModule
from square_roi_unet_module import SquareROIUnetModule
import numpy as np


def load_config(config_path="roi_unet.yaml"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def filter_model_config(config):
    """Filter config to include only parameters expected by SquareROIUnetModule."""
    model_params = [
        "alpha",
        "roi_size",
        "in_chans",
        "out_chans",
        "chans",
        "num_pool_layers",
        "drop_prob",
        "lr",
        "lr_step_size",
        "lr_gamma",
        "weight_decay",
    ]
    return {k: v for k, v in config.items() if k in model_params}


def compute_metrics(pred, target, roi_mask):
    """Compute MSE and SSIM within the ROI."""
    mse = ((pred - target) ** 2 * roi_mask).mean()
    # Simplified SSIM approximation (for display purposes)
    pred_mean = (pred * roi_mask).mean()
    target_mean = (target * roi_mask).mean()
    pred_var = ((pred - pred_mean) ** 2 * roi_mask).mean()
    target_var = ((target - target_mean) ** 2 * roi_mask).mean()
    cov = ((pred - pred_mean) * (target - target_mean) * roi_mask).mean()
    ssim = (2 * pred_mean * target_mean + 1e-8) * (2 * cov + 1e-8) / (
        pred_mean**2 + target_mean**2 + 1e-8
    ) / (pred_var + target_var + 1e-8)
    return mse.item(), ssim.item()


def visualize_predictions(checkpoint_path, config, output_dir="visualizations/square_roi_predictions", num_samples=5):
    # Create output directory
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device("cuda" if config["use_gpu"] and torch.cuda.is_available() else "cpu")

    # Filter config for model parameters
    model_config = filter_model_config(config)

    # Load model from checkpoint
    model = SquareROIUnetModule.load_from_checkpoint(checkpoint_path, **model_config)
    model.eval()
    model.to(device)

    # Masking
    mask = create_mask_for_mask_type(
        mask_type_str=config["mask_type"],
        center_fractions=config["center_fractions"],
        accelerations=config["accelerations"],
    )

    # Test transform
    test_transform = UnetDataTransform(which_challenge=config["challenge"])

    # Data module
    data_module = FastMriDataModule(
        data_path=pathlib.Path(config["data_path"]),
        challenge=config["challenge"],
        train_transform=None,
        val_transform=None,
        test_transform=test_transform,
        test_split=config.get("test_split", "val"),
        test_path=pathlib.Path(config.get("test_path", None)),
        sample_rate=config.get("sample_rate", None),
        batch_size=1,  # Single batch for visualization
        num_workers=config["num_workers"],
        distributed_sampler=False,
    )
    data_module.setup("test")
    test_loader = data_module.test_dataloader()

    mse_list, ssim_list = [], []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break

            # Move batch to device
            image = batch.image.to(device)  # Shape: (B, H, W) or (B, 1, H, W)
            target = batch.target.to(device)  # Shape: (B, 1, H, W)
            print(f"Sample {i+1} - Image shape: {image.shape}, Target shape: {target.shape}")

            # Get prediction
            pred = model(image)  # Shape: (B, 1, H, W)
            print(f"Sample {i+1} - Prediction shape: {pred.shape}")

            # Ensure prediction has channel dimension
            if pred.dim() == 3:  # (B, H, W)
                pred = pred.unsqueeze(1)  # (B, 1, H, W)
            print(f"Sample {i+1} - Prediction shape after adjustment: {pred.shape}")

            # Create ROI mask
            roi_mask = model.create_roi_mask(
                image_shape=batch.image.shape,
                roi_size=model.roi_size,
                device=device
            )
            # Adjust ROI mask shape to (B, C, H, W)
            roi_mask = roi_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, H, W)
            roi_mask = roi_mask.expand(pred.shape[0], 1, pred.shape[2], pred.shape[3])  # (B, 1, H, W)
            print(f"Sample {i+1} - ROI mask shape: {roi_mask.shape}")
            assert roi_mask.shape[1] == 1, f"Expected 1 channel in ROI mask, got {roi_mask.shape[1]}"

            # Convert tensors to numpy
            image_np = image.squeeze().cpu().numpy()  # (H, W)
            pred_np = pred.squeeze().cpu().numpy()  # (H, W)
            target_np = target.squeeze().cpu().numpy()  # (H, W)
            roi_mask_np = roi_mask[:, 0].squeeze().cpu().numpy()  # Select channel 0, (H, W)
            print(f"Sample {i+1} - Image_np shape: {image_np.shape}, Pred_np shape: {pred_np.shape}, Target_np shape: {target_np.shape}, ROI_mask_np shape: {roi_mask_np.shape}")
            assert len(roi_mask_np.shape) == 2, f"Expected 2D ROI_mask_np, got shape {roi_mask_np.shape}"

            # Compute metrics in ROI
            mse, ssim = compute_metrics(pred, target, roi_mask)
            mse_list.append(mse)
            ssim_list.append(ssim)

            # Plot
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            axes[0].imshow(image_np, cmap="gray")
            axes[0].set_title("Input (Undersampled)")
            axes[0].axis("off")
            axes[1].imshow(pred_np, cmap="gray")
            axes[1].set_title(f"Prediction\nMSE: {mse:.4f}, SSIM: {ssim:.4f}")
            axes[1].axis("off")
            axes[2].imshow(target_np, cmap="gray")
            axes[2].set_title("Ground Truth")
            axes[2].axis("off")

            axes[3].imshow(pred_np, cmap="gray")
            axes[3].imshow(roi_mask_np, cmap="jet", alpha=0.5)
            axes[3].set_title("Prediction with ROI Overlay")
            axes[3].axis("off")
            plt.suptitle(f"Sample {i+1}")
            plt.tight_layout()

            plt.savefig(output_dir / f"prediction_sample_{i+1}.png")
            plt.close()

    print(f"Prediction visualizations saved to {output_dir}")
    print(f"Average MSE: {np.mean(mse_list):.4f}, Average SSIM: {np.mean(ssim_list):.4f}")


def main():
    config = load_config()

    checkpoint_path = "C:/Users/navee/Documents/My Stuff/Georgia Tech/Classes/CS7643-Deep_Learning/FastMRI/fastMRI/roi_unet_checkpoints/logs/version_8/checkpoints/epoch=4-step=17370.ckpt"
    print(f"Using checkpoint: {checkpoint_path}")

    visualize_predictions(checkpoint_path, config)


if __name__ == "__main__":
    main()