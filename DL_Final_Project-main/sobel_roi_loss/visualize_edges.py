import os
import pathlib
import yaml
import torch
import matplotlib.pyplot as plt
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import UnetDataTransform
from fastmri.pl_modules import FastMriDataModule
from roi_unet_module import ROIUnetModule


def load_config(config_path="roi_unet.yaml"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def filter_model_config(config):
    """Filter config to include only parameters expected by ROIUnetModule."""
    model_params = [
        "alpha",
        "sobel_threshold",
        "dilation_size",
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


def visualize_edges(checkpoint_path, config, output_dir="visualizations/edges", num_samples=5):
    # Create output directory
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device("cuda" if config["use_gpu"] and torch.cuda.is_available() else "cpu")

    # Filter config for model parameters
    model_config = filter_model_config(config)

    # Load model from checkpoint
    model = ROIUnetModule.load_from_checkpoint(checkpoint_path, **model_config)
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

    # Visualize edges for num_samples
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break

            # Move batch to device
            image = batch.image.to(device)  # Shape: (B, H, W) or (B, 1, H, W)
            print(f"Sample {i+1} - Image shape: {image.shape}")

            # Create ROI mask
            edge_map, roi_mask = model.create_roi_mask(
                image_shape=batch.image.shape,
                image=image,
                device=device,
                return_edge_map=True
            )
            print(f"Sample {i+1} - Edge map shape: {edge_map.shape}, ROI mask shape: {roi_mask.shape}")

            # Convert tensors to numpy for plotting
            image_np = image.squeeze().cpu().numpy()  # (H, W)
            edge_map_np = edge_map.squeeze().cpu().numpy()  # (H, W)
            roi_mask_np = roi_mask.squeeze().cpu().numpy()  # (H, W)
            print(f"Sample {i+1} - Image_np shape: {image_np.shape}, Edge_map_np shape: {edge_map_np.shape}, ROI_mask_np shape: {roi_mask_np.shape}")

            # Plot
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(image_np, cmap="gray")
            axes[0].set_title("Input Image")
            axes[0].axis("off")
            axes[1].imshow(edge_map_np, cmap="gray")
            axes[1].set_title("Sobel Edge Map")
            axes[1].axis("off")
            axes[2].imshow(roi_mask_np, cmap="gray")
            axes[2].set_title("ROI Mask")
            axes[2].axis("off")
            plt.suptitle(f"Sample {i+1}")
            plt.tight_layout()

            # Save plot
            plt.savefig(output_dir / f"edge_sample_{i+1}.png")
            plt.close()

    print(f"Edge visualizations saved to {output_dir}")


def main():
    config = load_config()

    checkpoint_dir = pathlib.Path(f'{config["default_root_dir"]}/logs/version_10/checkpoints')
    ckpt_files = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
    if not ckpt_files:
        raise FileNotFoundError("No checkpoint files found in checkpoint_dir")
    checkpoint_path = str(ckpt_files[-1])
    print(f"Using checkpoint: {checkpoint_path}")

    visualize_edges(checkpoint_path, config)


if __name__ == "__main__":
    main()