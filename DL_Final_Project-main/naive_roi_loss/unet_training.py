import os
import pathlib
import yaml
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import UnetDataTransform
from fastmri.data import SliceDataset
from fastmri.pl_modules import UnetModule

from pytorch_lightning.callbacks import ModelCheckpoint


def load_config(config_path="configs/unet.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    # Load config
    config = load_config()

    # Set device and paths
    data_path = pathlib.Path(config["data_path"])
    print("Data path:", data_path)

    # New add!!!!!!!!!!!!!!!
    # Checkpoint loading logic
    # checkpoint_dir = pathlib.Path("unet_checkpoints/checkpoints")
    checkpoint_dir = pathlib.Path(f'{config["default_root_dir"]}/checkpoints')
    checkpoint_path = None
    if checkpoint_dir.exists():
        ckpt_files = sorted(
            checkpoint_dir.glob("*.ckpt"),
            key=os.path.getmtime
        )
        if ckpt_files:
            checkpoint_path = str(ckpt_files[-1])
            print(f"Resuming from checkpoint: {checkpoint_path}")

    # New add!!!!!!!!!!!!!!!
    # Define checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch{epoch}-step{step}",
        save_top_k=1,
        monitor="train_loss",
        mode="min",
        save_last=True,
    )



    # Masking
    mask = create_mask_for_mask_type(
        mask_type_str=config["mask_type"],
        center_fractions=config["center_fractions"],
        accelerations=config["accelerations"],
    )

    transform = UnetDataTransform(
        which_challenge=config["challenge"],
        mask_func=mask,
        use_seed=True,
    )

    # Dataset and Dataloader
    dataset = SliceDataset(
        root=data_path,
        transform=transform,
        challenge=config["challenge"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
    )

    # Model
    model = UnetModule(
        in_chans=config["in_chans"],
        out_chans=config["out_chans"],
        chans=config["chans"],
        num_pool_layers=config["num_pool_layers"],
        drop_prob=config["drop_prob"],
        lr=config["lr"],
        lr_step_size=config["lr_step_size"],
        lr_gamma=config["lr_gamma"],
        weight_decay=config["weight_decay"],
    )

    # Trainer
    trainer = pl.Trainer(
        gpus=config["gpus"],
        distributed_backend=config["distributed_backend"],
        max_epochs=config["max_epochs"],
        default_root_dir=config["default_root_dir"],
        logger=False,
        # new items!!!
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=checkpoint_path
        # new items!!!
    )

    # Train!
    trainer.fit(model, train_dataloaders=dataloader)

    # nah we can do this manually later
    # # Visualize one example
    # sample = next(iter(dataloader))
    # with torch.no_grad():
    #     output = model(sample["image"].unsqueeze(1))

    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # axes[0].imshow(sample["target"][0].numpy(), cmap="gray")
    # axes[0].set_title("Target")
    # axes[1].imshow(output.detach().squeeze().numpy(), cmap="gray")
    # axes[1].set_title("U-Net Output")
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()
