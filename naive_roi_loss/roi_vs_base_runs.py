import os
import pathlib
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import UnetDataTransform
# from ..pl_modules import UnetModule, FastMriDataModule
from fastmri.pl_modules import FastMriDataModule

from modules.roi_unet_module import ROIUnetModule


def load_config(config_path="configs/roi_vs_base.yaml"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def initialize_data_module(config, data_path):
    # Masking
    mask = create_mask_for_mask_type(
        mask_type_str=config["mask_type"],
        center_fractions=config["center_fractions"],
        accelerations=config["accelerations"],
    )

    # Transforms
    # setting seed cause I wanna train regular Unet on same samples as ROI Unet
    train_transform = UnetDataTransform(
        which_challenge=config["challenge"], mask_func=mask, use_seed=True
    )
    val_transform = UnetDataTransform(
        which_challenge=config["challenge"], mask_func=mask, use_seed=True
    )
    test_transform = UnetDataTransform(
        which_challenge=config["challenge"]
    )

    # Data Module
    data_module = FastMriDataModule(
        data_path=data_path,
        challenge=config["challenge"],
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=config.get("test_split", "val"),
        test_path=pathlib.Path(config.get("test_path", None)),
        sample_rate=config.get("sample_rate", None),
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        distributed_sampler=True,
    )

    return data_module

def main():
    # Load config
    config = load_config()

    pl.seed_everything(config["seed"])


    # Set device and paths
    data_path = pathlib.Path(config["data_path"])
    # print("Data path:", data_path)

    # Checkpoint loading logic
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

    # Define checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch{epoch}-step{step}",
        save_top_k=1,
        monitor="val_loss",  # changed from train_loss to val_loss
        mode="min",
        save_last=True,
    )



    #################### loop here 
    import numpy as np
    # alpha_range = np.arange(1.5, 2.0, 0.25)
    # roi_range = np.arange(60, 140, 20)

    # alpha_range = np.arange(0.5, 2.01, 0.25)  
    # roi_range = np.arange(60, 81, 20)         
    # alpha_range = np.arange(0.25, 2.01, 0.25)  
    # roi_range = np.arange(60, 101, 20)        
    alpha_range = [1.25, 1.0, 0.75]
    roi_range = [100, 100, 100]



    # for alpha in alpha_range:
        # for roi in roi_range:


    for alpha, roi in zip(alpha_range, roi_range):
        data_module = initialize_data_module(config, data_path)
        alpha = float(alpha)
        roi = int(roi)
        print(f"Training with alpha={alpha} \troi_size={roi}")
        # Model
        model = ROIUnetModule(
            alpha=alpha,
            roi_size=roi,
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


        # try to add logging
        log_dir = pathlib.Path(f'{config["default_root_dir"]}/logs/')

        log_dir.mkdir(parents=True, exist_ok=True)

        print(f"Logging metrics to: {log_dir}")

        # tb_logger = pl.loggers.TensorBoardLogger(save_dir=log_dir, name="")

        # Logging dir with params
        run_name = f"alpha{alpha}_roi{roi}"

        # don't wanna waste time on params if dir exists 
        if os.path.exists(f"{log_dir}/{run_name}"):
            print(f"Skipping {run_name} as it already exists")
            continue
            
        # # also we only care about 1 1.0 run 
        # if os.path.exists(f"{log_dir}/alpha1.0_roi*"):
        #     print(f"Skipping {run_name} as we already have a 1.0 run")
        #     continue

        tb_logger = pl.loggers.TensorBoardLogger(
            save_dir=log_dir,
            name=run_name
        )


        tb_logger.log_hyperparams({
            "alpha": alpha,
            "roi_size": roi,
            "in_chans": config["in_chans"],
            "out_chans": config["out_chans"],
            "chans": config["chans"],
            "num_pool_layers": config["num_pool_layers"],
            "drop_prob": config["drop_prob"],
            "lr": config["lr"],
            "lr_step_size": config["lr_step_size"],
            "lr_gamma": config["lr_gamma"],
            "weight_decay": config["weight_decay"],
        })

        # Trainer
        trainer = pl.Trainer(
            gpus=config["gpus"] if config['use_gpu'] else 0,
            distributed_backend=config["distributed_backend"],
            max_epochs=config["max_epochs"],
            default_root_dir=config["default_root_dir"],
            # logger=True,
            # logger=csv_logger,
            logger=tb_logger,
            callbacks=[checkpoint_callback] if config.get("resume") else None,
            resume_from_checkpoint=checkpoint_path if config.get("resume") else None,
            replace_sampler_ddp=False,
            # add this or else get that annoying warn
            plugins=pl.plugins.DDPPlugin(find_unused_parameters=False),
        )

        # Train!
        trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()
