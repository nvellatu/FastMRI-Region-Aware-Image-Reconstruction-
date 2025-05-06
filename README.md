# fastMRI: Region-Aware Image Reconstruction

This repository contains the code and experiments for the paper:

**fastMRI: Region-Aware Image Reconstruction**

**Authors:**
- Luis Tupac (luis.tupac0@gmail.com)
- Igor Kamenetskiy (ikamenetskiy3@gatech.edu)
- Emma Resmini (eresmini3@gatech.edu)
- Naveen Vellaturi (nvellaturi3@gatech.edu)

Submitted as the final project for the Deep Learning course at Georgia Institute of Technology.

## Abstract

Magnetic Resonance Imaging, a popular noninvasive diagnostic procedure, is not currently used in some applications due to the duration of the scan. In many of these applications the pathology is highly localized, yet the data is collected in the frequency domain (k-space) that must be fully sampled, its under-sampling leads to distorted reconstructed images. Increasing the speed of MRI has been approached by training deep learning models (UNet, DDPM, Cold Diffusion) to reconstruct under-sampled (and hence faster to collect) k-space data. We propose to further improve the quality of the reconstructed images by locating the Region-of-Interest (ROI) and weighting the learning rate in the ROI to focus training of the model on the pathology region. We proved the concept with UNet model, using both manually chosen ROI and one detected by the Sobel edge-detecting operator, where improvement in the image quality characteristics (SSIM, NMSE, and PSNR) were demonstrated. The same approach was further applied to the Cold Diffusion model.

## Repository Structure

Each folder in this repository corresponds to a major section of the paper.

- **naive_roi_weighted/**
  - Implementation of the initial Naive ROI-Weighted Loss approach.

- **sobel_roi_weighted/**
  - Extension of the ROI-Weighted Loss approach using Sobel edge detection to automate ROI selection.

- **cold_diffusion_roi/**
  - Application of the ROI-weighted loss concept to Cold Diffusion models.


## Important Notes

- Files have been reorganized to fit a GitHub layout and may require path debugging to run correctly.
- No model checkpoint files (`.ckpt`) or TensorBoard logs are included.
- `hparams.yaml` files from prior training runs are available for reproducibility.

## Summary

This repository contains the implementations and experiments that demonstrate the potential for improving MRI reconstruction quality by focusing model training on pathology regions through ROI-weighted learning.

Please refer to the specific subfolder for more detailed information and setup instructions for each experiment.

