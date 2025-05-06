# Naive ROI-Weighted Loss Implementation and Validation

This folder contains code related to the Naive ROI-Weighted Loss implementation for the fastMRI project. It includes the validation of the idea that upweighting the loss within a specified region of interest (ROI) improves reconstruction performance.

## Folder Structure

- **Validation Scripts**
  - `compare_alphas.ipynb`: Jupyter notebook to compare model performance across different ROI weighting values (alphas).
  - `roi_vs_base_runs.py`: Script to compare ROI-weighted models against baseline (non-weighted) models.

- **Modules**
  - Custom modules that extend Meta's fastMRI codebase to implement ROI-weighted loss functionality.
  - The original fastMRI training modules were modified to incorporate region-specific weighting in the loss calculation.

## Important Notes

- **Code Layout**: Files have been reorganized to fit a GitHub-style project structure.
- **Execution Warning**: The current file structure has not been fully validated to run as-is. Some debugging and path adjustments may be necessary before execution.
- **Checkpoints and Logs**: No model checkpoint files (`.ckpt`) or TensorBoard logs are included in this repository. Only the `hparams.yaml` configuration files from prior runs are provided for reference.

## Summary

This code serves as the foundation for evaluating the impact of ROI-weighted loss on image reconstruction quality. Additional setup and validation work will be required to run the code from its current form.

