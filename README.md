# GDEC

## Project Running Environment:

- **Python Version:** 3.8
- **Library Dependencies:**
  - PyTorch: 1.13.0
  - CUDA Version: 11.7
  - Torch-Geometric: 2.3.1

## Python Files Overview:

1. **channel.py:** Responsible for generating gene pathway adjacency matrix files.
2. **utils.py:** Utilized to read gene feature files.
3. **GDEC.py:** This is the primary framework file. The corresponding model can be found in the "model" folder. It facilitates the generation of a low-dimensional feature gene file, named "x_tsne.csv".

## R Files Code:

1. **Elbow diagram.R:** Draw elbow diagram analysis.
2. **PCA.R:** Principal component analysis.
3. **Survival.R:** Drawing a Survival Curve.
4. **rf.R:** Random forest feature importance analysis.
