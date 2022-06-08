# Implementations for RandStainNA

This is the official implementation of the paper "RandStainNA: Learning Stain-Agnostic Features from Histology Slides by Bridging Stain Augmentation and Normalization"

## 1. Method overview
RandStainNA is a augmentation scheme to train a stain agnostic deep learning model for histology analysis. 
It unifies Stain Normalization and Stain Augmentation by constraining variable stain styles in a practicable range. 
The RandStainNA is applicable to stain normalization in a collection of color spaces i.e., HED, HSV, LAB.

## 2. Organization
The codes are organized as follows:
- `visualization`: provides the visualization of stain normalizations, stain augmentations, and RandStainNA.
    - `origin`: the raw images.
    - `randstainna`: visualizations of images augmented by RandStainNA.
    - `stain_augmentation`: visualizations of images augmented by stain augmentation. 
    - `stain_augmentation`: visualizations of images processed by stain normalization. 
- `preprocess`: perform the statistics in the pre-processing stage. Results should be presented as [`CRC_LAB_randomTrue_n0.yaml`](./CRC_LAB_randomTrue_n0.yaml).
- `randstainna.py`: we warp RandStainNA in this file. 
- `main.py`: provides the use case for RandStainNA in `randstainna.py`.


## 3. Set up the environmets
Run 
'''
pip install -r requirements.txt
'''
to install the required libraries. Note that `opencv-python`, `pyyaml` and `scikit-image` are necessary for `RandStainNA`; while we use `fitter` for performing the statistic analysis in the pre-processing stage. 

