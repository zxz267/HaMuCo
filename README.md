# Hand Pose Estimation via Multiview Collaborative Self-Supervised Learning

## Data Preparation
### 1. Download the [HanCo](https://lmb.informatik.uni-freiburg.de/resources/datasets/HanCo.en.html) dataset from the official website.
  1. https://lmb.informatik.uni-freiburg.de/data/HanCo/HanCo_rgb.zip
  2. https://lmb.informatik.uni-freiburg.de/data/HanCo/HanCo_xyz.zip
  3. https://lmb.informatik.uni-freiburg.de/data/HanCo/HanCo_shape.zip
  4. https://lmb.informatik.uni-freiburg.de/data/HanCo/HanCo_calib_meta.zip
  5. https://lmb.informatik.uni-freiburg.de/data/HanCo/HanCo_rgb_merged.zip
### 2. We provide the 2D pseudo labels generated from OpenPose in `./data/HanCo/HaMuCo_*.zip`.
### 3. Unzip files and organize the data as follows:
```
${ROOT}  
|-- data  
|   |-- HanCo
|   |   |-- calib
|   |   |-- rgb 
|   |   |-- rgb_2d_keypoints
|   |   |-- rgb_merged
|   |   |-- xyz
```
## Installation
### Requirements
- Python=3.7
- PyTorch=1.9.1+cu111
- torchgeometry (need some slight changes following [here](https://github.com/mks0601/I2L-MeshNet_RELEASE/issues/6#issuecomment-675152527).)

### Setup with Conda
```
conda create -n hamuco python=3.7
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
cd HaMuCo
pip install -r ./requirements.txt
```

## Training
### 1. Run `./train.py` to train and evaluate on the HanCo dataset.