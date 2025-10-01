# Fixes for MAESTER

Example usage for Fluo-N2DL-HeLa from celltrackingchallenge.net:

### Get data
```
git clone https://github.com/cdalinghaus/MAESTER_fix
cd MAESTER_fix/
wget https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DL-HeLa.zip
unzip Fluo-N2DL-HeLa.zip
python preprocess.py
```

### Run training

Important: For 2d+time data (like Fluo-N2DL-HeLa), we want to only slice along the time axis. [This repository is configured to only slice along the time axis](https://github.com/cdalinghaus/MAESTER_fix/blob/main/MAESTER/dataset.py#L94).  
To slice along all dimensions (for 3d volume data, like in the original MAESTER), [re-enable random selection of slice axis](https://github.com/cdalinghaus/MAESTER_fix/blob/main/MAESTER/dataset.py#L93)

```
mkdir checkpoints
cd MAESTER
SLURM_NODEID=0 SLURM_LOCALID=0 CUDA_VISIBLE_DEVICES=0 python train.py --model_config_dir ./config --model_config_name default.yaml --world_size 1 --logdir ./checkpoints
```

Using 4x A100:
```
srun -p grete:shared -N 1 -n 4 --cpus-per-task=8 \
     --gpus=A100:4 --constraint=80gb_vram \
     --time=12:00:00 --mem=0 --preserve-env \
     bash -lc 'source ~/.bashrc && conda activate maester && \
               MASTER_ADDR=$(hostname) MASTER_PORT=29500 WORLD_SIZE=4 \
               RANK=$SLURM_PROCID LOCAL_RANK=$SLURM_LOCALID \
               python train.py \
                 --model_config_dir ./config \
                 --model_config_name default.yaml \
                 --init_method env:// \
                 --world_size 4 \
                 --logdir ./checkpoints'
```

### Run inference
```
cd examples
python run_inference.py
```

#### Example results:
- 2 clusters:
<img width="3600" height="2400" alt="image" src="https://github.com/user-attachments/assets/c7bd3535-5d82-4943-bd66-d9fcbe1fa5d5" />



- 6 clusters:
<img width="3600" height="2400" alt="image" src="https://github.com/user-attachments/assets/a8ec9239-5abc-4319-9747-9f4e0336eed4" />



# CVPR2023 Highlight | MAESTER: Masked Autoencoder Guided Segmentation at Pixel Resolution for Accurate, Self-Supervised Subcellular Structure Recognition

Check out the [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Xie_MAESTER_Masked_Autoencoder_Guided_Segmentation_at_Pixel_Resolution_for_Accurate_CVPR_2023_paper.pdf)!
and our [Youtube Talk](https://youtu.be/MB2J9eeR0zc)!

## 💥 Introduction

We introduce MAESTER (**M**asked **A**uto**E**ncoder guided **S**egmen**T**ation at pix**E**l **R**esolution), a self-supervised method for accurate, subcellular structure segmentation at pixel resolution. MAESTER treats volume electron microscopy(vEM) image segmentation as a representation learning and clustering problem. Specifically, MAESTER learns semantically meaningful token representations of multi-pixel image patches while simultaneously maintaining a sufficiently large field of view for contextual learning. We also develop a _cover-and-stride_ inference strategy to achieve pixel-level subcellular strueture segmentation.

![](./figs/intro.jpeg)

## ⚙️ Installation

- Clone the repository:

```
git clone https://github.com/bowang-lab/MAESTER
```

- Set up the environment:

```
poetry install
poetry shell
pip install torch==2.0.1 torchvision==0.15.2
```

- Download the trained model for demo

  - Google drive: `https://drive.google.com/drive/folders/143W_VSl5ONE3NGbnI0i19S8lBRml7lRz?usp=sharing`
  - Put it under `./MAESTER/model_weights/`

- Dataset:

  - Download the [betaSeg](https://rupress.org/jcb/article/220/2/e202010039/211599/3D-FIB-SEM-reconstruction-of-microtubule-organelle) dataset by:

  ```
  wget https://cloud.mpi-cbg.de/index.php/s/UJopHTRuh6f4wR8/download
  ```

  - and unzip the dataset, put it under `./MAESTER/data/`

## 🎉 Example

- Check out our detailed demo:
  - Inference with MAESTER `./examples/inference_demo.ipynb`.

## 📝 To-do

- [x] Add inference demo
- [ ] Add scalable inference example
- [ ] Add DDP training example

## 📄 Citation

```
@InProceedings{Xie_2023_CVPR,
    author    = {Xie, Ronald and Pang, Kuan and Bader, Gary D. and Wang, Bo},
    title     = {MAESTER: Masked Autoencoder Guided Segmentation at Pixel Resolution for Accurate, Self-Supervised Subcellular Structure Recognition},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {3292-3301}
}
```

## Acknowledgement

- This repository is built upon [MAE](https://github.com/facebookresearch/mae).
