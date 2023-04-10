<div align="center">


# Diffusion Models for Brain Waves
**A Diffusion Framework for Dynamics/EEG Signals Synthesizing or Denoising**

______________________________________________________________________

WIP ...

 [![python](https://img.shields.io/badge/python-%20%203.9-blue.svg)]()
[![license](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/duanyiqun/DiffusionDepth/blob/main/LICENSE)

</div> 





______________________________________________________________________
  

## Installation

Please install required pip packages
```ssh
pip install transformers==4.27.1
pip install diffuser 
```
If warning of any unfound versions , just install the latest version with conda. Also recommend to use newer version of transformers 


```sh
git clone https://github.com/duanyiqun/Brain-Diffusion.git
cd Brain-Diffusion
pip install -e .
pip instlal -r requirements.txt
```


## Data Preparation

Download the BCI-IV data under the dataset folder (apply for liscence). The data is from [here](https://www.bbci.de/competition/iv/#dataset2). The data should be preprossed in .mat format. We provide example file of a single subject. 

This step convert the EEG waves into spectro latent and create a dataset for accelerate the training. This convertion is inspired by in audio conversion. To accelerate the conversion we use the mel spectro from audio area. 

```sh
python scripts/wave2spectro.py \
        --resolution 31,64 \
        --hop_length 50 \
        --input_dir path-to-mat-files \
        --output_dir path-to-output-data
```

## Training

The training entry scripts:
```bash
accelerate launch --config_file config/accelerate_local.yaml \
scripts/train_unet.py \
    --dataset_name dataset/bci_iv/spectro_dp \
    --hop_length 50 \
    --output_dir models/spectro_dp-3264 \
    --train_batch_size 2 \
    --num_epochs 100 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_warmup_steps 500 \
    --mixed_precision no
```
If you want use wandb to log metrics on webiste first run init 
```bash
wandb login
```

## Inference

Please config args as below as you training file give. Also set the runid to your trained checkpoints folder. Given the example below please set ```runid=34854f3ed38711edb808e4434b7714aa``` A sample structure under the log/experiment_name folder could be:

```sh
.
└── 34854f3ed38711edb808e4434b7714aa
    ├── checkpoints
    ├── configs.yaml
    ├── indicators.yaml
    ├── pids
    ├── run.yaml
    ├── source.diff
    └── wandb
```

The inference entry scripts:
```bash
python sample_save.py
```


The generated example of denoised EEG signals (blue) with separated noise (green). This method may also be used for other time sequence signals. Below is a sampled animation of the generated process of sampling sythetic EEG signals from noise given random noise and subjects. The noise decrease through time steps. The clean signal then rapidly take the lead of the whole process. 

<div align="center">
<img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"  src=./visualization/noise_curve_animate_subject_[0].gif width = "400" alt="图片名称" align=center /> <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"  src=./visualization/noise_curve_animate_subject_[1].gif width = "400" alt="图片名称" align=center />
</div>
