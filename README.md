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

<div align="left">

## Base Diffusion (BCI-IV Dataset)

</div>

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

Training on Mel spectrogram

```bash
accelerate launch --config_file config/accelerate_local.yaml \
scripts/train_unet.py \
    --dataset_name dataset/bci_iv/spectro_dp\
    --hop_length 50 \
    --n_fft 100 \
    --output_dir models/bciiv_mel_64 \
    --train_batch_size 2 \
    --num_epochs 100 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_warmup_steps 500 \
    --mixed_precision no \
    --original_shape 22,32,64 \
    --force_rescale 22,32,64 \
```

```bash
accelerate launch --config_file config/accelerate_local.yaml \
scripts/train_unet.py \
    --dataset_name dataset/bci_iv/stft_64-24 \
    --hop_length 93 \
    --output_dir models/bciiv_stft_64 \
    --train_batch_size 2 \
    --num_epochs 100 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_warmup_steps 500 \
    --mixed_precision no \
    --original_shape 22,24,64 \
    --force_rescale 22,32,64 \
    --stft
```
If you want use wandb to log metrics on webiste first run init 
```bash
wandb login
```


You can switch between logging in tensorboard or wandb by modify the accelerator config file. The default generator is wandb. The project name could be modified by given additional args ```--wandb_projects```. 
The visualization of the image is saved locally to prevent breaking the training process. The visualization of the training process could be found in modesl/name/visualization file. The saving interval could be modified by ```--save_images_epochs```. 
Here are examples of the visualization nearly end of the training, mainly include the spectrogram, and the reconstructed wave.


<div align="center">
<img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"  src=./visualization/vismel_bci_iv.png width = "200" alt="图片名称" align=center /> <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);"  src=./visualization/psd_mapwave_97.png width = "570" alt="图片名称" align=center />
</br>
</div>


<div align="left">
</br>

## Conditioned Diffusion (Zuco Dataset)

</div>


## Data Preparation

* Download raw data from ZuCo
  * Download ZuCo v1.0 for 'task1-SR','task2-NR','task3-TSR' from https://osf.io/q3zws/files/ under 'OSF Storage' root,
  unzip and move all .mat files to /dataset/ZuCo/task1-SR/Matlab_files,/dataset/ZuCo/task2-NR/Matlab_files,/dataset/ZuCo/task3-TSR/Matlab_files respectively.
  * Download ZuCo v2.0 'Matlab files' for 'task1-NR' from https://osf.io/2urht/files/ under 'OSF Storage' root, unzip and move all .mat files to /dataset/ZuCo/task2-NR-2.0/Matlab_files.
* Preparation scripts for eye fixation sliced data
  * Modify the data paths and roots in [construct_wave_mat_to_pickle_v1.py](util/construct_dataset_mat_to_pickle_v1.py) and [construct_wave_mat_to_pickle_v2.py](util/construct_dataset_mat_to_pickle_v2.py)
  * Run [scripts/data_preparation_freq.sh](scripts/data_preparation_freq.sh) for eye-tracking fixation sliced EEG waves. 
* Preparation scripts for raw waves
  * Modify the roots variable in  [scripts/data_preparation_wave.sh](scripts/data_preparation_wave.sh) and run the scripts, -r suggest the input dir, -o denotes the output dir.
  ```python3 
  ./util/construct_wave_mat_to_pickle_v1.py -t task3-TSR -r /projects/CIBCIGroup/00DataUploading/yiqun/bci -o /projects/CIBCIGroup/00DataUploading/yiqun/bci/ZuCo/dewave_sent
  ```
  The processed data will be saved in the output root. An direction of this root is like this:
  ```sh
  .
  └── ZuCo
      ├── task1-SR
      │   └── pickle
      │       └── task1-SR-dataset.pickle
      ├── task2-NR
      │   └── pickle
      │       └── task2-NR-dataset.pickle
      ├── task2-NR-2.0
      │   └── pickle
      │       └── task2-NR-2.0-dataset.pickle
      └── task3-TSR
          └── pickle
              └── task3-TSR-dataset.pickle
  ```

* Generate Spectros: Please note this may cost 100G+ memory. If you have enough memory, you can run the following command to generate spectrograms. Or you may modify the code to generate spectrograms each time for a split by comment out data parts in [scripts/wave2spectro_zuco.py](scripts/wave2spectro_zuco.py).
  ```sh 
  python scripts/wave2spectro_zuco.py --resolution 96,96 --input_dir path-to-preprocessed-zuco --output_dir path-to-output-data --hop_length 75 --sample_rate 500 --n_fft 100
  ```
  please note that the hop_length is the number of samples between the starts of consecutive frames. The sample_rate is the sampling frequency of the wave. The n_fft is the number of samples in each frame. The resolution is the resolution of the spectrogram. 
  
  Please also not that you can modify this scripts to swich between generate mel **spectro** and **stft sprctro**. 


## Training

* Encode condition text files for training. Here we use Berttokenizer to encode the text. The pretrained model is from [bert-base-uncased](https://huggingface.co/bert-base-uncased). The encoded text embedding cache will be saved in the output root.
  ```sh
  python scripts/encode_condition.py --input_dir path-to-preprocessed-zuco --output_dir path-to-output-data --task_name task1-SR
  ```

* Training scripts
  
  The current training version support both diffusion on mel spectrogram and STFT spectrogram. Here I give examples below:

  * Training on STFT spectrogram Data

  ```sh
  CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file config/accelerate_local.yaml \
  scripts/train_unet_conditioned.py \
      --dataset_name dataset/zuco/stft-96-64 \
      --stft \
      --hop_length 69 \
      --eeg_channels 105 \
      --n_fft 127 \
      --sample_rate 500 \
      --output_dir models/zuco-stft9664-test \
      --train_batch_size 2 \
      --num_epochs 100 \
      --gradient_accumulation_steps 1 \
      --learning_rate 1e-4 \
      --lr_warmup_steps 500 \
      --mixed_precision fp16
  ```
  * Training on Mel spectrogram Data

  ```sh
  CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file config/accelerate_local.yaml \
  scripts/train_unet_conditioned.py \
      --dataset_name dataset/zuco/condition \
      --hop_length 75 \
      --eeg_channels 105 \
      --n_fft 100 \
      --sample_rate 500 \
      --output_dir models/zuco-mel-test \
      --train_batch_size 2 \
      --num_epochs 100 \
      --gradient_accumulation_steps 1 \
      --learning_rate 1e-4 \
      --lr_warmup_steps 500 \
      --max_freq 64 \
      --original_shape 105,112,96 \
      --force_rescale 105,96,64 \
      --mixed_precision fp16 
  ```

  ```sh
  CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file config/accelerate_local.yaml \
  scripts/train_unet_freq.py \
      --dataset_name dataset/zuco/freqmap_32_840 \
      --hop_length 50 \
      --eeg_channels 1 \
      --n_fft 100 \
      --sample_rate 500 \
      --output_dir models/zuco-freq_map_32840 \
      --train_batch_size 2 \
      --num_epochs 100 \
      --gradient_accumulation_steps 1 \
      --learning_rate 1e-4 \
      --lr_warmup_steps 500 \
      --max_freq 315 \
      --original_shape 1,32,840 \
      --force_rescale 1,32,840 \
      --mixed_precision fp16 \
      --debug
  ```

  ```sh
  CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file config/accelerate_local.yaml \
  scripts/train_unet_freq_2d.py \
      --dataset_name dataset/zuco/freqmap_8_105_56 \
      --hop_length 50 \
      --eeg_channels 8 \
      --n_fft 100 \
      --sample_rate 500 \
      --output_dir models/zuco-freq_map_810556_nora \
      --train_batch_size 3 \
      --num_epochs 100 \
      --gradient_accumulation_steps 1 \
      --learning_rate 1e-4 \
      --lr_warmup_steps 500 \
      --max_freq 32 \
      --original_shape 8,105,56 \
      --force_rescale 8,105,56 \
      --mixed_precision fp16 \
      --debug
  ```



