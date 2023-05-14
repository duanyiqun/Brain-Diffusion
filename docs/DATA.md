
# Data Preparation

## Uncodintional Diffusion (BCI-IV Dataset)

Download the BCI-IV data under the dataset folder (apply for liscence). The data is from [here](https://www.bbci.de/competition/iv/#dataset2). The data should be preprossed in .mat format. We provide example file of a single subject. 

This step convert the EEG waves into spectro latent and create a dataset for accelerate the training. This convertion is inspired by in audio conversion. To accelerate the conversion we use the mel spectro from audio area. 

```sh
python scripts/wave2spectro.py \
        --resolution 31,64 \
        --hop_length 50 \
        --input_dir path-to-mat-files \
        --output_dir path-to-output-data
```
 


## Conditioned Diffusion (Zuco Dataset)

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

