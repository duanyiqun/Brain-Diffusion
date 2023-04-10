import argparse
import io
import logging
import os
import re

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Image, Value
from braindiffusion.data import zuco_nr
from braindiffusion.utils.wave_spectron import Spectro
from tqdm.auto import tqdm
import pickle5 as pickle
import json
import matplotlib.pyplot as plt
from glob import glob
from transformers import BartTokenizer, BertTokenizer, BertTokenizerFast
from tqdm import tqdm
from fuzzy_match import match
from fuzzy_match import algorithims
import torch


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger("EEG waves to spectro images")


def main(args):
    spectro = Spectro(
        x_res=args.resolution[0],
        y_res=args.resolution[1],
        hop_length=args.hop_length,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        eeg_channels=args.eeg_channels,
    )
    # spectro.set_resolution(args.resolution[0], args.resolution[1])
    os.makedirs(args.output_dir, exist_ok=True)

    whole_dataset_dicts = []
        
    dataset_path_task1 = '{}/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle'.format(args.input_dir)
    with open(dataset_path_task1, 'rb') as handle:
        whole_dataset_dicts.append(pickle.load(handle))
    
    
    dataset_path_task2 = '{}/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle'.format(args.input_dir)
    with open(dataset_path_task2, 'rb') as handle:
        whole_dataset_dicts.append(pickle.load(handle))

    dataset_path_task3 = '{}/ZuCo/task3-TSR/pickle/task3-TSR-dataset.pickle'.format(args.input_dir)
    with open(dataset_path_task3, 'rb') as handle:
        whole_dataset_dicts.append(pickle.load(handle))
    
    dataset_path_task2_v2 = '{}/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle'.format(args.input_dir)
    with open(dataset_path_task2_v2, 'rb') as handle:
        whole_dataset_dicts.append(pickle.load(handle))
    
    
    print()
    for key in whole_dataset_dicts[0]:
        print(f'task2_v2, sentence num in {key}:',len(whole_dataset_dicts[0][key]))
    print()

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    dataset_setting = 'unique_sent'
    subject_choice = 'ALL'
    print(f'![Debug]using {subject_choice}')
    eeg_type_choice = 'GD'
    print(f'[INFO]eeg type {eeg_type_choice}') 
    bands_choice = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] 
    print(f'[INFO]using bands {bands_choice}')

    train_set = zuco_nr.ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
    dev_set = zuco_nr.ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
    test_set = zuco_nr.ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)

    examples = []
    try:
        for dataslice in tqdm(test_set):
            """dataslice is a tuple of (data, label)
            try:
                print(dataslice.size())
                spectro.load_wave(raw_wave=dataslice[0])
            except KeyboardInterrupt:
                raise
            except:
                continue
            """
            # for slice in range(spectro.get_number_of_slices()):
            seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG = dataslice
            # print(sent_level_EEG.shape)
            # print(seq_len)
            # print(input_masks.shape)
            # print(input_mask_invert.shape)
            # print(target_ids.shape)
            # print(target_mask.shape)
            # print(sentiment_labels.shape)
            # print(sent_level_EEG.shape)

            spectro.load_wave(raw_wave=sent_level_EEG.unsqueeze(0).numpy())
            latent = spectro.single_wave_to_latent(wave=spectro.wave)
            # print(latent.shape)
            assert latent.shape[2] == args.resolution[1] and latent.shape[0] == args.eeg_channels, "Wrong resolution"
            # skip completely silent slices
            if all(np.frombuffer(latent.tobytes(), dtype=np.uint8) == 255):
                logger.warn("File slice is completely silent")
                continue
            # print(latent.shape)
            # print(np.frombuffer(latent.tobytes(), dtype=np.float64).shape)
            # print(dataslice[1])
            examples.extend(
                [
                    {
                        "image": latent.tobytes(),
                        # "audio_file": audio_file,
                        "label" : None,
                        "subjects": "mixed",
                        "eeg_type": "GD",
                        "seq_len": seq_len,
                        "input_masks": input_masks.tolist(),
                        "input_mask_invert": input_mask_invert.tolist(),
                        "target_ids": target_ids.tolist(),
                        "target_mask": target_mask.tolist(),
                    }
                ]
            )
    except Exception as e:
        print(e)
    finally:
        if len(examples) == 0:
            logger.warn("No valid waves were found.")
            return
        ds_test = Dataset.from_pandas(
            pd.DataFrame(examples),
        )
    dsd = DatasetDict({"test": ds_test})
    dsd.save_to_disk(os.path.join(args.output_dir))
    if args.push_to_hub:
        dsd.push_to_hub(args.push_to_hub)

    try:
        for dataslice in tqdm(dev_set):
            """dataslice is a tuple of (data, label)
            try:
                print(dataslice.size())
                spectro.load_wave(raw_wave=dataslice[0])
            except KeyboardInterrupt:
                raise
            except:
                continue
            """
            # for slice in range(spectro.get_number_of_slices()):
            seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG = dataslice
            # print(sent_level_EEG.shape)
            # print(seq_len)
            # print(input_masks.shape)
            # print(input_mask_invert.shape)
            # print(target_ids.shape)
            # print(target_mask.shape)
            # print(sentiment_labels.shape)
            # print(sent_level_EEG.shape)

            spectro.load_wave(raw_wave=sent_level_EEG.unsqueeze(0).numpy())
            latent = spectro.single_wave_to_latent(wave=spectro.wave)
            # print(latent.shape)
            assert latent.shape[2] == args.resolution[1] and latent.shape[0] == args.eeg_channels, "Wrong resolution"
            # skip completely silent slices
            if all(np.frombuffer(latent.tobytes(), dtype=np.uint8) == 255):
                logger.warn("File slice is completely silent")
                continue
            # print(latent.shape)
            # print(np.frombuffer(latent.tobytes(), dtype=np.float64).shape)
            # print(dataslice[1])
            examples.extend(
                [
                    {
                        "image": latent.tobytes(),
                        # "audio_file": audio_file,
                        "label" : None,
                        "subjects": "mixed",
                        "eeg_type": "GD",
                        "seq_len": seq_len,
                        "input_masks": input_masks.tolist(),
                        "input_mask_invert": input_mask_invert.tolist(),
                        "target_ids": target_ids.tolist(),
                        "target_mask": target_mask.tolist(),
                    }
                ]
            )
    except Exception as e:
        print(e)
    finally:
        if len(examples) == 0:
            logger.warn("No valid waves were found.")
            return
        ds_dev = Dataset.from_pandas(
            pd.DataFrame(examples),
        )
    
    dsd = DatasetDict({"test": ds_test, "dev": ds_dev})
    dsd.save_to_disk(os.path.join(args.output_dir))
    if args.push_to_hub:
        dsd.push_to_hub(args.push_to_hub)
    
    try:
        for dataslice in tqdm(train_set):
            """dataslice is a tuple of (data, label)
            try:
                print(dataslice.size())
                spectro.load_wave(raw_wave=dataslice[0])
            except KeyboardInterrupt:
                raise
            except:
                continue
            """
            # for slice in range(spectro.get_number_of_slices()):
            seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG = dataslice
            # print(sent_level_EEG.shape)
            # print(seq_len)
            # print(input_masks.shape)
            # print(input_mask_invert.shape)
            # print(target_ids.shape)
            # print(target_mask.shape)
            # print(sentiment_labels.shape)
            # print(sent_level_EEG.shape)

            spectro.load_wave(raw_wave=sent_level_EEG.unsqueeze(0).numpy())
            latent = spectro.single_wave_to_latent(wave=spectro.wave)
            # print(latent.shape)
            assert latent.shape[2] == args.resolution[1] and latent.shape[0] == args.eeg_channels, "Wrong resolution"
            # skip completely silent slices
            if all(np.frombuffer(latent.tobytes(), dtype=np.uint8) == 255):
                logger.warn("File slice is completely silent")
                continue
            # print(latent.shape)
            # print(np.frombuffer(latent.tobytes(), dtype=np.float64).shape)
            # print(dataslice[1])
            examples.extend(
                [
                    {
                        "image": latent.tobytes(),
                        # "audio_file": audio_file,
                        "label" : None,
                        "subjects": "mixed",
                        "eeg_type": "GD",
                        "seq_len": seq_len,
                        "input_masks": input_masks.tolist(),
                        "input_mask_invert": input_mask_invert.tolist(),
                        "target_ids": target_ids.tolist(),
                        "target_mask": target_mask.tolist(),
                    }
                ]
            )
    except Exception as e:
        print(e)
    finally:
        if len(examples) == 0:
            logger.warn("No valid waves were found.")
            return
        ds_train = Dataset.from_pandas(
            pd.DataFrame(examples),
        )

    dsd = DatasetDict({"test": ds_test, "dev": ds_dev, "train": ds_train})
    dsd.save_to_disk(os.path.join(args.output_dir))
    if args.push_to_hub:
        dsd.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset of Mel spectrograms from directory of audio files.")
    parser.add_argument("--input_dir", default='/projects/CIBCIGroup/00DataUploading/yiqun/bci/ZuCo/dewave_sent', type=str)
    parser.add_argument("--output_dir", type=str, default="./dataset/zuco/spectro_dp")
    parser.add_argument(
        "--resolution",
        type=str,
        default="96,96", # x_res should be n -1 due to dummy resolution
        help="Either square resolution or width,height.",
    )
    parser.add_argument("--hop_length", type=int, default=75)
    parser.add_argument("--push_to_hub", type=str, default=None)
    parser.add_argument("--sample_rate", type=int, default=500)
    parser.add_argument("--n_fft", type=int, default=100)
    parser.add_argument("--eeg_channels", type=int, default=105)
    args = parser.parse_args()
    print(args)

    if args.input_dir is None:
        raise ValueError("You must specify an input directory for the preprocessed EEG mat files.")

    # Handle the resolutions.
    try:
        args.resolution = (int(args.resolution), int(args.resolution))
    except ValueError:
        try:
            args.resolution = tuple(int(x) for x in args.resolution.split(","))
            if len(args.resolution) != 2:
                raise ValueError
        except ValueError:
            raise ValueError("Resolution must be a tuple of two integers or a single integer.")
    assert isinstance(args.resolution, tuple)

    main(args)
