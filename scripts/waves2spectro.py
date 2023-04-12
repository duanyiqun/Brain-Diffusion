import argparse
import io
import logging
import os
import re

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Image, Value
from braindiffusion.data import bci_comp_iv
from braindiffusion.utils.wave_spectron import Spectro, Spectro_STFT
from tqdm.auto import tqdm

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger("EEG waves to spectro images")


def main(args):
    spectro = Spectro_STFT(
        x_res=args.resolution[0],
        y_res=args.resolution[1],
        hop_length=args.hop_length,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        eeg_channels=args.eeg_channels,
    )
    # spectro.set_resolution(args.resolution[0], args.resolution[1])
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_generator = bci_comp_iv.DatasetLoader_BCI_IV_mix_subjects('train', datafolder=args.input_dir, train_aug=False)
    examples = []
    try:
        for dataslice in tqdm(dataset_generator):
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
            # spectro.load_wave(raw_wave=dataslice[0].unsqueeze(0).numpy())
            # latent = spectro.single_wave_to_latent(wave=spectro.wave)

            # for stft spectrogram
            raw_wave=dataslice[0].unsqueeze(0).numpy()
            latent = spectro.single_wave_to_latent(wave=raw_wave)
            # print(latent.shape)
            assert latent.shape[2] == args.resolution[1] and latent.shape[0] == args.eeg_channels, "Wrong resolution"
            # skip completely silent slices
            if all(np.frombuffer(latent.tobytes(), dtype=np.uint8) == 255):
                logger.warn("File slice is completely silent")
                continue
            # print(latent.shape)
            # print(np.frombuffer(latent.tobytes(), dtype=np.float64).shape)
            # print(dataslice[1])
            """
            with io.BytesIO() as output:
                np.save(output, latent)
                bytes = output.getvalue()
            """
            examples.extend(
                [
                    {
                        "image": latent.tobytes(),
                        # "audio_file": audio_file,
                        "label" : dataslice[1].astype(np.int16),
                        "subjects": "mixed",
                    }
                ]
            )
    except Exception as e:
        print(e)
    finally:
        if len(examples) == 0:
            logger.warn("No valid waves were found.")
            return
        ds = Dataset.from_pandas(
            pd.DataFrame(examples),
        )

    dataset_generator = bci_comp_iv.DatasetLoader_BCI_IV_mix_subjects('test', datafolder=args.input_dir, train_aug=False)
    examples = []
    try:
        for dataslice in tqdm(dataset_generator):
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
            # for mel spectrogram
            # spectro.load_wave(raw_wave=dataslice[0].unsqueeze(0).numpy())
            # latent = spectro.single_wave_to_latent(wave=spectro.wave)

            # for stft spectrogram
            raw_wave=dataslice[0].unsqueeze(0).numpy()
            latent = spectro.single_wave_to_latent(wave=raw_wave)

            # print(latent.shape)
            assert latent.shape[2] == args.resolution[1] and latent.shape[0] == args.eeg_channels, "Wrong resolution"
            # skip completely silent slices
            if all(np.frombuffer(latent.tobytes(), dtype=np.uint8) == 255):
                logger.warn("File slice is completely silent")
                continue
            # print(latent.shape)
            # print(np.frombuffer(latent.tobytes(), dtype=np.float64).shape)
            # print(dataslice[1])
            """
            with io.BytesIO() as output:
                np.save(output, latent)
                bytes = output.getvalue()
            """
            examples.extend(
                [
                    {
                        "image": latent.tobytes(),
                        # "audio_file": audio_file,
                        "label" : dataslice[1].astype(np.int16),
                        "subjects": "mixed",
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

    dsd = DatasetDict({"train": ds, 'test': ds_test})
    dsd.save_to_disk(os.path.join(args.output_dir))
    if args.push_to_hub:
        dsd.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset of Mel spectrograms from directory of audio files.")
    parser.add_argument("--input_dir", default='/home/yiqduan/Data/bci/EEG_MI_DARTS/Mudus_BCI/data/bci_iv/', type=str)
    parser.add_argument("--output_dir", type=str, default="./dataset/bci_iv/stft_64-24")
    parser.add_argument(
        "--resolution",
        type=str,
        default="24,64", # x_res should be n -1 due to dummy resolution
        help="Either square resolution or width,height.",
    )
    parser.add_argument("--hop_length", type=int, default=93)
    parser.add_argument("--push_to_hub", type=str, default=None)
    parser.add_argument("--sample_rate", type=int, default=250)
    parser.add_argument("--n_fft", type=int, default=127)
    parser.add_argument("--eeg_channels", type=int, default=22)
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
