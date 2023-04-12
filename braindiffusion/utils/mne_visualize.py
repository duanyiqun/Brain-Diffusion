import numpy as np
from mne import create_info, concatenate_raws
from mne.io import RawArray
from mne.channels import make_standard_montage
from mne.viz import plot_epochs_image
from PIL import Image


def visualize_eeg1020(data, sample_rate=250, duration=3, n_channels=22):
    # create MNE info object
    ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3',
                'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1',
                'Oz', 'O2', 'A1', 'A2']
    ch_types = ['eeg'] * n_channels
    sfreq = sample_rate  # Hz
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)

    # create MNE RawArray object
    # data_2d = np.expand_dims(data, axis=0)  # expand dimensions to match expected shape
    raw = RawArray(data, info)

    # read EEG electrode montage
    montage = make_standard_montage('standard_1020')

    # apply montage to RawArray object
    raw.set_montage(montage)
    psdfig = raw.plot_psd(fmax=50, show=False)
    rawfig = raw.plot(duration=duration, n_channels=n_channels, show=False)

    psdfig.canvas.draw()
    width, height = psdfig.get_size_inches() * psdfig.get_dpi()
    image = np.fromstring(psdfig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    pil_image = Image.fromarray(image)

    rawfig.canvas.draw()
    width, height = rawfig.get_size_inches() * rawfig.get_dpi()
    image = np.fromstring(rawfig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    pil_image_raw = Image.fromarray(image)

    return pil_image, pil_image_raw


def visualize_eeg128(data, sample_rate=500, duration=3, n_channels=105):
    # create MNE info object
    # ch_names =  ['Fp1', 'Fpz', 'Fp2', 'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFz', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', 'F9', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'F10', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T9', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'T10', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'O1', 'Oz', 'O2', 'Iz', 'OI1', 'OI2', 'FTT9', 'FTT7', 'FCC5', 'FCC3', 'FCC1', 'FCC2', 'FCC4', 'FCC6', 'FTT8', 'FTT10', 'TPP9', 'TPP7', 'CPP5', 'CPP3', 'CPP1', 'CPP2', 'CPP4', 'CPP6', 'TPP8', 'TPP10', 'PPO9', 'PPO7', 'PPO5', 'PPO3', 'PPO1', 'PPO2', 'PPO4', 'PPO6', 'PPO8', 'PPO10', 'POO9', 'POO3', 'POO4', 'POO10', 'POO1', 'POO2', 'POO5', 'POO6', 'POO7', 'POO8', 'Status']
    # ch_names = ch_names[:n_channels]

    # read EEG electrode montage
    montage = make_standard_montage('biosemi128')
    ch_names = montage.ch_names[:n_channels]

    ch_types = ['eeg'] * n_channels
    sfreq = sample_rate  # Hz
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)

    # create MNE RawArray object
    raw = RawArray(data, info)

    # apply montage to RawArray object
    raw.set_montage(montage)
    psdfig = raw.plot_psd(fmax=50, show=False)
    rawfig = raw.plot(duration=duration, n_channels=n_channels, show=False)

    psdfig.canvas.draw()
    width, height = psdfig.get_size_inches() * psdfig.get_dpi()
    image = np.fromstring(psdfig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    pil_image = Image.fromarray(image)

    rawfig.canvas.draw()
    width, height = rawfig.get_size_inches() * rawfig.get_dpi()
    image = np.fromstring(rawfig.canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    pil_image_raw = Image.fromarray(image)

    return pil_image, pil_image_raw


if __name__ == '__main__':
    data = np.random.rand(22, 1550)
    pil_image, pil_image_raw = visualize_eeg1020(data)
    pil_image.save('test.png')
    pil_image_raw.save('test_raw.png')

    data = np.random.rand(105, 8350)
    pil_image, pil_image_raw = visualize_eeg128(data)
    pil_image.save('test128.png')
    pil_image_raw.save('test128_raw.png')