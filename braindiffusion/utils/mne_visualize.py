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







if __name__ == '__main__':
    data = np.random.rand(22, 1550)
    pil_image, pil_image_raw = visualize_eeg1020(data)
    pil_image.save('test.png')
    pil_image_raw.save('test_raw.png')