import numpy as np
from mne import create_info, concatenate_raws
from mne.io import RawArray
from mne.channels import make_standard_montage
from mne.viz import plot_epochs_image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
from transformers import BertTokenizerFast



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

def visualize_feature_map(feature_map):
    """
    Visualize a feature map with shape (56, 840) and return a PIL image with color bar.

    :param feature_map: A 2D NumPy array with shape (56, 840)
    :return: A PIL image object
    """

    # if feature_map.shape != (56, 840):
    #     raise ValueError("Feature map must have shape (56, 840)")

    fig, ax = plt.subplots(figsize=(12, 4))
    canvas = FigureCanvas(fig)

    img = ax.imshow(feature_map, cmap="viridis", aspect="auto")
    fig.colorbar(img, ax=ax)

    ax.set_title("Feature Map Visualization")
    ax.set_xlabel("Frequency Features Dimension")
    ax.set_ylabel("Features tokens")

    plt.tight_layout()

    # Draw the plot to the canvas buffer
    canvas.draw()

    # Convert the plot to a PIL image
    pil_image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
    plt.close(fig)

    return pil_image


def visualize_feature_map_withtoken(feature_map, target_tokens):
    """
    Visualize a feature map with shape (56, 840) and return a PIL image with color bar and target tokens as a sentence.

    :param feature_map: A 2D NumPy array with shape (56, 840)
    :param target_tokens: A list of tokenized words from BERT uncased tokenizer
    :return: A PIL image object
    """

    # Revert tokens to a sentence using BertTokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    sentence = tokenizer.decode(target_tokens, skip_special_tokens=True)

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 6), gridspec_kw={'height_ratios': [4, 1]})
    canvas = FigureCanvas(fig)

    img = ax1.imshow(feature_map, cmap="viridis", aspect="auto")
    fig.colorbar(img, ax=ax1)

    ax1.set_title("Feature Map Visualization")
    ax1.set_xlabel("Frequency Features Dimension")
    ax1.set_ylabel("Features tokens")

    # Display the sentence under the feature map
    ax2.axis('off')
    ax2.text(0.5, 0.5, sentence, wrap=True, horizontalalignment='center', fontsize=12)

    plt.tight_layout()

    # Draw the plot to the canvas buffer
    canvas.draw()

    # Convert the plot to a PIL image
    pil_image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
    plt.close(fig)

    return pil_image




if __name__ == '__main__':
    data = np.random.rand(22, 1550)
    pil_image, pil_image_raw = visualize_eeg1020(data)
    pil_image.save('test.png')
    pil_image_raw.save('test_raw.png')

    data = np.random.rand(105, 8350)
    pil_image, pil_image_raw = visualize_eeg128(data)
    pil_image.save('test128.png')
    pil_image_raw.save('test128_raw.png')


    data = np.random.rand(56, 840)
    pil_image = visualize_feature_map(data)
    pil_image.save('test_feature.png')

    data = np.random.rand(56, 840)
    target_tokens = [101, 1045, 2572, 3467, 2000, 3422, 2070, 5561, 102] 
    pil_image = visualize_feature_map_withtoken(data, target_tokens)
    pil_image.save('test_feature.png')