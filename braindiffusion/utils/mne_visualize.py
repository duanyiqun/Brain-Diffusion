import numpy as np
import mne
import io
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
import gif


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

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 7), gridspec_kw={'height_ratios': [4, 1]})
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


def create_topo_heat_maps(feature_map, token_ids, channels_indices, skip_special_tokens=True):
    # Band names
    band_names = ['theta_1 (4–6 Hz)', 'theta_2 (6.5–8 Hz)', 'alpha_1 (8.5–10 Hz)', 'alpha_2 (10.5–13 Hz)', 'beta_1 (13.5–18 Hz)', 'beta_2 (18.5–30 Hz)', 'gamma_1 (30.5–40 Hz)', 'gamma_2 (40–49.5 Hz)']
    print("token_ids", len(token_ids))
    # Convert token IDs to text
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
    
    # Load biosemi128 layout
    montage = make_standard_montage('biosemi128')
    ch_names = montage.ch_names
    # Select the required channels
    selected_ch_names = [ch_names[i] for i in channels_indices]

    ch_types = ['eeg'] * len(channels_indices)
    sfreq = 500  # Hz
    
    
    # Create a list of 56 PIL Images
    images = []
    
    # Iterate through all time points
    for i in range(len(tokens)):
        fig, axes = plt.subplots(1, 8, figsize=(16, 2))
        canvas = FigureCanvas(fig)
        
        # Iterate through all bands
        for j, band_name in enumerate(band_names):
            # Create an empty info object with 105 channels
            info = create_info(ch_names=selected_ch_names, ch_types=ch_types, sfreq=sfreq)
            # Extract EEG data for a specific time point and band
            eeg_data = feature_map[j, :, i]
            # Create Epochs data
            info.set_montage(montage)
            epochs = mne.EpochsArray(np.array([[eeg_data],[eeg_data]]).transpose(0,2,1), info)
            
            # Plot the brain area topographic map
            im, _ = mne.viz.plot_topomap(eeg_data, epochs.info, axes=axes[j], show=False)
            axes[j].set_title(f"{band_name}")
        
        # Add text
        plt.suptitle(f"Token: {tokens[i]}", fontsize=14, y=1)
        
        # Save the plotted image as a PIL Image
        plt.tight_layout()

        # Draw the plot to the canvas buffer
        canvas.draw()

        # Convert the plot to a PIL image
        pil_image = Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
        plt.close(fig)
        images.append(pil_image)
        pil_image.save('test.png')
    
    return images


def stack_topomaps(heat_map_images, file_nmae='merged_image.png'):
    total_height = sum([img.height for img in heat_map_images])
    merged_image = Image.new('RGB', (heat_map_images[0].width, total_height))
    y_offset = 0
    for img in heat_map_images:
        merged_image.paste(img, (0, y_offset))
        y_offset += img.height

    merged_image.save(file_nmae)


def normalize_feature_map(feature_map, min_val=-8, max_val=10):
    """
    Normalize the values in the input feature_map to the range [min_val, max_val].

    :param feature_map: A numpy array of shape (8, 105, 56)
    :param min_val: The minimum value of the normalization range, default is -10
    :param max_val: The maximum value of the normalization range, default is 10
    :return: The normalized feature_map
    """
    # Calculate the minimum and maximum values of the input array
    input_min = np.min(feature_map)
    input_max = np.max(feature_map)

    # Normalize the array to the range [min_val, max_val]
    normalized_feature_map = (feature_map - input_min) * (max_val - min_val) / (input_max - input_min) + min_val

    return normalized_feature_map



if __name__ == '__main__':
    """
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
    """
    target_tokens = [101, 1045, 2572, 3467, 2000, 3422, 2070, 5561, 102, 0, 0, 0, 0, 0, 0, 0, 0]
    feature_map = np.random.rand(8, 105, 56)
    token_ids = target_tokens
    channels_indices = np.random.choice(range(128), 105, replace=False)

    # 调用函数
    feature_map = normalize_feature_map(feature_map)
    heat_map_images = create_topo_heat_maps(feature_map, token_ids, channels_indices)
    heat_map_images[0].save('test.png')
    stack_topomaps(heat_map_images)
    
