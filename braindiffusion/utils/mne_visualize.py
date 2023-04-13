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
    sentence = tokenizer.decode(target_tokens, skip_special_tokens=False)

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


def create_topo_heat_maps(feature_map, token_ids, channels_indices, skip_special_tokens=False):
    # 频带名称
    band_names = ['theta_1', 'theta_2', 'alpha_1', 'alpha_2', 'beta_1', 'beta_2', 'gamma_1', 'gamma_2']
    
    # 将token ID转换为文本
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
    
    # 加载biosemi128布局
    # layout = mne.channels.make_standard_montage('biosemi128')
    # ch_names = layout.ch_names
    
    montage = make_standard_montage('biosemi128')
    ch_names = montage.ch_names
    # 选择需要的通道
    selected_ch_names = [ch_names[i] for i in channels_indices]

    ch_types = ['eeg'] * len(channels_indices)
    sfreq = 500  # Hz
    
    
    # 创建包含56张PIL Image的列表
    images = []
    
    # 遍历所有时刻
    for i in range(len(tokens)):
        fig, axes = plt.subplots(1, 8, figsize=(16, 2))
        canvas = FigureCanvas(fig)
        
        # 遍历所有频带
        for j, band_name in enumerate(band_names):
            # 创建一个包含105个通道的空info对象
            info = create_info(ch_names=selected_ch_names, ch_types=ch_types, sfreq=sfreq)
            # 提取特定时刻和频带的EEG数据
            eeg_data = feature_map[j, :, i]
            # print(len(channels_indices))
            # print(np.array([[eeg_data],[eeg_data]]).transpose(0,2,1).shape)
            # 创建Epochs数据
            info.set_montage(montage)
            epochs = mne.EpochsArray(np.array([[eeg_data],[eeg_data]]).transpose(0,2,1), info)
            
            # 绘制脑区拓扑地形图
            im, _ = mne.viz.plot_topomap(eeg_data, epochs.info, axes=axes[j], show=False)
            axes[j].set_title(f"{band_name}")
        
        # 添加文本
        # print(i)
        # print(tokens[i])
        plt.suptitle(f"Token: {tokens[i]}", fontsize=14, y=1)
        
        # 将绘制的图像保存为PIL Image
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
    # 计算总的高度
    total_height = sum([img.height for img in heat_map_images])

    # 创建一个新的空白图像，其宽度等于第一张图像的宽度，高度等于所有图像的总高度
    merged_image = Image.new('RGB', (heat_map_images[0].width, total_height))

    # 纵向拼接图像
    y_offset = 0
    for img in heat_map_images:
        merged_image.paste(img, (0, y_offset))
        y_offset += img.height

    # 显示拼接后的图像
    # merged_image.show()

    # 如果需要保存拼接后的图像
    merged_image.save(file_nmae)


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
    heat_map_images = create_topo_heat_maps(feature_map, token_ids, channels_indices)
    heat_map_images[0].save('test.png')
    stack_topomaps(heat_map_images)
