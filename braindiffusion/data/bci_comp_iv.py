import torch
import numpy as np
import scipy.io as sio
import csv
from braindiffusion.utils import wave_spectron
from braindiffusion.utils import sampling_func

class DatasetLoader_BCI_IV_signle(torch.utils.data.Dataset):

    def __init__(self, setname, datafolder=None, train_aug=False, subject_id=3):
        self.wave_max = 30
        self.max_sampledepth = 255 # closer to image, not int type
        subject_id = subject_id
        if datafolder is None:
            data_folder = '../data'
        else:
            data_folder = datafolder
        data = sio.loadmat(data_folder + "/single_sep/single_subject_data_" + str(subject_id) + ".mat")
        test_X = data["test_x"][:, :, 750:1500]  # [trials, channels, time length]
        train_X = data["train_x"][:, :, 750:1500]

        test_y = data["test_y"].ravel()
        train_y = data["train_y"].ravel()

        train_y -= 1
        test_y -= 1
        window_size = 224
        step = 75 # 这里必须保证产出的tensor 是偶数，这里是超大overlap的形式
        # window_size = 400
        # step = 50 # 这里必须保证产出的tensor 是偶数，这里是超大overlap的形式
        n_channel = 22

        def windows(data, size, step):
            start = 0
            while (start + size) < data.shape[0]:
                yield int(start), int(start + size)
                start += step

        def segment_signal_without_transition(data, window_size, step):
            segments = []
            for (start, end) in windows(data, window_size, step):
                if len(data[start:end]) == window_size:
                    segments = segments + [data[start:end]]
            return np.array(segments)

        def segment_dataset(X, window_size, step):
            win_x = []
            for i in range(X.shape[0]):
                win_x = win_x + [segment_signal_without_transition(X[i], window_size, step)]
            win_x = np.array(win_x)
            return win_x

        train_raw_x = np.transpose(train_X, [0, 2, 1])
        test_raw_x = np.transpose(test_X, [0, 2, 1])

        train_win_x = segment_dataset(train_raw_x, window_size, step)
        test_win_x = segment_dataset(test_raw_x, window_size, step)
        train_win_y = train_y
        test_win_y = test_y

        expand_factor = train_win_x.shape[1]

        train_x = np.reshape(train_win_x, (-1, train_win_x.shape[2], train_win_x.shape[3]))
        test_x = np.reshape(test_win_x, (-1, test_win_x.shape[2], test_win_x.shape[3]))
        train_y = np.repeat(train_y, expand_factor)
        test_y = np.repeat(test_y, expand_factor)

        train_x = np.reshape(train_x, [train_x.shape[0], 1, train_x.shape[1], train_x.shape[2]]).astype('float32')
        train_y = np.reshape(train_y, [train_y.shape[0]]).astype('float32')

        test_x = np.reshape(test_x, [test_x.shape[0], 1, test_x.shape[1], test_x.shape[2]]).astype('float32')
        test_y = np.reshape(test_y, [test_y.shape[0]]).astype('float32')

        # test_x = test_x[2000:, :, :, :]
        # test_y = test_y[2000:]

        # val_x = test_x[:2000, :, :, :]
        # val_y = test_y[:2000]

        train_win_x = train_win_x.astype('float32')
        
        ratio = 0.5
        idx = list(range(len(test_win_y)))
        np.random.shuffle(idx)
        test_win_x = test_win_x[idx]
        test_win_y = test_win_y[idx]

        val_win_x = test_win_x[:int(len(test_win_x)*0.5), :, :, :].astype('float32')
        val_win_y = test_win_y[:int(len(test_win_x)*0.5)]

        real_test_win_x = test_win_x[int(len(test_win_x)*0.5):, :, :, :].astype('float32')
        real_test_win_y = test_win_y[int(len(test_win_x)*0.5):]

        self.X_val = val_win_x
        self.y_val = val_win_y

        # print("The shape of sample x0 is {}".format(test_win_x.shape))

        if setname == 'train':
            self.data = train_win_x
            self.label = train_win_y
        elif setname == 'val':
            self.data = val_win_x
            self.label = val_win_y
        elif setname == 'test':
            self.data = real_test_win_x
            self.label = real_test_win_y

        self.num_class = 4
        # print("the final output shape is")
        # print(train_win_x.shape)
        # print(train_win_y.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        return data, label


class DatasetLoader_BCI_IV_mix_subjects(torch.utils.data.Dataset):

    def __init__(self, setname, datafolder, train_aug=False):
        self.wave_max = 30
        self.max_sampledepth = 255 # closer to image, not int type
        test_X_list = []
        train_X_list = []
        test_y_list = []
        train_y_list = []
        self.converter = wave_spectron.Spectro()
        for i in range(9):
            subject_id = i + 1
            if datafolder is None:
                data_folder = '../data'
            else:
                data_folder = datafolder

            data = sio.loadmat(data_folder + "/single_sep/single_subject_data_" + str(subject_id) + ".mat")
            test_X = data["test_x"][:, :, 750:1500]  # [trials, channels, time length]
            train_X = data["train_x"][:, :, 750:1500]

            test_y = np.ones((test_X.shape[0],)) * i
            train_y = np.ones((train_X.shape[0],)) * i
            # print(test_y.shape)
            # print(train_y.shape)
            test_X_list.append(test_X)
            train_X_list.append(train_X)
            test_y_list.append(test_y)
            train_y_list.append(train_y)

        test_X = np.vstack(test_X_list)
        train_y = np.concatenate(train_y_list, axis=0)
        train_X = np.vstack(train_X_list)
        test_y = np.concatenate(test_y_list, axis=0)
        print(test_X.shape)
        print(test_y.shape)
        print(train_X.shape)
        print(train_y.shape)

        train_x = train_X.astype('float32')
        test_x = test_X.astype('float32')
        
        train_x = sampling_func.convert_waveform(train_x, self.wave_max, self.max_sampledepth)
        test_x = sampling_func.convert_waveform(test_x, self.wave_max, self.max_sampledepth)

        # train_x = np.reshape(train_x, [train_x.shape[0], train_x.shape[1], train_x.shape[2]]).astype('float32')
        train_y = np.reshape(train_y, [train_y.shape[0]]).astype('float32')

        # test_x = np.reshape(test_x, [test_x.shape[0], 1, test_x.shape[1], test_x.shape[2]]).astype('float32')
        test_y = np.reshape(test_y, [test_y.shape[0]]).astype('float32')

        # self.converter.load_wave(raw_wave=train_x)
        # train_x = self.converter.wave_to_latent_multitr()
        # print("The shape of sample x0 is {}".format(test_x[0][1]))
        # self.converter.load_wave(raw_wave=test_x)
        # test_x = self.converter.wave_to_latent_multitr()
        
        ratio = 0.5
        idx = list(range(len(test_y)))
        np.random.shuffle(idx)
        test_win_x = test_x[idx]
        test_win_y = test_y[idx]

        idx = list(range(len(train_y)))
        np.random.shuffle(idx)
        train_x = train_x[idx]
        train_y = train_y[idx]

        val_win_x = test_win_x[:int(len(test_win_x)*0.5), :, :].astype('float32')
        val_win_y = test_win_y[:int(len(test_win_x)*0.5)]

        real_test_win_x = test_win_x[int(len(test_win_x)*0.5):, :, :].astype('float32')
        real_test_win_y = test_win_y[int(len(test_win_x)*0.5):]

        if setname == 'train':
            self.data = train_x
            # print("the shape of train data is {}".format(train_x.shape))
            self.label = train_y
        elif setname == 'val':
            self.data = val_win_x
            self.label = val_win_y
        elif setname == 'test':
            self.data = real_test_win_x
            self.label = real_test_win_y

        self.num_class = 9

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ## comment out F.normalize if you want unconstrained diffusion (without normalize to 1)
        # data, label = F.normalize(torch.Tensor(self.data[i]), p=2, dim=2), self.label[i]
        data, label = torch.Tensor(self.data[i]), self.label[i]
        return data, label


def datasetLoader_BCI_IV_signle():
    """
    Create BCI IV dataset
    """
    
    return DatasetLoader_BCI_IV_signle('train', datafolder='/home/yiqduan/Data/bci/EEG_MI_DARTS/Mudus_BCI/data/bci_iv/', subject_id=3)


def datasetLoader_BCI_IV_mix_subjects():
    """
    Create BCI IV dataset
    """
    return DatasetLoader_BCI_IV_mix_subjects('train', datafolder='/home/yiqduan/Data/bci/EEG_MI_DARTS/Mudus_BCI/data/bci_iv/')


def datasetLoader_BCI_IV_mix_subjects_test():
    """
    Create BCI IV dataset
    """
    return DatasetLoader_BCI_IV_mix_subjects('test', datafolder='/home/yiqduan/Data/bci/EEG_MI_DARTS/Mudus_BCI/data/bci_iv/')


if __name__ == '__main__':
    dataset = datasetLoader_BCI_IV_signle()
    # print(dataset[0][0].shape)
    # print(dataset[0][1])

    dataset = datasetLoader_BCI_IV_mix_subjects()
    print(dataset[0][0].shape)
    # print(dataset[0][0][1])

    """
    dataset.converter.plot_spectrogram(-dataset[0][0][0], save_fig='./test.png')

    pos_count = np.count_nonzero(np.sign(dataset[0][0][1]) == 1)
    neg_count = np.count_nonzero(np.sign(dataset[0][0][1]) == -1)

    print("Positive count:", pos_count)
    print("Negative count:", neg_count)
    
    print(dataset.data.shape)
    waves = dataset.converter.single_latent_to_wave(np.array(dataset[0][0]))

    
    todo: figure out why we need to use -dataset[0][0] to get the correct waveform
    power_to_db 函数将功率谱转换为分贝 (dB) 谱。由于功率谱的单位是平方的振幅，因此其值可以是任何非负数。
    而分贝的定义是基于对数的，所以分贝谱的值可以是正数或负数，其中 0 dB 表示参考功率或振幅，而负数则表示相对于参考功率或振幅的衰减。
    因此，对于某些音频信号，特别是低频信号，power_to_db 可能会返回负值。
    这是因为低频信号具有较高的振幅，但其功率却相对较低，所以需要使用负值来表示分贝谱的相对幅度。
    

    # waves = sampling_func.invert_waveform(waves, dataset.wave_max, dataset.max_sampledepth)
    # print(waves.shape)
    waves = sampling_func.invert_waveform(waves, 100, 255)
    print(waves[0])
    """