import numpy as np
import torch
from diffusers import ConfigMixin, ModelMixin
from torch import nn
from braindiffusion.utils import wave_spectron, sampling_func

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            bias=False,
            padding=1,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate):
        super(ConvBlock, self).__init__()
        self.sep_conv = SeparableConv2d(in_channels, out_channels, (3, 3))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01)
        self.max_pool = nn.MaxPool2d((2, 2))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.sep_conv(x)
        x = self.leaky_relu(x)
        x = self.batch_norm(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        return x


class DenseBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate):
        super(DenseBlock, self).__init__()
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(in_features, out_features)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.batch_norm = nn.BatchNorm1d(out_features, eps=0.001, momentum=0.01)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.flatten(x.permute(0, 2, 3, 1))
        x = self.dense(x)
        x = self.leaky_relu(x)
        x = self.batch_norm(x)
        x = self.dropout(x)
        return x


class AudioEncoder(ModelMixin, ConfigMixin):
    def __init__(self):
        super().__init__()
        self.converter = wave_spectron.Spectro(
            x_res=31,
            y_res=64,
            sample_rate=250,
            n_fft=100,
            hop_length=25,
            top_db=50,
            n_iter=32,
            eeg_channels=22,
        )
        self.conv_blocks = nn.ModuleList([ConvBlock(22, 32, 0.2), ConvBlock(32, 64, 0.3), ConvBlock(64, 128, 0.4)])
        self.dense_block = DenseBlock(4096, 1024, 0.5)
        self.embedding = nn.Linear(1024, 100)
        self.converter = wave_spectron.Spectro()
        
    def forward(self, x):
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        x = self.dense_block(x)
        x = self.embedding(x)
        return x

    @torch.no_grad()
    def encode(self, sample_wave):
        self.eval()
        self.converter.load_wave(raw_wave=sample_wave)
        latent = self.converter.wave_to_latent_multitr(0)
        return torch.from_numpy(latent).float()


if __name__ == "__main__":
    model = AudioEncoder()
    # print(model.encode(np.random.rand(1, 22, 800)))
    spectro = model.encode(np.random.rand(1, 22, 800))
    embedding = model.forward(spectro)
    print(embedding.shape)
    