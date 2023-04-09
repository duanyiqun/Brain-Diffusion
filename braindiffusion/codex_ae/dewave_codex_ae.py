from __future__ import print_function
import abc

import numpy as np
import logging
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

# from nearest_embed import NearestEmbed, NearestEmbedEMA
from braindiffusion.codex_ae.nearest_embed import NearestEmbed, NearestEmbedEMA

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, Wav2Vec2Model
import transformers.utils as hf_utils
from transformers import Wav2Vec2Config, Wav2Vec2Model


import math
import numpy as np

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False
    print("No module 'xformers'. Proceeding without it.")


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

##################### facilities of encoder #################
def get_model_path(model_name):
    model_path = '{}/models--{}--{}/snapshots/{}'.format(hf_utils.PYTORCH_PRETRAINED_BERT_CACHE, model_name.split('/')[0], model_name.split('/')[1], model_name.split('/')[2])
    return model_path

def make_attn(in_channels, attn_type="vanilla", attn_kwargs=None):
    assert attn_type in ["vanilla", "vanilla-xformers", "memory-efficient-cross-attn", "linear", "none"], f'attn_type {attn_type} unknown'
    if XFORMERS_IS_AVAILBLE and attn_type == "vanilla":
        attn_type = "vanilla-xformers"
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        assert attn_kwargs is None
        return AttnBlock1d(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        raise NotImplementedError()

def get_act(act_name):
    if act_name=='Swish':
        return Swish
    elif act_name == 'None':
        return nn.Identity
    else:
        return getattr(nn, act_name)

class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""
    def forward(self, x):
        return x * torch.sigmoid(x)

def Normalize(in_channels, num_groups=5):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

class AttnBlock1d(nn.Module):
    """
    Keep conv2d is because the wave2vec output have group dimension on time wise, 
    Since we use kernel size 1, so it could be still regareded as pure attention between channels
    """

    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv1d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv1d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,w = q.shape
        q = q.reshape(b,c,w)
        q = q.permute(0,2,1)   # b,w,c
        k = k.reshape(b,c, w)   # b,c,w
        w_ = torch.bmm(q,k)     # b,w,w    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,w)
        w_ = w_.permute(0,2,1)   # b,w,w (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,w)

        h_ = self.proj_out(h_)

        return x+h_
    

class BrainCodex_Encoder_Dewave(nn.Module):
    """
    Encoder using wave2vec structure:
    Input: bs x channels x samplepoints
    Output: bs x z, muliple z shapes. 
    """

    def __init__(self, checkpoint='facebook/wav2vec2-base-960h', in_feature=[105, 2000], init=False, codex_number=1024, act ='Swish', attn_type='vanilla', give_pre_end=False, latent_codex_size=512): 
        super(BrainCodex_Encoder_Dewave,self).__init__()
         
        if len(checkpoint.split('/'))>2:
            self.wav2vec = Wav2Vec2Model.from_pretrained(get_model_path(checkpoint), local_files_only=True)
        else:
            self.wav2vec = Wav2Vec2Model.from_pretrained(checkpoint)#"facebook/wav2vec2-base-960h", 发现这个会自动更新, 很烦
        self.channels =in_feature[0]
        self.samples = in_feature[1]

        if init:
            print('####### wav2Vec feature extractor weight init!')
            self.wav2vec.init_weights()
        self.act  = get_act(act)()
        out_groups = self.CalculateOutSize(in_feature)
        self.channel_fusion = torch.nn.Conv2d(self.channels,
                                 1,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        # self.channel_fusion = nn.Linear(out_groups, 1) # put 2 groups to 1 
        # self.projector = nn.Linear(channels*512, num_hiddens)# 添加2个线性层

        self.del_unused_layers()

    def del_unused_layers(self):
        self.wav2vec.feature_projection = nn.Sequential()
        self.wav2vec.encoder = nn.Sequential()
        print('delete unsed layers')

    def resemble(self):
        # remix of features abbandon
        pass 

    def get_pFeatures(self):
        pass

    def CalculateOutSize(self, input_sample_shape):
        data = torch.rand(input_sample_shape) # image [bs, c=eeg_ch, w=sample 也就是输入的eeg长度]
        self.eval()
        feature  = self.wav2vec.feature_extractor(data) # feature torch.Size([66, 512, 2])
        out = feature.shape
        return out[-1]
    
    def forward(self, x):
        x_= x.reshape(x.size(0)*x.size(1),x.size(2))
        # print(x_.shape)
        feature = self.wav2vec.feature_extractor(x_) # feature torch.Size([22, 512, 2])
        # print(feature.shape)
        feature = self.channel_fusion(feature.reshape(x.size(0),x.size(1),feature.size(1), feature.size(2))) # feature torch.Size([1, 22, 512, 2])
        # print(feature.shape)
        # feature_fuse_group =self.act(self.channel_fusion(feature)) # torch.Size([1, 22, 512, 1])
        # print('feature_fuse_group',feature_fuse_group.size())
        extract_features=torch.squeeze(feature_fuse_group, -1) # torch.Size([66, 512])
        # print(feature_fuse_group.size())
        # print('extract_features',extract_features.size())
        # extract_features = extract_features.reshape(x.size(0),x.size(1)*extract_features.size(-1)) # [bs, eeg_channel*embed_channel]
        # print('extract_features',extract_features.size())
        # proj= self.act(self.projector(extract_features))
        # print('proj',proj.size())
        extract_features=torch.squeeze(feature_fuse_group, -1) # torch.Size([66, 512])
        return extract_features
    


class BrainCodex_Encoder_Dewave_c6_512(nn.Module):
    """
    Encoder using wave2vec structure:
    Input: bs x channels x samplepoints
    Output: bs x z, muliple z shapes. 
    """

    def __init__(self, checkpoint='facebook/wav2vec2-base-960h', in_feature=[105, 2000], init=False, codex_number=1024, act ='Swish', attn_type='vanilla', give_pre_end=False, latent_codex_size=512): 
        super(BrainCodex_Encoder_Dewave_c6_512,self).__init__()
         
        configuration = Wav2Vec2Config(conv_stride=(3, 2, 2, 2, 2, 2),conv_kernel = (10, 3, 3, 3, 3, 2), conv_dim = (512, 512, 512, 512, 512, 512))
        self.wav2vec = Wav2Vec2Model(configuration)
        self.channels =in_feature[0]
        self.samples = in_feature[1]

        if init:
            print('####### wav2Vec feature extractor weight init!')
            self.wav2vec.init_weights()
        # self.act  = get_act(act)()
        out_groups = self.CalculateOutSize(in_feature)
        self.channel_fusion = torch.nn.Conv2d(self.channels,
                                 1,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        # self.channel_fusion = nn.Linear(out_groups, 1) # put 2 groups to 1 
        # self.projector = nn.Linear(channels*512, num_hiddens)# 添加2个线性层

        self.del_unused_layers()

    def del_unused_layers(self):
        self.wav2vec.feature_projection = nn.Sequential()
        self.wav2vec.encoder = nn.Sequential()
        print('delete unsed layers')

    def resemble(self):
        # remix of features abbandon
        pass 

    def get_pFeatures(self):
        pass

    def CalculateOutSize(self, input_sample_shape):
        data = torch.rand(input_sample_shape) # image [bs, c=eeg_ch, w=sample 也就是输入的eeg长度]
        self.eval()
        feature  = self.wav2vec.feature_extractor(data) # feature torch.Size([66, 512, 2])
        out = feature.shape
        return out[-1]
    
    def forward(self, x):
        x_= x.reshape(x.size(0)*x.size(1),x.size(2))
        # print(x_.shape)
        feature = self.wav2vec.feature_extractor(x_) # torch.Size([105, 512, 56])
        # print(feature.shape)
        feature = self.channel_fusion(feature.reshape(x.size(0),x.size(1),feature.size(1), feature.size(2))) # torch.Size([1, 1, 512, 56])
        # print(feature.shape)

        extract_features=torch.squeeze(feature, 1) # torch.Size([66, 512])
        # print(feature_fuse_group.size())
        # print('extract_features',extract_features.size())
        # extract_features = extract_features.reshape(x.size(0),x.size(1)*extract_features.size(-1)) # [bs, eeg_channel*embed_channel]
        # print('extract_features',extract_features.size())
        # proj= self.act(self.projector(extract_features))
        # print('proj',proj.size())
        # extract_features=torch.squeeze(feature_fuse_group, -1) # torch.Size([66, 512])
        return extract_features
    


class BrainCodex_Encoder_Dewave_shallow(nn.Module):
    """
    Encoder using wave2vec structure:
    Input: bs x channels x samplepoints
    Output: bs x z, muliple z shapes. 
    """

    def __init__(self, checkpoint='facebook/wav2vec2-base-960h', in_feature=[105, 2000], init=False, codex_number=1024, act ='Swish', attn_type='vanilla', give_pre_end=False, latent_codex_size=512): 
        super(BrainCodex_Encoder_Dewave_shallow,self).__init__()
         
        configuration = Wav2Vec2Config(conv_stride=(5, 2, 2, 2, 2, 2),conv_kernel = (10, 3, 3, 3, 3, 2), conv_dim = (512, 512, 512, 512, 512, 512))
        self.wav2vec = Wav2Vec2Model(configuration)
        self.channels =in_feature[0]
        self.samples = in_feature[1]

        if init:
            print('####### wav2Vec feature extractor weight init!')
            self.wav2vec.init_weights()
        # self.act  = get_act(act)()
        out_groups = self.CalculateOutSize(in_feature)
        self.channel_fusion = torch.nn.Conv2d(self.channels,
                                 1,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        # self.channel_fusion = nn.Linear(out_groups, 1) # put 2 groups to 1 
        # self.projector = nn.Linear(channels*512, num_hiddens)# 添加2个线性层

        self.del_unused_layers()

    def del_unused_layers(self):
        self.wav2vec.feature_projection = nn.Sequential()
        self.wav2vec.encoder = nn.Sequential()
        print('delete unsed layers')

    def resemble(self):
        # remix of features abbandon
        pass 

    def get_pFeatures(self):
        pass

    def CalculateOutSize(self, input_sample_shape):
        data = torch.rand(input_sample_shape) # image [bs, c=eeg_ch, w=sample 也就是输入的eeg长度]
        self.eval()
        feature  = self.wav2vec.feature_extractor(data) # feature torch.Size([66, 512, 2])
        out = feature.shape
        return out[-1]
    
    def forward(self, x):
        x_= x.reshape(x.size(0)*x.size(1),x.size(2))
        # print(x_.shape)
        feature = self.wav2vec.feature_extractor(x_) # torch.Size([105, 512, 56])
        # print(feature.shape)
        feature = self.channel_fusion(feature.reshape(x.size(0),x.size(1),feature.size(1), feature.size(2))) # torch.Size([1, 1, 512, 56])
        # print(feature.shape)

        extract_features=torch.squeeze(feature, 1) # torch.Size([66, 512])
        # print(feature_fuse_group.size())
        # print('extract_features',extract_features.size())
        # extract_features = extract_features.reshape(x.size(0),x.size(1)*extract_features.size(-1)) # [bs, eeg_channel*embed_channel]
        # print('extract_features',extract_features.size())
        # proj= self.act(self.projector(extract_features))
        # print('proj',proj.size())
        # extract_features=torch.squeeze(feature_fuse_group, -1) # torch.Size([66, 512])
        return extract_features


class BrainCodex_Decoder_Freq(nn.Module):
    """
    Decoder match to encoder
    """
    def __init__(self, in_feature = 840, latent_codex_size = 512, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048, codex_number=128):
        super(BrainCodex_Decoder_Freq, self).__init__()

        # additional transformer encoder, following BART paper about 
        self.positional_embedding = PositionalEncoding(latent_codex_size)
        self.translayer = nn.TransformerEncoderLayer(d_model=latent_codex_size, nhead=additional_encoder_nhead,  dim_feedforward = additional_encoder_dim_feedforward, batch_first=True)
        self.decoder = nn.TransformerEncoder(self.translayer, num_layers=6)
        
        # print('[INFO]adding positional embedding')
        # self.positional_embedding = PositionalEncoding(in_feature)

        self.fc1 = nn.Linear(latent_codex_size, in_feature)

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted):
        """input_embeddings_batch: batch_size*Seq_len*840"""
        """input_mask: 1 is not masked, 0 is masked"""
        """input_masks_invert: 1 is masked, 0 is not masked"""
        
        input_embeddings_batch = self.positional_embedding(input_embeddings_batch) 

        # use src_key_padding_masks
        decoded_embedding = self.decoder(input_embeddings_batch, src_key_padding_mask = input_masks_invert) 
        freq_feature = F.relu(self.fc1(decoded_embedding))          
        
        return freq_feature
    


class VQ_Codex(nn.Module):
    """Vector Quantized AutoEncoder for mnist"""

    def __init__(self, in_feature = [105, 5500], latent_codex_size = 512, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048, 
                codex_number=1014, vq_coef=0.2, comit_coef=0.4, **kwargs):
        super(VQ_Codex, self).__init__()

        self.brain_codex_encoder = BrainCodex_Encoder_Dewave_c6_512(in_feature = in_feature, latent_codex_size = latent_codex_size, codex_number=codex_number)
        self.brain_codex_decoder = None
        self.latent_codex_size = latent_codex_size

        self.emb = NearestEmbed(codex_number, self.latent_codex_size)
        print("using the codex with shape {}".format(self.emb.weight.shape))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.vq_coef = vq_coef
        self.comit_coef = comit_coef
        self.hidden = latent_codex_size
        self.ce_loss = 0
        self.vq_loss = 0
        self.commit_loss = 0
        

    def encode(self, input_embeddings_batch):
       
        return self.brain_codex_encoder(input_embeddings_batch)

    def decode(self, latent_z, input_masks_batch, input_masks_invert, target_ids_batch_converted):
        return None

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted):
        z_e = self.encode(input_embeddings_batch)
        # print("z_e {}".format(z_e.shape))
        z_q, _ = self.emb(z_e, weight_sg=True)
        # print("z_q {}".format(z_q.shape))
        emb, _ = self.emb(z_e.detach())
        return z_q, z_e, emb
    
    def forward_codex(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted):
        z_e = self.encode(input_embeddings_batch)
        # print("z_e {}".format(z_e.shape))
        z_q, _ = self.emb(z_e, weight_sg=True)
        # print("z_q {}".format(z_q.shape))
        emb, _ = self.emb(z_e)
        return z_q, z_e, emb

    def sample(self, size, lenght=56):
        sample = torch.randn(size, self.latent_codex_size,
                             int(lenght))
        if self.cuda():
            sample = sample.cuda()
        emb, _ = self.emb(sample)
        sample = self.decode(emb(sample).view(-1, self.hidden), None, None, None).cpu()
        return sample

    def loss_function(self, dec_loss, z_e, emb):
        # self.ce_loss = F.binary_cross_entropy(recon_x, x)
        # self.ce_loss = F.binary_cross_entropy_with_logits(recon_x, x)
        self.vq_loss = F.mse_loss(emb, z_e.detach())
        self.commit_loss = F.mse_loss(z_e, emb.detach())

        return self.vq_coef*self.vq_loss + self.comit_coef*self.commit_loss

    def latest_losses(self):
        return {'cross_entropy': 0, 'vq': self.vq_loss, 'commitment': self.commit_loss}


""" Decoder for EEG Codex Book """
class BrainTranslator(nn.Module):
    def __init__(self, pretrained_layers, in_feature = 840, decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048):
        super(BrainTranslator, self).__init__()
        
        self.pretrained = pretrained_layers
        # additional transformer encoder, following BART paper about 
        self.additional_encoder_layer = nn.TransformerEncoderLayer(d_model=in_feature, nhead=additional_encoder_nhead,  dim_feedforward = additional_encoder_dim_feedforward, batch_first=True)
        self.additional_encoder = nn.TransformerEncoder(self.additional_encoder_layer, num_layers=6)
        
        # print('[INFO]adding positional embedding')
        # self.positional_embedding = PositionalEncoding(in_feature)

        self.fc1 = nn.Linear(in_feature, decoder_embedding_size)

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted):
        """input_embeddings_batch: batch_size*Seq_len*840"""
        """input_mask: 1 is not masked, 0 is masked"""
        """input_masks_invert: 1 is masked, 0 is not masked"""
        
        # input_embeddings_batch = self.positional_embedding(input_embeddings_batch) 

        # use src_key_padding_masks
        encoded_embedding = self.additional_encoder(input_embeddings_batch, src_key_padding_mask = input_masks_invert) 
        
        # encoded_embedding = self.additional_encoder(input_embeddings_batch) 
        encoded_embedding = F.relu(self.fc1(encoded_embedding))
        out = self.pretrained(inputs_embeds = encoded_embedding, attention_mask = input_masks_batch, return_dict = True, labels = target_ids_batch_converted)                    
        
        return out




if __name__ == '__main__':
    test_sample = torch.rand(8, 105, 5500)
    test_mask = torch.ones(8,56).bool()
    test_mask_invert = torch.zeros(8,56).bool()
    test_target_ids = torch.ones(8,56).long()
    wave2vec_enc = BrainCodex_Encoder_Dewave()
    wave2vec_enc = BrainCodex_Encoder_Dewave_c6_512()
    test_codex = wave2vec_enc(test_sample)
    print(test_codex.shape)

    vq_vae_test = VQ_Codex()
    rec_feature, z_e, emb = vq_vae_test(test_sample,  test_mask, test_mask_invert, None)
    # print("rec_feature {}".format(rec_feature.shape))
    print("z_e {}".format(z_e.shape))
    print("emb {}".format(emb.shape))
    # print(test_sample)
    loss = vq_vae_test.loss_function(0, z_e, emb)
    loss.backward()
    pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    brain_codex_translator = BrainTranslator(pretrained, in_feature = 512, decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)
    z_q, z_e, emb = vq_vae_test(test_sample,  test_mask, test_mask_invert, None)
    emb = emb.permute(0,2,1)
    print(emb.shape)
    out =  brain_codex_translator(emb, test_mask, test_mask_invert, test_target_ids)

    