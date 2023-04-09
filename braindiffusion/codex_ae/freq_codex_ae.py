from __future__ import print_function
import abc

import numpy as np
import logging
import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F

# from nearest_embed import NearestEmbed, NearestEmbedEMA
from codex_ae.nearest_embed import NearestEmbed, NearestEmbedEMA

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import math
import numpy as np


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


class BrainCodex_Encoder_Freq(nn.Module):
    """
    Simple encoder to encode descrete feature for EEG frequencies
    The straight forward idea is to let the encoding sequence equals to codex sequence
    We assume the codex with positional embedding. 
    """
    def __init__(self, in_feature = 840, latent_codex_size = 512, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048, codex_number=128):
        super(BrainCodex_Encoder_Freq, self).__init__()
        
        # additional transformer encoder, following BART paper about 
        self.positional_embedding = PositionalEncoding(in_feature)
        self.translayer = nn.TransformerEncoderLayer(d_model=in_feature, nhead=additional_encoder_nhead,  dim_feedforward = additional_encoder_dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.translayer, num_layers=6)
        
        # print('[INFO]adding positional embedding')
        # self.positional_embedding = PositionalEncoding(in_feature)

        self.fc1 = nn.Linear(in_feature, latent_codex_size)

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted):
        """input_embeddings_batch: batch_size*Seq_len*840"""
        """input_mask: 1 is not masked, 0 is masked"""
        """input_masks_invert: 1 is masked, 0 is not masked"""
        
        input_embeddings_batch = self.positional_embedding(input_embeddings_batch) 

        # use src_key_padding_masks
        encoded_embedding = self.encoder(input_embeddings_batch, src_key_padding_mask = input_masks_invert) 
        latent_codex = F.relu(self.fc1(encoded_embedding))          
        
        return latent_codex
    

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

    def __init__(self, in_feature = 840, latent_codex_size = 512, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048, 
                codex_number=1014, vq_coef=0.2, comit_coef=0.4, **kwargs):
        super(VQ_Codex, self).__init__()

        self.brain_codex_encoder = BrainCodex_Encoder_Freq(in_feature = in_feature, latent_codex_size = latent_codex_size, additional_encoder_nhead=additional_encoder_nhead, 
                                                   additional_encoder_dim_feedforward = additional_encoder_dim_feedforward, codex_number=codex_number)
        self.brain_codex_decoder = BrainCodex_Decoder_Freq(in_feature = in_feature, latent_codex_size = latent_codex_size, additional_encoder_nhead=additional_encoder_nhead,
                                                   additional_encoder_dim_feedforward = additional_encoder_dim_feedforward, codex_number=codex_number)
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

    def encode(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted):
       
        return self.brain_codex_encoder(input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted)

    def decode(self, latent_z, input_masks_batch, input_masks_invert, target_ids_batch_converted):
        return self.brain_codex_decoder(latent_z, input_masks_batch, input_masks_invert, target_ids_batch_converted)

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted):
        z_e = self.encode(input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted).permute(0,2,1)
        # print("z_e {}".format(z_e.shape))
        z_q, _ = self.emb(z_e, weight_sg=True)
        # print("z_q {}".format(z_q.shape))
        emb, _ = self.emb(z_e.detach())
        return self.decode(z_q.permute(0,2,1), input_masks_batch, input_masks_invert, target_ids_batch_converted), z_e, emb, z_q

    def forward_codex(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted):
        z_e = self.encode(input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted).permute(0,2,1)
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

    def loss_function(self, x, recon_x, z_e, emb):
        # self.ce_loss = F.binary_cross_entropy(recon_x, x)
        self.ce_loss = F.binary_cross_entropy_with_logits(recon_x, x)
        self.vq_loss = F.mse_loss(emb, z_e.detach())
        self.commit_loss = F.mse_loss(z_e, emb.detach())

        return self.ce_loss + self.vq_coef*self.vq_loss + self.comit_coef*self.commit_loss

    def latest_losses(self):
        return {'cross_entropy': self.ce_loss, 'vq': self.vq_loss, 'commitment': self.commit_loss}



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
    test_sample = torch.rand(32, 56, 840)
    test_mask = torch.ones(32,56).bool()
    test_mask_invert = torch.zeros(32,56).bool()
    test_target_ids = torch.ones(32,56).long()
    encoder = BrainCodex_Encoder_Freq()
    decoder = BrainCodex_Decoder_Freq()
    latent_codex = encoder(test_sample, test_mask, test_mask_invert, None)
    print(latent_codex.shape)
    freq_feature = decoder(latent_codex, test_mask, test_mask_invert, None)
    print(freq_feature.shape)
    vq_vae_test = VQ_Codex()
    rec_feature, z_e, emb = vq_vae_test(test_sample,  test_mask, test_mask_invert, None)
    print("rec_feature {}".format(rec_feature.shape))
    print("z_e {}".format(z_e.shape))
    print("emb {}".format(emb.shape))
    # print(test_sample)
    loss = vq_vae_test.loss_function(test_sample, rec_feature, z_e, emb)
    loss.backward()
    pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    brain_codex_translator = BrainTranslator(pretrained, in_feature = 512, decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)
    _, _, emb = vq_vae_test.forward_codex(test_sample,  test_mask, test_mask_invert, None)
    emb = emb.permute(0,2,1)
    out =  brain_codex_translator(emb, test_mask, test_mask_invert, test_target_ids)


    
    