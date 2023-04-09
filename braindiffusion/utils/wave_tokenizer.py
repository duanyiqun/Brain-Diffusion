# This is brain waves to discrete tokens
# Migrated from https://github.com/huggingface/diffusers Under liscence
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
from braindiffusion.codex_ae.dewave_codex_ae import *

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



if __name__ == '__main__':
    test_sample = torch.rand(8, 105, 15000)
    test_mask = torch.ones(8,56).bool()
    test_mask_invert = torch.zeros(8,56).bool()
    test_target_ids = torch.ones(8,56).long()
    # wave2vec_enc = BrainCodex_Encoder_Dewave()
    wave2vec_enc = BrainCodex_Encoder_Dewave_c6_512()
    test_codex = wave2vec_enc(test_sample)
    print(test_codex.shape)

    vq_vae_test = VQ_Codex(latent_codex_size = 512, codex_number=2048)
    rec_feature, z_e, emb = vq_vae_test(test_sample,  test_mask, test_mask_invert, None)
    # print("rec_feature {}".format(rec_feature.shape))
    print("z_e {}".format(z_e.shape))
    print("emb {}".format(emb.shape))
    # print(test_sample)
    loss = vq_vae_test.loss_function(0, z_e, emb)
    loss.backward()
    # pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    # brain_codex_translator = BrainTranslator(pretrained, in_feature = 512, decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)
    z_q, z_e, emb = vq_vae_test(test_sample,  test_mask, test_mask_invert, None)
    emb = emb.permute(0,2,1)
    print(emb.shape)
    # out =  brain_codex_translator(emb, test_mask, test_mask_invert, test_target_ids)

    