import argparse
import os
import pickle
import torch
import torch.nn as nn
from datasets import load_dataset, load_from_disk
from tqdm.auto import tqdm
from transformers import BertTokenizerFast, BertModel, BertLMHeadModel
from braindiffusion.modeling.x_transformer import Encoder, TransformerWrapper 

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""
    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda",use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)#.to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)
    


def main(args):
    text_encoder = BertModel.from_pretrained("bert-base-uncased")
    # text_encoder = BertLMHeadModel.from_pretrained("bert-base-uncased")

    if args.dataset_name is not None:
        if os.path.exists(args.dataset_name):
            dataset = load_from_disk(args.dataset_name)["train"]
        else:
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
                use_auth_token=True if args.use_auth_token else None,
                split="train",
            )

    encodings = {}
    # for dataslice in tqdm(dataset.to_pandas()["audio_file"].unique()):
    for dataslice in tqdm(dataset):
        seq_len = dataslice["seq_len"]
        input_masks = dataslice["input_masks"]
        input_mask_invert = dataslice["input_mask_invert"]
        target_ids = dataslice["target_ids"]
        target_mask = dataslice["target_mask"]

        input_masks_t = torch.tensor(input_masks, dtype=torch.int)
        input_mask_invert_t = torch.tensor(input_mask_invert, dtype=torch.int)
        target_ids_t = torch.tensor(target_ids, dtype=torch.int)
        target_mask_t = torch.tensor(target_mask, dtype=torch.int)

        target_ids_batch = target_ids_t.unsqueeze(0)
        input_masks_batch = input_masks_t.unsqueeze(0)

        output = text_encoder(input_ids = target_ids_batch, attention_mask = input_masks_batch)    
        # print(output)
        # print(output.last_hidden_state.squeeze().shape)
        encodings[str(target_ids)] = output.last_hidden_state.squeeze()
    pickle.dump(encodings, open(args.output_file, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create pickled audio encodings for dataset of audio files.")
    parser.add_argument("--dataset_name", type=str, default="./dataset/zuco/spectro_dp")
    parser.add_argument("--output_file", type=str, default="dataset/zuco/spectro_dp/text_encodings_train.pt")
    parser.add_argument("--use_auth_token", type=bool, default=False)
    args = parser.parse_args()
    main(args)
