from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader

class BilingualDataset(Dataset):
    def __init__(self , ds , tokenizer_src , tokenizer_tgt , src_lang , tgt_lang , seq_len) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        # we can also save the special tokens like SOS , EOS , PAD 
        # that we'll use to create tensors for the model . 

        # converting these 3 tokens into input id i.e a number

        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        # gives the length of the ds itself
        return len(self.ds)

    def __getitem__(self, index : Any) -> Any:
        src_target_pair = self.ds[index] # extracts the en-hi subset from the dataset
        src_text = src_target_pair["translation"][self.src_lang] # english
        tgt_text = src_target_pair["translation"][self.tgt_lang] # hindi

        # tokenize each word and then map each word to its corresponding
        # number in vocab.
        # ids = gives input id so the nos corresponding to each word 
        # will be given as an array
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # we'll use padding token to fill the spaces until it reaches the sentence length
        # we need this because for same sentence both languages can have different
        # sentence length
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) -2 # -2 for SOS and EOS tokens on encoder side
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) -1 # -1 for EOS token on decoder side

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("sentence is too long")

        # create tensors from sentence for encoder and decoder input and for the label. 
        # label = sentence that we expect as the output of the decoder 
        
        # Add SOS and EOS to the source text 
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens , dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens , dtype=torch.int64)
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens , dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens , dtype=torch.int64)
            ]
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens , dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens , dtype = torch.int64)
            ]
        )

        # double check if we actually reach the seq_len 
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input" : encoder_input, # seq_len
            "decoder_input" : decoder_input, # seq_len
            # why this mask ?
            # because we're increase the size of encoder input by adding PAD tokens ,
            # but we don't want these tokens to get involved in attention mechanism  
            # "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).int(), # (1 , 1 , seq_len)
            "encoder_mask" : (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1 , 1 , seq_len)
            # special tokens for decoder which is casual mask . which means that each word can only look at 
            # previous words and ignore PAD tokens
            "decoder_mask" : (decoder_input != self.pad_token).unsqueeze(0).int() & casual_mask(decoder_input.size(0)), # (1 , seq_len) & (1 , seq_len , seq_len)
            "label" : label, # (seq_len)
            "src_text" : src_text,
            "tgt_text" : tgt_text
        }

def casual_mask(size):
    mask = torch.triu(torch.ones(1 , size , size), diagonal=1).type(torch.int64)
    return mask == 0



