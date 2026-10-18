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

        # we can also save the special tokens like SOS , EOS , PAD 
        # that we'll use to create tensors for the model . 

        # converting these 3 tokens into input id i.e a number

        self.sos_token = torch.Tensor([tokenizer_src.token_to_id(["[SOS]"])] , dtype = torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id(["[EOS]"])] , dtype = torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id(["[PAD]"])] , dtype = torch.int64)

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
