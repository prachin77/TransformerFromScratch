import torch
import torch.nn as nn
import torch.functional as F
import math 

class InputEmbeddings(nn.Module):
    def __init__(self, d_model : int , vocab_size : int): 
        # d_model = dimension of the vector , here 512 
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size , d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    # - vector of size 512
    # - computed only once
    # - reused for every sentence during training and inference     


    def __init__(self, d_model : int , seq_len : int , dropout : float) -> None:
        # seq_len = max length of the sentence
        # dropout = randomly freezes few networks/layers to induce uncertainity and reduce overfitting
        super().__init__()
        self.d_model = d_model 
        self.seq_length = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len , d_model)
        pe = torch.zeros(seq_len , d_model)

        # create a vector of shape (seq_len , 1) -> represents position of word inside the sentence
        position = torch.arange(0 , seq_len , dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0 , d_model , 2).float() * (-math.log(10000.0) / d_model))
        # every word gets sin but for even dims 
        # every word gets cosine but for odd dims 
        pe[: , 0::2] = torch.sin(position * div_term)
        pe[: , 1::2] = torch.cos(position * div_term)

        # add new batch dim because we have batch of sentences 
        pe = pe.unsqueeze(0) # (1 , seq_len , d_model)

        # register tensor in the buffer of the module
        # it gets saved inside the module not as a saved parameter  
        self.register_buffer("pe" , pe)

    def forward(self , x):
        # add positional encoding to every word of the sentence
        # we also have to tell the model that we don't wanna learn this 
        # PEs every time  because they're fixed 
        x = x + (self.pe[: , :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


