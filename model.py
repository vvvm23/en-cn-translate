import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.nn.Module that implements positional encoding
# injects positional data into the input as Transformers do not provide any sense of position by default
# blantantly taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, nb_in, dropout=0.1, max_length=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_length, nb_in)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, nb_in, 2).float() * (-torch.log(torch.tensor(10000.0)) / nb_in))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# torch.nn.Module to contain the full model as one unit
# based on the model architecture defined in Attention is All You Need (Vaswani et. al.)
# https://arxiv.org/abs/1706.03762
class LangTransformer(nn.Module):
    def __init__(self, src_dim, tgt_dim, emd_dim, hidden_dim, device, dropout=0.1):
        super().__init__()
        self.device = device # store reference to device for creating masks later
        self.mask_tgt = None

        # Embedding and PositionalEncoding layers for the two model inputs
        self.emb_src = nn.Embedding(src_dim, emd_dim)
        self.emb_tgt = nn.Embedding(tgt_dim, emd_dim)
        self.pos_src = PositionalEncoding(emd_dim)
        self.pos_tgt = PositionalEncoding(emd_dim)

        # Main transformer module 
        self.transformer = nn.Transformer(emd_dim, 8, 6, 6, hidden_dim, dropout=dropout)

        # Fully connected output layer that is applied identically across all time steps
        self.linear = nn.Linear(emd_dim, tgt_dim)

    def forward(self, src, tgt):
        # if mask has not been initialised yet, create it
        # TODO: can this be moved to __init__ ?
        if self.mask_tgt == None: 
            self.mask_tgt = self.transformer.generate_square_subsequent_mask(tgt.shape[1]).to(self.device)

        # padding mask for src input
        mask_pad_src = torch.zeros_like(src).type(torch.BoolTensor).to(self.device)
        mask_pad_src[src == 0] = True

        # padding mask for tgt input
        mask_pad_tgt = torch.zeros_like(tgt).type(torch.BoolTensor).to(self.device)
        mask_pad_tgt[tgt == 0] = True

        # pass both inputs through embedding layers
        src = self.emb_src(src)
        tgt = self.emb_tgt(tgt)

        # pass both embeddings through separate positional encoders
        src = self.pos_src(src)
        tgt = self.pos_tgt(tgt)
    
        # swap batch and time dimensions as per nn.Transformer documentation
        src, tgt = src.permute(1, 0, 2), tgt.permute(1, 0, 2)
        out = self.transformer(src, tgt, tgt_mask = self.mask_tgt, src_key_padding_mask=mask_pad_src, tgt_key_padding_mask=mask_pad_tgt)

        # swap back batch and time after passing inputs through Transformer
        out = out.permute(1, 0, 2)

        # Apply final fully connected layer and log softmax function
        out = self.linear(out)
        out = F.log_softmax(out, dim=-1)

        return out
