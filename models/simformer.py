import math
import torch 
import torch.nn as nn


def mean_pooling(x, padding_masks):
    """
    input: batch_size, seq_len, hidden_dim 
    output: batch_size, hidden_dim 
    """
    x = x * padding_masks.unsqueeze(-1)  # padding的地方置零
    x = torch.sum(x, dim=1)/torch.sum(padding_masks, dim=1).unsqueeze(-1)  #  mean pooling excluding the padding part. 
    return x  # 对输出的向量padding的地方置零后求均值

class SIMformer(nn.Module):
    def __init__(self, feat_dim, hidden_dim, dimfeedforward=256, n_heads=16, num_layers=1, 
                 pos_encoding="learnable"):
        """ SIMformer, a 1-layer vanilla siamese transformer, Encoder Only
        Args:
            - feat_dim (int): Input feature dimension, for trajectory data, feat_dim=2, which is longitude and latitude
            - hidden_dim (int): Dimension after linear mapping of longitude and latitude, which is the input dimension of transformer
            - dimfeedforward (int): Dimension of feedforward network in transformer
            - n_heads (int): Number of heads in multi-head attention in transformer
            - num_layers (int): Number of layers in transformer
            - pos_encoding (str): Position encoding method, can be learnable or sin_cos
        """
        super(SIMformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = 200 # set according to the maximum trajectory length in the dataset, and used for positional encoding
        self.project_inp = nn.Linear(feat_dim, hidden_dim) # point dimensionality enhancement
        self.pos_enc = get_pos_encoder(pos_encoding)(hidden_dim, max_len=self.max_seq_len, batch_first=True)  # self.pos_enc.pe的shape: [1, self.max_seq_len, hidden_dim]
        encoder_layer = nn.TransformerEncoderLayer(d_model = hidden_dim, nhead = n_heads, dim_feedforward= dimfeedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)  # num_layers: block的数量
        # This is to ensure that the output is a positive number, 
        # controlling the range of similarity between 0 and 1, because sometimes cosine similarity is used.
        self.act = nn.ReLU()
    
    def forward(self, x, padding_masks):  # x的shape: [batch_size*sampling_num, max_seq_len, 2]
        input = self.project_inp(x)
        input = self.pos_enc(input)  # add positional encoding 后接一层dropout
        output = self.transformer_encoder(input, src_key_padding_mask=~padding_masks.bool())  # [batch_size*sampling_num, max_seq_len, hidden_dim]
        output = mean_pooling(output, padding_masks)  # 对输出的向量padding的地方置零后求均值 [batch_size*sampling_num, hidden_dim]
        output = self.act(output)  # 将x映射为hidden vector, [batch_size*sampling_num, hidden_dim]
        return output


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0, batch_first=False):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first 
        pe = torch.zeros(max_len, d_model)  # positional encoding, (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model/2) 
        pe[:, 0::2] = torch.sin(position * div_term) # (max_len, d_model/2)
        pe[:, 1::2] = torch.cos(position * div_term) # (max_len, d_model/2)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1) # (1, max_len, d_model) -> (max_len, 1, d_model)
        
        if self.batch_first:
            pe = pe.permute(1, 0, 2)  # (max_len, 1, d_model) -> (1, max_len, d_model) 
            
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)
        print(f"[PositionalEncoding] Fixed: d_model={d_model}, max_len={max_len}, batch_first={batch_first}")
        
    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        if self.batch_first:
            x = x + self.pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024, batch_first=False):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        if self.batch_first:
            self.pe = nn.Parameter(torch.empty(1, max_len, d_model))
        else:
            self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)  # 将self.pe中填充为从均匀分布U(-0.02, 0.02)中采样的随机数
        # print(f"[PositionalEncoding] Learnable: d_model={d_model}, max_len={max_len}, batch_first={batch_first}")
        

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        if self.batch_first:
            x = x + self.pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))