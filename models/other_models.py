import torch.nn as nn
from .simformer import get_pos_encoder, mean_pooling

class my_GRU(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_layers=2, hidden_size=128, bidirectional=False
                 , pos_encoding="learnable"):
        """ 
        Args:
            - feat_dim (int): Input feature dimension, for trajectory data, feat_dim=2, which is longitude and latitude
            - hidden_dim (int): Dimension after linear mapping of longitude and latitude, which is the input dimension of rnn
            - num_layers (int): Number of layers in rnn
            - 
            - pos_encoding (str): Position encoding method, can be learnable or sin_cos
        """
        super(my_GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = 200 # set according to the maximum trajectory length in the dataset, and used for positional encoding
        self.project_inp = nn.Linear(feat_dim, hidden_dim) # point dimensionality enhancement
        self.pos_enc = get_pos_encoder(pos_encoding)(hidden_dim, max_len=self.max_seq_len, batch_first=True)  # self.pos_enc.pe的shape: [1, self.max_seq_len, hidden_dim]
        self.rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers, 
            bidirectional=bidirectional
        )
    
        # This is to ensure that the output is a positive number, 
        # controlling the range of similarity between 0 and 1, because sometimes cosine similarity is used.
        self.act = nn.ReLU()
    
    def forward(self, x, padding_masks):  # x的shape: [batch_size*sampling_num, max_seq_len, 2]
        input = self.project_inp(x)
        input = self.pos_enc(input)  # add positional encoding 后接一层dropout
        output = self.rnn(input)[0]
        output = mean_pooling(output, padding_masks)  # 对输出的向量padding的地方置零后求均值 [batch_size*sampling_num, hidden_dim]
        output = self.act(output)  # 将x映射为hidden vector, [batch_size*sampling_num, hidden_dim]
        return output
    
class my_LSTM(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_layers=2, hidden_size=128, bidirectional=False
                 , pos_encoding="learnable"):
        """ 
        Args:
            - feat_dim (int): Input feature dimension, for trajectory data, feat_dim=2, which is longitude and latitude
            - hidden_dim (int): Dimension after linear mapping of longitude and latitude, which is the input dimension of rnn
            - num_layers (int): Number of layers in rnn
            - 
            - pos_encoding (str): Position encoding method, can be learnable or sin_cos
        """
        super(my_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = 200 # set according to the maximum trajectory length in the dataset, and used for positional encoding
        self.project_inp = nn.Linear(feat_dim, hidden_dim) # point dimensionality enhancement
        self.pos_enc = get_pos_encoder(pos_encoding)(hidden_dim, max_len=self.max_seq_len, batch_first=True)  # self.pos_enc.pe的shape: [1, self.max_seq_len, hidden_dim]
        self.rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers, 
            bidirectional=bidirectional
        )
    
        # This is to ensure that the output is a positive number, 
        # controlling the range of similarity between 0 and 1, because sometimes cosine similarity is used.
        self.act = nn.ReLU()
    
    def forward(self, x, padding_masks):  # x的shape: [batch_size*sampling_num, max_seq_len, 2]
        input = self.project_inp(x)
        input = self.pos_enc(input)  # add positional encoding 后接一层dropout
        output = self.rnn(input)[0]
        output = mean_pooling(output, padding_masks)  # 对输出的向量padding的地方置零后求均值 [batch_size*sampling_num, hidden_dim]
        # output = self.act(output)  # 将x映射为hidden vector, [batch_size*sampling_num, hidden_dim]
        return output
    
class my_Linear(nn.Module):
    def __init__(self, feat_dim, hidden_dim):
        super(my_Linear, self).__init__()
        self.linear = nn.Linear(feat_dim, hidden_dim)
        self.act = nn.ReLU()

    def forward(self, x, padding_masks):
        input = self.linear(x)
        output = mean_pooling(input, padding_masks)
        output = self.act(output)  # 将x映射为hidden vector, [batch_size*sampling_num, hidden_dim]
        return output
