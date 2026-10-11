from .simformer import SIMformer
import torch
from torch import nn

class DualSIMformer(nn.Module):
    def __init__(self, grid_num, grid_embedding_size, hidden_dim, dimfeedforward=256, n_heads=16, num_layers=1, pos_encoding="learnable"):
        super().__init__()
        self.embs = nn.Embedding(num_embeddings=grid_num, embedding_dim=grid_embedding_size)

        self.simformer1 = SIMformer(feat_dim=2,  # input dim
                            hidden_dim=hidden_dim,  # 输入2维轨迹输入SIMformer后被线性映射到args.hidden_dim维, 也是Transformer的input_dim, 亦是轨迹被编码为向量的维度
                            num_layers=num_layers,  # Transformer Encoder中block的数量
                            n_heads=n_heads, 
                            dimfeedforward=2 * hidden_dim,  # Transformer中feedforward network的hidden_size
                            pos_encoding=pos_encoding) # "learnable" or "fixed"
        
        self.simformer2 = SIMformer(feat_dim=grid_embedding_size,  # input dim
                            hidden_dim=hidden_dim,  # 输入2维轨迹输入SIMformer后被线性映射到args.hidden_dim维, 也是Transformer的input_dim, 亦是轨迹被编码为向量的维度
                            num_layers=num_layers,  # Transformer Encoder中block的数量
                            n_heads=n_heads, 
                            dimfeedforward=2 * hidden_dim,  # Transformer中feedforward network的hidden_size
                            pos_encoding=pos_encoding) # "learnable" or "fixed"
        
        self.linear = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, x, x_grid, padding_masks):

        x = self.simformer1(x, padding_masks)
        grid_embeddings = self.embs(x_grid.long())
        grid_x = self.simformer2(grid_embeddings, padding_masks)

        final_x = torch.concat((x, grid_x), dim = -1)

        return self.linear(final_x)
        