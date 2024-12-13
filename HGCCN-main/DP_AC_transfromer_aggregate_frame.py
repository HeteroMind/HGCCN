import torch.nn.functional as F
from torch import nn
import torch
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv

from typing import Any, Callable, List, NamedTuple, Optional
from functools import partial

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=128):#64
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        #原本的：
        w = self.project(z)
        beta = torch.softmax(w, dim=0)
        return (beta * z).sum(0), beta

class M_GCN_t(nn.Module):
    def __init__(self, num_features_list, hidden_dim):
        super().__init__()
        self.view_num = len(num_features_list)
        self.conv1 = nn.ModuleList([GATConv(in_channels, hidden_dim) for in_channels in num_features_list])  #GCN 92.59
        self.conv2 = nn.ModuleList([GATConv(hidden_dim, hidden_dim) for _ in range(self.view_num)])#GCNConv  #GAT 92.59
        #self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads=4)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, dropout=0.1) #heads 8     *4     0.2   92.59
        self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, dropout=0.1) #heads  8     *4    0.2  92.59
        #self.encoder_layer = EncoderBlock(hidden_dim=hidden_dim, num_heads=4, dropout=0.1, attention_dropout=0.1, mlp_dim=hidden_dim * 4)
        #self.encoder_layer1 = EncoderBlock(hidden_dim=hidden_dim, num_heads=4, dropout=0.1, attention_dropout=0.1,mlp_dim=hidden_dim * 4)
        self.dropout = nn.Dropout(0.3) #0.2   92.59
        self.layer_norm = torch.nn.LayerNorm(hidden_dim)
        self.attention = Attention(hidden_dim)
        #self.out = nn.Linear(hidden_dim, out_channels)

    def mv_gnn(self, view_d, edge_list, conv):
        emb_list = []
        for i in range(self.view_num):
            x, edge_index = view_d[i], edge_list[i]
            x = conv[i](x, edge_index)
            x = F.relu(x)
            emb_list.append(x)
        emb_view = torch.stack(emb_list, dim=0)
        return emb_view
    def forward(self, view_data):
        edge_list = [view_data[i].edge_index for i in range(self.view_num)]  #这里是整图的多视角图数据！！！！！！！！！
        view_data_in = [view_data[i].x for i in range(self.view_num)]         #这里是整图的多视角图数据！！！！！！！！！！！
        # [num_views, num_nodes, num_features]

        # 第一层
        emb_view = self.mv_gnn(view_data_in, edge_list, self.conv1)
        emb_view = self.dropout(emb_view)
        #emb_view=self.layer_norm(emb_view)
        # [num_views, num_nodes, emb_dim]
        attn_output = self.encoder_layer(emb_view)


        # 第二层
        emb_view_2 = self.mv_gnn(attn_output, edge_list, self.conv2)
        emb_view_2 = self.dropout(emb_view_2)
        #emb_view_2 = self.layer_norm(emb_view_2)
        attn_output1 = self.encoder_layer1(emb_view_2)

        #attn_output1 = attn_output1 + attn_output

        # 全局特征融合
        global_emb, att = self.attention(attn_output1)
        # global_emb = torch.mean(attn_output1, dim=0)
        #output = self.out(global_emb)
        return  [attn_output, attn_output1], global_emb

        #return output, [attn_output, attn_output1], global_emb


class MLPBlock(nn.Sequential):
    """Transformer MLP block."""

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(mlp_dim, in_dim)
        self.dropout_2 = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.normal_(self.linear_1.bias, std=1e-6)
        nn.init.normal_(self.linear_2.bias, std=1e-6)
class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(query=x, key=x, value=x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y