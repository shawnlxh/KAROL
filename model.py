import torch
from torch import Tensor
print(torch.__version__)
from torch import nn
from torch.nn.functional import normalize
from torch_geometric.nn.conv import MessagePassing, HeteroConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import NoneType
from typing import Union, Tuple, Optional
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.utils import (
    add_self_loops,
    is_torch_sparse_tensor,
    remove_self_loops,
    softmax,
)
from torch_geometric.nn.conv import LGConv
from torch.nn import ModuleList

class GATConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.edge_dim = edge_dim
        
        self.lin_src = Linear(in_channels[0], out_channels, False, weight_initializer='glorot')
        self.lin_dst = Linear(in_channels[1], out_channels, False, weight_initializer='glorot')
        
        self.att_src = Parameter(torch.empty(1, out_channels))
        self.att_dst = Parameter(torch.empty(1, out_channels))
        
        self.lin_edge = Linear(edge_dim, out_channels, bias=False, weight_initializer='glorot')
        self.att_edge = Parameter(torch.empty(1, out_channels))
        
        self.bias = Parameter(torch.empty(out_channels))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        super().reset_parameters()
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)
        
    def forward(self, x, edge_index, edge_attr = None, size = None, return_attention_weights=None):
        C = self.out_channels
        x_src, x_dst = x
        x_src = self.lin_src(x_src).view(-1, C)
        x_dst = self.lin_src(x_dst).view(-1, C)
        x = (x_src, x_dst)
        
        
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)
        
        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)
        out = out + x_dst
        if self.bias is not None:
            out = out + self.bias
        return out 
        
    def edge_update(self, alpha_j, alpha_i, edge_attr, index, ptr, size_i):
        alpha = alpha_j + alpha_i
        if index.numel() == 0:
            return alpha
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)
        edge_attr = self.lin_edge(edge_attr)
        edge_attr = edge_attr.view(-1, self.out_channels)
        alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
        alpha = alpha + alpha_edge
        
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha
    
    
    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return alpha.unsqueeze(-1) * x_j

    

    
class Model_i2s(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers = 2):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and items:
        # Instantiate homogeneous GNN:
        # Convert GNN model into a heterogeneous variant:
        self.sub_graph_i2s = data.edge_type_subgraph([('item', 'i2s', 'seller'),('seller', 's2i', 'item')])
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({('item', 'i2s', 'seller'): \
                            GATConv((hidden_channels, hidden_channels), hidden_channels,edge_dim = hidden_channels ),
                ('seller', 's2i', 'item'): \
                    GATConv((hidden_channels, hidden_channels), hidden_channels,edge_dim = hidden_channels)},aggr='sum')
            self.convs.append(conv)

    def forward(self, sampled_data: HeteroData, x_dict,edge_attr) -> Tensor:
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        for conv in self.convs:
            x_dict = conv(x_dict, sampled_data.edge_index_dict, edge_attr_dict = edge_attr)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
        return x_dict


class Lightgcn(torch.nn.Module):
    def __init__(self, num_layers = 2, normalize = True,alpha: Optional[Union[float, Tensor]] = None):
        super().__init__()
        self.normalize = normalize
        self.num_layers = num_layers
        if alpha is None:
            alpha = 1. / (num_layers + 1)

        if isinstance(alpha, Tensor):
            assert alpha.size(0) == num_layers + 1
        else:
            alpha = torch.tensor([alpha] * (num_layers + 1))
        self.register_buffer('alpha', alpha)
        self.convs = ModuleList([LGConv(normalize = self.normalize) for _ in range(self.num_layers)])
        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight = None,perturbed = False,device = 'cpu') -> Tensor:
        # Define a 2-layer GNN computation graph.
        # Use a *single* `ReLU` non-linearity in-between.
        eps = 0.1
        out = x * self.alpha[0]
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            if perturbed:
                random_noise = torch.rand_like(x).to(device)
                x += torch.sign(x) * F.normalize(random_noise, dim=-1) * eps
            out = out + x * self.alpha[i + 1]
        return out


    
class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x_h: Tensor, x_t: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_h = x_h[edge_label_index[0]]
        edge_feat_t = x_t[edge_label_index[1]]
        #print("edge_feat_user",edge_feat_user.size())
        #print("edge_feat_item",edge_feat_item.size())
        # Apply neural tensor layer to get a prediction per supervision edge:
        output = (edge_feat_h  * edge_feat_t).sum(dim=-1)
        return output 
    
    
class Model_u2i(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and items:
        # Convert GNN model into a heterogeneous variant:
        self.sub_graph_u2i = data.edge_type_subgraph\
            ([('user', 'u2i', 'item'),('item', 'i2u', 'user')])
        self.conv = Lightgcn()
        self.classifier = Classifier()
        
    def forward(self, sampled_data: HeteroData, x_dict, stage, device, perturbed) -> Tensor:
        
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        homo_data = sampled_data.to_homogeneous()
        u_len = x_dict['user'].size()[0]
        i_len = x_dict['item'].size()[0]
        x_concat = torch.cat((x_dict['user'],x_dict['item']),0)
        if perturbed:
            x_concat = self.conv(x_concat, homo_data.edge_index,perturbed = perturbed,device = device) 
        else:
            x_concat = self.conv(x_concat, homo_data.edge_index) 
        x_dict["user"] = x_concat[:u_len,:]
        x_dict["item"] = x_concat[u_len:,:]
        if stage == 'stage1':
            pred = self.classifier(
                x_dict["user"],
                x_dict["item"],
                sampled_data["user", "u2i", "item"].edge_label_index,
            )
            return pred
        return x_dict
    
    
class Model_u2s(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and items:        
        # Convert GNN model into a heterogeneous variant:
        self.sub_graph_u2s = data.edge_type_subgraph\
            ([('user', 'u2s', 'seller'),('seller', 's2u', 'user'),("item", "i2s", "seller"),("seller", "s2i", "item")])
        self.conv = Lightgcn()
        self.classifier = Classifier()

    def forward(self, sampled_data: HeteroData, x_dict, stage, device, perturbed) -> Tensor: 
        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        homo_data = sampled_data.to_homogeneous()
        u_len = x_dict['user'].size()[0]
        s_len = x_dict['seller'].size()[0]
        x_concat = torch.cat((x_dict['user'],x_dict['seller']),0)
        if perturbed:
            x_concat = self.conv(x_concat, homo_data.edge_index,perturbed = perturbed,device = device) 
        else:
            x_concat = self.conv(x_concat, homo_data.edge_index) 
        x_dict["user"] = x_concat[:u_len,:]
        x_dict["seller"] = x_concat[u_len:,:]
        if stage == 'stage1':
            pred = self.classifier(
                x_dict["user"],
                x_dict["seller"],
                sampled_data["user", "u2s", "seller"].edge_label_index,
            )
            return pred 
        return x_dict
    
class Multi_task_model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.item_lin = torch.nn.Linear(item_feat_dim, hidden_channels)
        self.seller_lin = torch.nn.Linear(seller_feat_dim, hidden_channels)
        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.item_emb = torch.nn.Embedding(data["item"].num_nodes, hidden_channels)
        self.seller_emb = torch.nn.Embedding(data["seller"].num_nodes, hidden_channels)
        #define rel emb: u2i u2i i2s i2u s2u s2i
        self.rel_emb = torch.nn.Embedding(6,hidden_channels)
        self.model_u2i = Model_u2i()
        self.model_u2s = Model_u2s()
        self.model_i2s = Model_i2s(hidden_channels)
    
    def forward(self, sampled_data: HeteroData,device = 'cpu',stage = 'stage1',transe = False,edge_attr = None, perturbed=False) :
        if (('user', 'u2i', 'item') in sampled_data.edge_types):
            x_dict = {
              "user": self.user_emb(sampled_data["user"].node_id),
              "item": sampled_data["item"].x,
            } 
            pred = self.model_u2i(sampled_data,x_dict,stage,device,perturbed)
            return pred
        
        if (("user", "u2s", "seller") in sampled_data.edge_types):
            x_dict = {
              "user": self.user_emb(sampled_data["user"].node_id),
              "seller": sampled_data["seller"].x,
            } 
            pred = self.model_u2s(sampled_data, x_dict,stage,device,perturbed)
            return pred
            
        if (("item", "i2s", "seller") in sampled_data.edge_types):
            x_dict = {
              "item": self.item_emb(sampled_data["item"].node_id),
              "seller": self.seller_emb(sampled_data["seller"].node_id),
            } 
            if transe == True:
                return x_dict, self.rel_emb
            edge_attr_i2s = self.rel_emb(edge_attr[0])
            edge_attr_i2s = edge_attr_i2s.repeat(sampled_data[("item", "i2s", "seller")].edge_index.size()[1],1)
            edge_attr_s2i = self.rel_emb(edge_attr[1])
            edge_attr_s2i = edge_attr_s2i.repeat(sampled_data[("seller", "s2i", "item")].edge_index.size()[1],1)
            edge_attr = {("item", "i2s", "seller"):edge_attr_i2s,\
                         ("seller", "s2i", "item"):edge_attr_s2i}
            return self.model_i2s(sampled_data, x_dict,edge_attr = edge_attr)
