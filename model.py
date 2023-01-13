import math
import torch
import torch.nn.functional as F
import torch_scatter
from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
#from torch_geometric.nn import GatedGraphConv

class GCNResidualBlock(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gcn_conv = GCNConv(hidden_size, hidden_size) # (hidden, num_out_features_per_node)
        self.linear = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, x, edge_index):
        x_block = self.gcn_conv(x, edge_index)
        x_block = F.relu(x_block)
        x_block = self.linear(x_block)
        x_block = F.relu(x_block)
        return x_block+x


# Residual connection
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size = 1280
        p = 0.4
        self.stem = torch.nn.Sequential(
            torch.nn.Linear(10, hidden_size),
            torch.nn.ReLU(),
        )

        self.block1 = GCNResidualBlock(hidden_size)
        self.bn1 = torch.nn.BatchNorm1d(hidden_size)
        self.do1 = torch.nn.Dropout(p)

        self.block2 = GCNResidualBlock(hidden_size)
        self.bn2 = torch.nn.BatchNorm1d(hidden_size)
        self.do2 = torch.nn.Dropout(p)

        self.block3 = GCNResidualBlock(hidden_size)
        self.bn3 = torch.nn.BatchNorm1d(hidden_size)
        self.do3 = torch.nn.Dropout(p)

        self.block4 = GCNResidualBlock(hidden_size)
        self.bn4 = torch.nn.BatchNorm1d(hidden_size)
        self.do4 = torch.nn.Dropout(p)

        self.block5 = GCNResidualBlock(hidden_size)
        self.bn5 = torch.nn.BatchNorm1d(hidden_size)
        self.do5 = torch.nn.Dropout(p)

        self.out = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 7)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.stem(x)
        x = self.block1(x, edge_index)
        x = self.bn1(x)
        x = self.do1(x)

        x = self.block2(x, edge_index)
        x = self.bn2(x)
        x = self.do2(x)

        x = self.block3(x, edge_index)
        x = self.bn3(x)
        x = self.do3(x)

        x = self.block4(x, edge_index)
        x = self.bn4(x)
        x = self.do4(x)

        x = self.block5(x, edge_index)
        x = self.bn5(x)
        x = self.do5(x)

        x = self.out(x)
        return x


class MLP(torch.nn.Module):
    """Multi-Layer perceptron"""
    def __init__(self, input_size, hidden_size, output_size, layers, layernorm=True):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(layers):
            self.layers.append(torch.nn.Linear(
                input_size if i == 0 else hidden_size,
                output_size if i == layers - 1 else hidden_size,
            ))
            if i != layers - 1:
                self.layers.append(torch.nn.ReLU())
        if layernorm:
            self.layers.append(torch.nn.LayerNorm(output_size))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data.normal_(0, 1 / math.sqrt(layer.in_features))
                layer.bias.data.fill_(0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class InteractionNetwork(MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""
    def __init__(self, hidden_size, layers):
        super().__init__()
        self.lin_edge = MLP(hidden_size * 3, hidden_size, hidden_size, layers)
        self.lin_node = MLP(hidden_size * 2, hidden_size, hidden_size, layers)

    def forward(self, x, edge_index, edge_feature):
        edge_out, aggr = self.propagate(edge_index, x=(x, x), edge_feature=edge_feature)
        node_out = self.lin_node(torch.cat((x, aggr), dim=-1))
        edge_out = edge_feature + edge_out
        node_out = x + node_out
        return node_out, edge_out

    def message(self, x_i, x_j, edge_feature):
        x = torch.cat((x_i, x_j, edge_feature), dim=-1)
        x = self.lin_edge(x)
        return x

    def aggregate(self, inputs, index, dim_size=None):
        out = torch_scatter.scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")
        return (inputs, out)


class LearnedSimulator(torch.nn.Module):
    """Graph Network-based Simulators(GNS)"""
    def __init__(
        self,
        hidden_size=128,
        n_mp_layers=10, # number of GNN layers
        dim=3, # dimension of the world, typical 2D or 3D
    ):
        super().__init__()
        self.node_in = MLP(dim * 2, hidden_size, hidden_size, 3)
        self.edge_in = MLP(dim + 1, hidden_size, hidden_size, 3)
        self.node_out = MLP(hidden_size, hidden_size, dim, 3, layernorm=False)
        self.n_mp_layers = n_mp_layers
        self.layers = torch.nn.ModuleList([InteractionNetwork(
            hidden_size, 3
        ) for _ in range(n_mp_layers)])

    def forward(self, data):
        """Forward pass through GNS.

        To understand the forward pass, use:
            N = number of nodes per graph, which is 9 for real data.
            B = batch size, or the number of graphs, which is 512 by default.

        Recall that for _each_ graph, the input is (9,6) since we have 6 features
        per node (3D position and 3D force vector). However, each graph has slightly
        different edge counts, so the size of `data.edge_attr` may vary.
        """

        # converts each node (in each graph) into 128-dim features.
        # data.x: [512*9,6] or [4608,6] ==> [4608,128]
        node_feature = self.node_in(data.x)

        # converts each edge (in each graph) into 128-dim features.
        # data.x: [4549, 4] ==> [4549, 128]
        edge_feature = self.edge_in(data.edge_attr)

        # stack of GNN layers -- shapes of node_feature and edge_feature do not change
        for i in range(self.n_mp_layers):
            node_feature, edge_feature = self.layers[i](
                node_feature, data.edge_index, edge_feature=edge_feature
            )

        # post-processing, converts each node (in each graph) to 3D output.
        # node_feature: [4608, 128] ==> out: [4608,3]
        out = self.node_out(node_feature)
        return out


if __name__=='__main__':
    simulator = LearnedSimulator()
    print(simulator)
