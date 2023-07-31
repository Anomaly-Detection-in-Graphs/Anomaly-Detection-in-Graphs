import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import GCNConv, SAGEConv
from tqdm import tqdm
import math

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=SAGEConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        
        self.k = k
        self.num_layers = k
        bias = True
        if k == 1:
            self.conv = [base_model(in_channels, out_channels)]
            self.linear = [Linear(in_channels, out_channels, bias=bias).cuda()]
        else:
            self.conv = [base_model(in_channels, 2 * out_channels)]
            self.linear = [Linear(in_channels, 2 * out_channels, bias=bias)]
            for _ in range(1, k-1):
                self.conv.append(base_model(2 * out_channels, 2 * out_channels))
                self.linear.append(Linear(2 * out_channels, 2 * out_channels, bias=bias).cuda())
            self.conv.append(base_model(2 * out_channels, out_channels))
            self.linear.append(Linear(2 * out_channels, 2 * out_channels, bias=bias).cuda())
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x, adjs, attr):
        
        for i, (edge_index, batch, size) in enumerate(adjs):
            print('hhhhhhh',attr[batch].get_device())
            if i <= self.k-1:
                if isinstance(x, list):
                    x_target = x[i][:size[1]]
                    # print('xxxxxxxxxxx',x.size())
                    # print('tttttttt',x_target.size())
                    if i >= 1 and i < self.k-1:
                        x = x[i] + self.conv[i]((x[i], x_target), edge_index)
                    else:
                        x = self.conv[i]((x[i], x_target), edge_index)
                    x = self.activation(x)
                else:
                    x_target = x[:size[1]]
                    # print('xxxxxxxxxxx',x.size())
                    # print('tttttttt',x_target.size())
                    x = self.conv[i]((x, x_target), edge_index)
                    
                    x_attr = self.linear[i](attr[batch].cuda())
                    x = x + 0.25 * x_attr

                    x = self.activation(x)
                    # print(x.size(), i)
        return x
    
    def inference(self, x_all, x_attr, layer_loader, device):
        pbar = tqdm(total=x_all.size(0) * self.num_layers, ncols=80)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in layer_loader:
                edge_index, batch, size = adj.to(device)
                attr = x_attr.to(device)[batch]
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                # x = self.conv[i](x, edge_index)
                x = self.conv[i]((x, x_target), edge_index)
                x_attr = self.linear[i](attr)
                x = x + 0.25 * x_attr
                x = self.activation(x)
                
                xs.append(x)

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5, sim_filter_threshold = None):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
        self.sim_filter_threshold = sim_filter_threshold/1
        self.start_epoch = 20
        self.dec_rate = 0.001
        self.con_value = sim_filter_threshold/2
        print(self.sim_filter_threshold, self.dec_rate, self.con_value)
        
    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor, attr: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index, attr)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
    
    def reset_parameters(self):
        for conv in self.encoder.conv:
            conv.reset_parameters()   

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        # print('z1',z1.size())
        # print('z2',z2.size())
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int, cur_epoch: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        # losses = [torch.zeros(z1.size()[1], requires_grad=True).to(device)]
        losses = [(torch.rand(z1.size()[1], requires_grad=True)/64).to(device)]
        
        if self.sim_filter_threshold is not None and cur_epoch > self.start_epoch:
            decline_ratio = math.exp(-self.dec_rate*(cur_epoch - self.start_epoch))
            threshold = (self.sim_filter_threshold-self.con_value) *decline_ratio + self.con_value
        else:
            threshold = self.sim_filter_threshold

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            s_ik = self.sim(z1[mask], z2)
            if s_ik.mean() > threshold:
                # print(s_ik.mean())
                
                refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
                between_sim = f(s_ik)  # [B, N]

                losses.append(-torch.log(
                    between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                    / (refl_sim.sum(1) + between_sim.sum(1)
                       - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
#         device = z1.device
#         num_nodes = z1.size(0)
#         num_batches = (num_nodes - 1) // batch_size + 1
#         f = lambda x: torch.exp(x / self.tau)
#         indices = torch.arange(0, num_nodes).to(device)
#         losses = []

#         for i in range(num_batches):
#             s_ik = self.sim(z1[mask], z2)
#             if s_ik.mean() > 0.005:
#                 print(s_ik.mean())
#                 mask = indices[i * batch_size:(i + 1) * batch_size]
#                 refl_sim = f(self.sim(z1[mask], z1))  # [B, N]

#                 between_sim = f(s_ik)  # [B, N]

#                 losses.append(-torch.log(
#                     between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
#                     / (refl_sim.sum(1) + between_sim.sum(1)
#                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = 0, cur_epoch: int = 0):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size == 0:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size, cur_epoch)
            l2 = self.batched_semi_loss(h2, h1, batch_size, cur_epoch)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x
