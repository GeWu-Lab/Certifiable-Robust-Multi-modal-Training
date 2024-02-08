import torch
import torch.nn as nn
import numpy as np

class ConcatFusion(nn.Module):
    def __init__(self, input_dim=1024, output_dim=100):
        super(ConcatFusion, self).__init__()
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        output = torch.cat((x, y), dim=1)
        output = self.fc_out(output)
        return x, y, output

class OrthFusion(nn.Module):
    def __init__(self, input_dim=[512, 512], output_dim=100):
        super(OrthFusion, self).__init__()
        self.orth_layer = orthogonal_linear(input_dim, output_dim)

    def forward(self, x, y):
        x_input = torch.cat((x, y), dim=1)
        outs, output = self.orth_layer(x_input)
        return outs[0], outs[1], output

# Thanks to Dr. Huang! Please refer to https://github.com/huangleiBuaa/OthogonalWN for the implementation of orthogonal weight normalization.

class OWNNorm(torch.nn.Module):
    def __init__(self, norm_groups=1, *args, **kwargs):
        super(OWNNorm, self).__init__()
        self.norm_groups = norm_groups

    def forward(self, weight: torch.Tensor):
        assert weight.shape[0] % self.norm_groups == 0
        Z = weight.view(self.norm_groups, weight.shape[0] // self.norm_groups, -1)  # type: torch.Tensor
        Zc = Z - Z.mean(dim=-1, keepdim=True)
        S = torch.matmul(Zc, Zc.transpose(1, 2))
        wm = torch.randn(S.shape).to(S)
        for i in range(self.norm_groups):
            U, Eig, _ = S[i].svd()
            Scales = Eig.rsqrt().diag()
            wm[i] = U.mm(Scales).mm(U.t())
        W = wm.matmul(Zc)
        return W.view_as(weight)

class orthogonal_linear(torch.nn.Module):
    """Linear Layer with Xavier Initialization, and 0 Bias."""
    
    def __init__(self, indim, outdim, xavier_init=False):
        """Initialize Linear Layer w/ Xavier Init.

        Args:
            indim (int): Input Dimension
            outdim (int): Output Dimension
            xavier_init (bool, optional): Whether to apply Xavier Initialization to Layer. Defaults to False.
        
        """
        super(orthogonal_linear, self).__init__()
        self.indim = indim
        self.modal_num = len(indim)
        self.input_dim = np.array(indim).sum()
        self.fc = nn.Linear(self.input_dim, outdim)
        self.orthogonal = torch.nn.ModuleList(OWNNorm(norm_groups = outdim) for i in range(self.modal_num))
        self.a_0 = nn.Parameter(torch.ones(outdim) * 0.1)
        self.a_1 = nn.Parameter(torch.ones(outdim) * 0.1)
        if xavier_init:
            nn.init.xavier_normal(self.fc.weight)
            self.fc.bias.data.fill_(0.0)

    def forward(self, x):
        """Apply Linear Layer to Input.

        Args:
            x (torch.Tensor): Input Tensor

        Returns:
            torch.Tensor: Output Tensor
        
        """
        cur_ind = 0
        outs = []
        for v in range(self.modal_num):       
            orth_weight_v = self.orthogonal[v](self.fc.weight[:, cur_ind : cur_ind + self.indim[v]] )
            x_v = x[:, cur_ind : cur_ind + self.indim[v]]         
            cur_ind = cur_ind + self.indim[v]
            outs.append(x_v.matmul(orth_weight_v.t()))
            if v == 0:
                output = outs[v]  * getattr(self, "a_"+str(v)).view(1, -1).expand(outs[v].shape)
            else:
                output = output + outs[v] * getattr(self, "a_"+str(v)).view(1, -1).expand(outs[v].shape)
        return outs, output + self.fc.bias