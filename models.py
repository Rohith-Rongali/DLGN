import torch
import torch.nn as nn


class DLGN(nn.Module):
    def __init__(self, dim_in, width, depth, beta,bias_fn=True,bias_vn=False):
        super().__init__()
        self.depth = depth
        self.width = width
        self.beta = beta
        self.gates = nn.ModuleList([nn.Linear(dim_in if i == 0 else width, width, bias=bias_fn) for i in range(self.depth)])  
        self.weights = nn.ModuleList([nn.Linear(width, width, bias=bias_vn) for _ in range(self.depth)])
        self.weight_last = nn.Linear(width, 1, bias=False)
        self.dim_in = dim_in
        self.sigmoid = nn.Sigmoid()

    def ScaledSig(self,x):
        y = self.beta*x
        S = nn.Sigmoid()
        return S(y)

    def forward(self, x):
        g = x
        h = torch.ones(self.width).to(x.device)

        for i in range(self.depth):
            g = self.gates[i](g)
            h = self.ScaledSig(g) * self.weights[i](h)

        h_last = self.weight_last(h)
        return self.sigmoid(h_last)

class DNN(nn.Module):
    def __init__(self, dim_in, dim_out, width, depth):
        super(DNN, self).__init__()
        self.depth = depth
        self.layers = nn.ModuleList([nn.Linear(dim_in if i == 0 else width, width) for i in range(self.depth)])
        self.output_layer = nn.Linear(width, dim_out)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
        x = self.output_layer(x)
        return self.sigmoid(x)