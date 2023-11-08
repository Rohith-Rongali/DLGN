import torch
import torch.nn as nn


class DLGN(nn.Module):
    def __init__(self, dim_in, width, depth, beta,bias_fn=True,bias_vn=False,const_init=False):
        super().__init__()
        self.depth = depth
        self.width = width
        self.beta = beta
        self.gates = nn.ModuleList([nn.Linear(dim_in if i == 0 else width, width, bias=bias_fn) for i in range(self.depth)])
        self.weights = nn.ModuleList([nn.Linear(width, width, bias=bias_vn) for _ in range(self.depth)])
        if const_init:
          for i in range(self.depth):
            nn.init.constant_(self.weights[i].weight,1.0/width)
        self.weight_last = nn.Linear(width, 1, bias=True)
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
    


class DLGN_SF(nn.Module):
    def __init__(self, dim_in, width, depth, beta,bias_fn=True,bias_vn=False):
        super().__init__()
        self.depth = depth
        self.width = width
        self.beta = beta
        self.gates = nn.ModuleList([nn.Linear(dim_in, width, bias=bias_fn) for i in range(self.depth)]) #This is the only diff between DLGN and DLGN_SF
        self.weights = nn.ModuleList([nn.Linear(width, width, bias=bias_vn) for _ in range(self.depth)])
        self.weight_last = nn.Linear(width, 1, bias=True)
        self.dim_in = dim_in
        self.sigmoid = nn.Sigmoid()

    def ScaledSig(self,x):
        y = self.beta*x
        S = nn.Sigmoid()
        return S(y)

    def forward(self, x):
        h = torch.ones(self.width).to(x.device)

        for i in range(self.depth):
            g = self.gates[i](x)   #This is the only diff between DLGN and DLGN_SF
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
        return x                    # no activation on output layer. This is done in the loss function like nn.CrossEntropyLoss() and nn.BCELosswithLogits()
    

class DLGN_Kernel(nn.Module):
    def __init__(self, num_data, dim_in, width, depth, beta = 4):   #output dimension is 1
        super().__init__()
        self.num_data = num_data
        self.beta = beta
        self.dim_in = dim_in
        self.width = width
        self.depth = depth
        sigma = 1/torch.sqrt(width)
        self.gates = nn.ParameterList([nn.Parameter(sigma*torch.randn(dim_in if i == 0 else width, width)) for i in range(depth)])
        self.alphas = nn.Parameter(torch.randn(num_data)/torch.sqrt(num_data))
        #self._cache = None

    def ScaledSig(self,x):
        y = self.beta*x
        S = nn.Sigmoid()
        return S(y)
    
    def get_weights(self):
        A = [self.gates[0]]
        for i in range(1,self.depth):
            A.append(A[-1]@self.gates[i])
        return torch.vstack(A)


    def forward(self, inp, data):
        #ones = torch.ones(self.dim_in).to(x.device())
        #self._cache = None
        data_gate_matrix = data @ self.gates[0]
        data_gate_score = self.ScaledSig(data_gate_matrix, self.beta)
        inp_gate_matrix = inp @ self.gates[0]
        inp_gate_score = self.ScaledSig(inp_gate_matrix, self.beta)
        output_kernel =  (inp_gate_score @ data_gate_score.T)
        for i in range(1,self.depth):
            data_gate_matrix = data_gate_matrix @ self.gates[i]
            inp_gate_matrix = inp_gate_matrix @ self.gates[i]
            data_gate_score = self.ScaledSig(data_gate_matrix, self.beta)
            inp_gate_score = self.ScaledSig(inp_gate_matrix, self.beta)
            output_kernel *= (inp_gate_score @ data_gate_score.T)/self.width
        #print(torch.max(output_kernel), torch.min(output_kernel))
        return self.ScaledSig(output_kernel @ self.alphas, 1)

