import torch
import torch.nn as nn

def ScaledSig(x,beta=4.):
    y = beta*x
    S = nn.Sigmoid()
    return S(y)

def gate_score(model, x):
    g = x
    gate_scores = [x]
    for i in range(model.depth):
        g = model.gates[i](g)
        gate_scores.append(model.ScaledSig(g))
    return gate_scores

def compute_npk(U, V, model: torch.nn.Module) -> torch.Tensor:
  gate_scores1 = gate_score(model,U.to(model.device))
  gate_scores2 = gate_score(model,V.to(model.device))
  overlap_kernel = 1
  for i in range(1,len(gate_scores1)):
    el1 = gate_scores1[i].to(model.device)
    el2 = gate_scores2[i].to(model.device)                   # careful to check the 4 instances of .to(model.device) later...
    overlap_kernel *= torch.matmul(el1,el2.T)
  return overlap_kernel.detach().cpu()                      # also check if detach().cpu() is necessary

class NPK():
    """
    Class for NPK computation, designed to be used along with sklearn.svm.SVR/SVC
    """
    def __init__(self, model: torch.nn.Module) -> None:
      self.model = model

    def get_npk(self, U, V) -> torch.Tensor:
      return compute_npk(U, V, self.model)