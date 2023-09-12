import torch

def log_features_dlgn(model,bias_log=False):
  '''
  This function returns the log of the features of the gates in model.
  Returns a 2-d tensor with rows of shape (num_gates, data_dim)
  '''
  weight = []
  bias = []
  for name, param in model.named_parameters():
      for i in range(0,model.depth):
          if name == 'gates.'+str(i)+'.weight':
              weight.append(param.data)
          if bias_log:
            if name == 'gates.'+str(i)+'.bias':
                bias.append(param.data)

  Feature_list = [weight[0]]

  for w in weight[1:]:
    Feature_list.append(w @ Feature_list[-1])

  features = torch.cat(Feature_list, axis = 0)

  return features #make it to .to("cpu") if you want to use it in numpy


def feature_stats(features,data_dim=18,tree_depth=4,dim_in=18,threshold=0.1,req_index=False): #can set tree_depth=0 to get root node stats...
  '''
  Returns the count of features that are close to the standard basis vectors within a threshold
  Can return the indices of the features as well if req_index=True
  count is a 1-d tensor of length 2**tree_depth-1
  index is a list of lists of length 2**tree_depth-1 with each list containing the indices of the
  features that are close to the standard basis vector corresponding to that node.
  '''
  num_nodes = 2**tree_depth-1
  tensor = torch.eye(data_dim)  #standard basis
  y=torch.randn(dim_in)
  rand_point=y/torch.norm(y, p=2)

  count = torch.zeros(num_nodes)
  index = [[]]*num_nodes
  for ind,item in enumerate(features):
      for i in range(num_nodes):
        if torch.linalg.vector_norm(item/(item.norm(dim=0, p=2))-tensor[i]) < threshold or torch.linalg.vector_norm(item/(item.norm(dim=0, p=2))+tensor[i]) < threshold:
            count[i] += 1
            if req_index:
              index[i].append(ind)
  if req_index:
    return count,index
  else:
    return count
