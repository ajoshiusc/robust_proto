import torch
from torch.nn import functional as F
from torch.nn.modules import Module
import pdb
import numpy as np


class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def contrastive_loss(X, tau):
    #X: B x 10
    nume = torch.exp(torch.diagonal(torch.matmul(X, X.T/tau), 0))
    loss = -torch.log(nume / torch.sum(torch.exp(torch.matmul(X, X.T/tau)), dim=1)).mean()

    return loss

def prototypical_loss(inputs, target, n_support):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py
    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    target_cpu = target.to('cpu')
    input_cpu = inputs.to('cpu')

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    # n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))

    prototypes = F.normalize(torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs]), dim=1) #normalization across different class prototypes, the shape is 10x10, first 10 is class prototypes, second 10 is there feature vector dimension.
    # FIXME when torch will support where as np
    
    # need normalization
    query_idxs = torch.cat(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:].squeeze(1), classes)))
    #.view(-1)
    n_query_of_cls = list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:].squeeze(1).size()[0], classes))
    query_samples = input_cpu[query_idxs]
    dists = euclidean_dist(query_samples, prototypes)
    log_p_y = F.log_softmax(-dists, dim=1)
    # target_inds = torch.arange(0, n_classes)
    # target_inds = target_inds.view(n_classes, 1, 1)
    # target_inds = target_inds.expand(n_classes, n_query, 1).long()
    target_inds = torch.zeros(sum(n_query_of_cls), 1, dtype=torch.int64)
    for i, n in enumerate(n_query_of_cls):
        curr_i = sum(n_query_of_cls[:i])
        target_inds[curr_i:curr_i+n] = i
    loss_proto = -log_p_y.gather(dim=1, index=target_inds).squeeze().view(-1).mean()
    # loss_con = contrastive_loss(input_cpu, tau)
    # loss_val = loss_ce + lamda2 * loss_proto + lamda1 * loss_conn
    loss_val = loss_proto
    # if log_p_y.size()[0] == 0:
    # print(log_p_y)
    # pdb.set_trace()
    _, y_hat = log_p_y.max(1)
    # pdb.set_trace()
    acc_val = y_hat.eq(target_inds[:, 0]).reshape((10, -1)).float().mean(axis=1)
    
    return loss_val, acc_val#, y_hat