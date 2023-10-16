import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .template import MetaTemplate
from .bdc_module import BDC

class MetaDeepBDC(MetaTemplate):
    def __init__(self, params, model_func, n_way, n_support):
        super(MetaDeepBDC, self).__init__(params, model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        print("pure BDC")
        
        reduce_dim = params.reduce_dim
        self.feat_dim = int(reduce_dim * (reduce_dim+1) / 2)
        self.dcov = BDC(is_vec=True, input_dim=self.feature.feat_dim, dimension_reduction=reduce_dim)

    def feature_forward(self, x):
        out = self.dcov(x)
        return out

    def set_forward_adapt(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        scores = self.metric(z_query, z_proto)
        return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())
        y_label = np.repeat(range(self.n_way), self.n_query)
        scores = self.set_forward_adapt(x)
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_label)

        return float(top1_correct), len(y_label), self.loss_fn(scores, y_query), scores

    def metric(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        if self.n_support > 1:
            dist = torch.pow(x - y, 2).sum(2)
            score = -dist
        else:
            # dist = torch.pow(x - y, 2).sum(2)
            # score = -dist
            score = (x * y).sum(2)
            # score = self.cos_dist(x ,y).sum(2)
        return score

    def cos_dist(self, x, y):
        # x: N x D
        # y: M x D
        # n = x.size(0)
        # m = y.size(0)
        # d = x.size(1)
        # assert d == y.size(1)

        # x = x.unsqueeze(1).expand(n, m, d)
        # y = y.unsqueeze(0).expand(n, m, d)
        # print(1)
        # ipdb.set_trace()

        x_norm = torch.norm(x, p=2, dim=2, keepdim=True).expand_as(x)
        y_norm = torch.norm(y, p=2, dim=2, keepdim=True).expand_as(y)

        # gt = nn.CosineSimilarity(dim = 2)(x, y)
        cos_sim = (x/x_norm) * (y/y_norm)

        return cos_sim
