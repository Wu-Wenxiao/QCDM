import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .template import MetaTemplate, channel_weight
from .bdc_module import BDC

class MetaDeepBDC(MetaTemplate):
    def __init__(self, params, model_func, n_way, n_support):
        super(MetaDeepBDC, self).__init__(params, model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        input_dim = self.n_support * self.n_way
        self.tau = 0.0
        self.k = 0.0
        self.m = 0.00
        print("v3max: layer_norm+feature_dim-adapt BDC+init_weight")
        print("tau {:4f} | inputdim {:d} | k {:4f} | margin {:4f}".format(self.tau, input_dim, self.k, self.m))
        self.regularizer = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(inplace=True),
                nn.Linear(input_dim, 1))
        reduce_dim = params.reduce_dim
        self.feat_dim = int(reduce_dim * (reduce_dim+1) / 2)
        self.dcov = BDC(is_vec=True, input_dim=self.feature.feat_dim, dimension_reduction=reduce_dim)
        self.init_weight = channel_weight(num_query=self.n_way*self.n_query, dim=self.feat_dim)
        self.init_weight.initialise()

    def feature_forward(self, x):
        out = self.dcov(x)
        return out

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        scores = self.metric(z_query, z_proto)
        return scores

    def set_forward_adapt(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)
        z_support = z_support.contiguous().view(self.n_way*self.n_support, -1)
        z_query = z_query.contiguous().view(self.n_way*self.n_query, -1)
        # print(z_support.shape)

        # layer_norm = torch.nn.LayerNorm([z_support.shape[-1]], elementwise_affine=False)
        # z_support = layer_norm(z_support)
        # z_query = layer_norm(z_query)

        n = z_query.size(0)
        m = z_support.size(0)
        d = z_query.size(1)
        assert d == z_support.size(1)

        z_q = z_query.unsqueeze(1).expand(n, m, d)
        z_s = z_support.unsqueeze(0).expand(n, m, d)

        if self.n_support > 1:
            sim_euc = torch.pow(z_q - z_s, 2)
        else:
            assert self.n_support==1
            # print(self.n_support)
            sim_euc = z_q * z_s
        input = sim_euc
        # max_col = torch.max(sim_euc, 1)[0]
        # max_col = max_col.unsqueeze(1).repeat(1,self.n_way*self.n_support,1)
        # input = sim_euc/max_col
        # input = torch.exp(input/self.tau)
        # sim_exp = torch.exp(sim_euc/self.tau)
        for i in range(self.n_way*self.n_query):
            if i == 0 :
                output = self.regularizer(input[i].T).T
            else:
                output = torch.cat((output, self.regularizer(input[i].T).T),dim = 0)

        # print(z_support.shape[-1])
        # layer_norm = torch.nn.LayerNorm([z_support.shape[-1]], elementwise_affine=False)
        # z_support = layer_norm(z_support)
        # z_query = layer_norm(z_query)

        z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        n = z_query.size(0)
        m = z_proto.size(0)
        d = z_query.size(1)
        assert d == z_proto.size(1)

        x = z_query.unsqueeze(1).expand(n, m, d)
        y = z_proto.unsqueeze(0).expand(n, m, d)
        
        # print(1)
        assert output.shape==self.init_weight.weight.shape
        output_f = (output*self.init_weight.weight).unsqueeze(1).expand(n, m, d)
        # output_f = output.unsqueeze(1).expand(n, m, d)

        # if self.training:
        #     margin_dir = x - y
        #     margin_norm = F.normalize(margin_dir, p = 2.0, dim = 2)
        #     # y_c =  y - c
        #     # length = torch.sqrt(torch.pow(y_c, 2).sum(2))
        #     length = torch.sqrt(torch.pow(y, 2).sum(2)) # 乘以以O为坐标系的长度
        #     length = length.unsqueeze(2).expand(n, m, d)
        #     # v2 构造仅与同类增大margin的mask
        #     T1 = np.eye(int(self.n_way))
        #     T2 = np.ones((self.n_query, 1))
        #     mask_margin = torch.FloatTensor(np.kron(T1, T2)).unsqueeze(2).expand(n, m, d).cuda()
        #     margin = margin_norm *(mask_margin * length)
        #     x_m = x + margin*self.m
        #     scores = -(output_f*torch.pow(x_m - y, 2)).sum(2)
        # else:
        #     scores = -(output_f*torch.pow(x - y, 2)).sum(2)

        if self.n_support > 1:
            scores = -(output_f*torch.pow(x - y, 2)).sum(2)
        else:
            scores = (output_f*(x * y)).sum(2)
        return scores

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())
        y_label = np.repeat(range(self.n_way), self.n_query)
        # scores = self.set_forward(x)
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
            score = (x * y).sum(2)
        return score
