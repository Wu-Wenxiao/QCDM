import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .template import MetaTemplate
from .bdc_module import BDC
from init import init_weights

class MetaDeepBDC(MetaTemplate):
    def __init__(self, params, model_func, n_way, n_support):
        super(MetaDeepBDC, self).__init__(params, model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        input_dim = self.n_support * self.n_way
        self.tau = 0.0
        self.k = 1.0
        self.m = 0.02
        print("v3: layer_norm+feature_dim-adapt")
        print("tau {:4f} | inputdim {:d} | k {:4f} | margin {:4f}".format(self.tau, input_dim, self.k, self.m))
        self.regularizer = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(inplace=True),
                nn.Linear(input_dim, 1))
        input_dim = 75
        self.regularizer_ins = nn.Sequential(
                nn.Linear(input_dim, input_dim),    
                nn.ReLU(inplace=True),
                nn.Linear(input_dim, self.n_support*self.n_way))
        init_weights(self.regularizer_ins, 'kaiming')
        reduce_dim = params.reduce_dim
        self.feat_dim = int(reduce_dim * (reduce_dim+1) / 2)
        self.dcov = BDC(is_vec=True, input_dim=self.feature.feat_dim, dimension_reduction=reduce_dim)

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
        n = z_query.size(0)
        m = z_support.size(0)
        d = z_query.size(1)
        assert d == z_support.size(1)

        # 75*25*640
        z_q = z_query.unsqueeze(1).expand(n, m, d)
        z_s = z_support.unsqueeze(0).expand(n, m, d) 
            # z_qs = z_q - z_s

        z_qs = (z_q - z_s).contiguous().view(self.n_way*self.n_query,self.n_way,self.n_support,-1)
        
        T1 = torch.repeat_interleave(z_qs, 5, dim=1)
        T1 = T1.contiguous().view(self.n_way*self.n_query,self.n_way,self.n_support*self.n_support,-1)

        tem = torch.triu(torch.ones(self.n_support, self.n_support), diagonal=0)
        tem = tem.contiguous().view(-1)
        index = torch.nonzero(tem).T[0]

        T = T1[:,:,index,:]

        T2 = torch.repeat_interleave(z_qs, torch.tensor([5,4,3,2,1]).cuda(), dim=2)

        sim = (T*T2).sum(3)
        input = sim.contiguous().view(self.n_way*self.n_query, -1)

        for i in range(self.n_way*self.n_query):
            if i == 0 :
                output = self.regularizer_ins(input[i]).unsqueeze(0)
            else:
                output = torch.cat((output, self.regularizer_ins(input[i]).unsqueeze(0)),dim = 0)

        # output_sum = output.contiguous().view(self.n_way*self.n_query, self.n_way, self.n_support, -1).sum(2)
        # output_sum = output_sum.repeat(1,1,5).view(self.n_way*self.n_query, self.n_way*self.n_support)
        # output = output/output_sum
        output = output.view(self.n_way*self.n_query, self.n_way, self.n_support, -1)
        # layer_norm = torch.nn.LayerNorm([640], elementwise_affine=False)
        # z_support = layer_norm(z_support)
        # z_query = layer_norm(z_query)
        z_s = z_s.contiguous().view(self.n_way*self.n_query,self.n_way,self.n_support,-1)
        z_s_weight = output*z_s
        z_proto     = z_s_weight.mean(2)
        # z_proto     = (output*z_s).sum(2)
        n = z_query.size(0)
        m = z_proto.size(1)
        d = z_query.size(1)
        assert d == z_proto.size(2)

        x = z_query.unsqueeze(1).expand(n, m, d)
        y = z_proto

        # 计算output_f
        z_s_weight = z_s_weight.contiguous().view(self.n_way*self.n_query,self.n_support*self.n_way,-1)
        sim_euc = torch.pow(z_q-z_s_weight, 2)
        input_c = sim_euc
        for i in range(self.n_way*self.n_query):
            if i == 0 :
                output_c = self.regularizer(input_c[i].T).T
            else:
                output_c = torch.cat((output_c, self.regularizer(input_c[i].T).T),dim = 0)

        output_f = output_c.unsqueeze(1).expand(n, m, d)
        scores = -(output_f*torch.pow(x - y, 2)).sum(2)
        # scores = -(torch.pow(x - y, 2)).sum(2)
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
