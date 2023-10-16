import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .template import MetaTemplate
from init import init_weights
from .extraloss import binary_cross_entropy


class ProtoNet(MetaTemplate):
    def __init__(self, params, model_func, n_way, n_support):
        super(ProtoNet, self).__init__(params, model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # input_dim = (self.n_support) * self.n_way
        self.tau = 0.0
        self.k = 0.0
        self.m = 0.0
        print("anova: layer_norm+feature_dim-anova")
        # print("tau {:4f} | inputdim {:d} | k {:4f} | margin {:4f}".format(self.tau, input_dim, self.k, self.m))

        # out_dim = input_dim
        # self.k = 10
        # self.regularizer = nn.Sequential(
        #         nn.Linear(input_dim, input_dim),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(input_dim, 1))
        # init_weights(self.regularizer, 'kaiming')

    def feature_forward(self, x):
        out = self.avgpool(x).view(x.size(0),-1)
        return out

    def set_forward(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        scores = self.euclidean_dist(z_query, z_proto)
        return scores

    def set_forward_adapt(self,x,is_feature = False):
        z_support, z_query  = self.parse_feature(x,is_feature)
        z_support = z_support.contiguous().view(self.n_way*self.n_support, -1)
        z_query = z_query.contiguous().view(self.n_way*self.n_query, -1)

        # layer_norm = torch.nn.LayerNorm([z_support.shape[-1]], elementwise_affine=False)
        # z_support = layer_norm(z_support)
        # z_query = layer_norm(z_query)
        # z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)

        n = z_query.size(0)
        m = z_support.size(0)
        d = z_query.size(1)
        assert d == z_support.size(1)

        z_q = z_query.unsqueeze(1).expand(n, m, d)
        z_s = z_support.unsqueeze(0).expand(n, m, d)

        sim_euc = torch.pow(z_q - z_s, 2)
        input = sim_euc

        # F = []
        # for i in range(self.n_way*self.n_query):
        #     data = input[i]
        #     F_temp = []
        #     for j in range(d):
        #         a = data.T[j]
        #         a = a.contiguous().view(self.n_support, self.n_way)
        #         F_a, _ = F_SST(a.T)
        #         F_temp.append(F_a)
        #     F.append(torch.stack(F_temp))

        # F_final = torch.vstack(F)
        F_=[]
        for i in range(self.n_way*self.n_query):
            data = input[i] # 25 * 640
            data = data.T.contiguous().view(d, self.n_way, self.n_support).transpose(1, 2)
            F_a, _ = F_SST_(data)
            F_.append(F_a)                  
        
        F_final = torch.vstack(F_)
        # var_v = torch.var(sim_euc, dim=1)

        # n = z_query.size(0)
        # m = z_proto.size(0)
        # d = z_query.size(1)
        # assert d == z_proto.size(1)
        # z_q = z_query.unsqueeze(1).expand(n, m, d)
        # z_p = z_proto.unsqueeze(0).expand(n, m, d)

        # sim_proto = torch.pow(z_q - z_p, 2)

        
        # input = torch.cat((sim_euc, sim_proto),dim=1)
        # print(input.shape)
        # max_col = torch.max(sim_euc, 1)[0]
        # max_col = max_col.unsqueeze(1).repeat(1,self.n_way*self.n_support,1)
        # input = sim_euc/max_col
        # input = torch.exp(input/self.tau)
        # sim_exp = torch.exp(sim_euc/self.tau)
        # for i in range(self.n_way*self.n_query):
        #     if i == 0 :
        #         output = self.regularizer(input[i].T).T
        #     else:
        #         output = torch.cat((output, self.regularizer(input[i].T).T),dim = 0)

        layer_norm = torch.nn.LayerNorm([z_support.shape[-1]], elementwise_affine=False)
        z_support = layer_norm(z_support)
        z_query = layer_norm(z_query)


        z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        n = z_query.size(0)
        m = z_proto.size(0)
        d = z_query.size(1)
        assert d == z_proto.size(1)

        x = z_query.unsqueeze(1).expand(n, m, d)
        y = z_proto.unsqueeze(0).expand(n, m, d)
        
        output_f = F_final.unsqueeze(1).expand(n, m, d)

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

        scores = -(output_f*torch.pow(x - y, 2)).sum(2)
        return scores


    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())
        y_label = np.repeat(range(self.n_way), self.n_query)
        # scores = self.set_forward(x)
        # scores, loss_u, y_label = self.set_forward_adapt(x)
        scores = self.set_forward_adapt(x)
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_label)
        # loss = self.loss_fn(scores, y_query)
        loss = self.loss_fn(scores, y_query)

        return float(top1_correct), len(y_label), loss, scores

    def euclidean_dist(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        score = -torch.pow(x - y, 2).sum(2)
        return score

    def cos_dist(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        cos_sim = nn.CosineSimilarity(dim = 2)(x, y)

        return cos_sim


def SST(Y):
    sst = (torch.pow(Y-torch.mean(Y), 2)).sum()
    return sst

def SST_(Y):
    d, n, k = Y.shape
    Y_mean_d = Y.mean(1).mean(1)
    # assert Y[0].mean()==Y_mean_d[0]
    sst_ = (torch.pow(Y-Y_mean_d.unsqueeze(1).unsqueeze(2).expand(d, n, k), 2)).sum(1).sum(1)
    return sst_

def SSE(Y):
    row, col = Y.shape
    Y_mean_col = torch.mean(Y, dim=0)
    Y_mean_col_exp = Y_mean_col.expand(row, col)
    sse = (torch.pow(Y-Y_mean_col_exp, 2)).sum()
    return sse

def SSE_(Y):
    d, ni, k = Y.shape
    Y_mean_col = torch.mean(Y, dim=1)
    Y_mean_col_exp = Y_mean_col.unsqueeze(1).expand(d, ni, k)
    sse = torch.pow(Y-Y_mean_col_exp, 2).sum(1).sum(1)
    return sse

def SSA(Y):
    row, col = Y.shape
    Y_mean_col = torch.mean(Y, dim=0)
    Y_mean = torch.mean(Y)
    ssa = row*(torch.pow(Y_mean_col-Y_mean, 2).sum())
    return ssa

def SSA_(Y):
    d, ni, k = Y.shape
    Y_mean_col = torch.mean(Y, dim=1)
    Y_mean_d = Y.mean(1).mean(1).unsqueeze(1).expand(d, k)
    ssa = (ni*torch.pow(Y_mean_col-Y_mean_d, 2).sum(1))
    return ssa

def F_SST(Y):
    ni, k = Y.shape
    sst = SSE(Y) + SSA(Y)
    msa = SSA(Y)/(k-1)
    mse = SSE(Y)/(ni*k - k)
    F = msa/mse
    return F, sst

def F_SST_(Y):
    d, ni, k = Y.shape
    sst_ = SST_(Y)
    msa_ = SSA_(Y)/(k-1)
    mse_ = SSE_(Y)/(ni*k - k)
    F_ = msa_/mse_
    return F_, sst_