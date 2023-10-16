import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .template import MetaTemplate, channel_weight
from init import init_weights
from .extraloss import binary_cross_entropy


class ProtoNet(MetaTemplate):
    def __init__(self, params, model_func, n_way, n_support):
        super(ProtoNet, self).__init__(params, model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        input_dim = (self.n_support) * self.n_way
        self.tau = 0.0
        self.k = 0.0
        self.m = 0.0
        print("v3max: layer_norm+feature_dim-adapt+init_weight")
        print("tau {:4f} | inputdim {:d} | k {:4f} | margin {:4f}".format(self.tau, input_dim, self.k, self.m))

        # out_dim = input_dim
        # self.k = 10
        self.regularizer = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(inplace=True),
                nn.Linear(input_dim, 1))
        init_weights(self.regularizer, 'kaiming')
        self.init_weight = channel_weight(num_query=self.n_way*self.n_query, dim=self.feature.feat_dim[0])
        self.init_weight.initialise()

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

        # n = z_query.size(0)
        # m = z_proto.size(0)
        # d = z_query.size(1)
        # assert d == z_proto.size(1)
        # z_q = z_query.unsqueeze(1).expand(n, m, d)
        # z_p = z_proto.unsqueeze(0).expand(n, m, d)

        # sim_proto = torch.pow(z_q - z_p, 2)

        input = sim_euc
        # input = torch.cat((sim_euc, sim_proto),dim=1)
        # print(input.shape)
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

        assert z_support.shape[-1]==self.init_weight.weight.shape[-1]
        
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
        
        # print(output.shape)
        # print(self.init_weight.weight.shape)
        assert output.shape==self.init_weight.weight.shape
        # output_f = output.unsqueeze(1).expand(n, m, d)
        output_f = (output*self.init_weight.weight).unsqueeze(1).expand(n, m, d)

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
        scores = -(output_f*self.cos_dist(x, y)).sum(2)
        # scores = -(output_f*torch.pow(x - y, 2)).sum(2)
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

    # def cos_dist(self, x, y):
    #     # x: N x D
    #     # y: M x D
    #     n = x.size(0)
    #     m = y.size(0)
    #     d = x.size(1)
    #     assert d == y.size(1)

    #     x = x.unsqueeze(1).expand(n, m, d)
    #     y = y.unsqueeze(0).expand(n, m, d)

    #     cos_sim = nn.CosineSimilarity(dim = 2)(x, y)

    #     return cos_sim

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
