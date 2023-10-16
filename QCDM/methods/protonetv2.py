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
        input_dim = params.n_query * self.n_way * self.n_support * self.n_way
        self.m = 0.1
        self.tau = 0.1
        self.nambda = 2.0
        print("tau {:4f} | inputdim {:d} | margin {:f} | nambda {:f}".format(self.tau, input_dim, self.m, self.nambda))

        out_dim = self.n_support * self.n_way
        # self.k = 10
        self.regularizer = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(inplace=True),
                nn.Linear(input_dim, out_dim))
        init_weights(self.regularizer, 'kaiming')

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
        z_s = z_support.contiguous().view(self.n_way*self.n_support, -1)
        z_q = z_query.contiguous().view(self.n_way*self.n_query, -1)

        # input = []
        sim_m = self.cos_dist(z_q, z_s)
        T1 = np.eye(int(self.n_way))
        T2 = np.ones((self.n_query, self.n_support)) # np.ones((15,5))
        label = torch.FloatTensor(np.kron(T1, T2)).cuda()
        loss_u = binary_cross_entropy(sim_m,
            label, use_sigmoid=True)

        sim_m = torch.exp(sim_m/self.tau)
        # for i in range(self.n_way*self.n_query):          
        #     for j in range(self.n_way*self.n_support):
        #         sim = nn.CosineSimilarity(dim=0)(z_q[i], z_s[j])
        #         # sim_exp = torch.exp(sim/0.1)
        #         input.append(sim)
        # input = torch.stack(input)
        input = sim_m.contiguous().view(self.n_query * self.n_way * self.n_support * self.n_way)
        # output = torch.exp(self.regularizer(input))
        output = self.regularizer(input)
        # output = torch.exp(self.regularizer(input))
        output_sum = output.contiguous().view(self.n_way, self.n_support, -1).sum(1)
        output_sum = output_sum.repeat(1,5).view(self.n_way*self.n_support)
        output = output/output_sum
        output = output.view(self.n_way, self.n_support)
            # z_support[i] = output.unsqueeze(1)*z_support[i]

        z_support   = z_support.contiguous().view(self.n_way, self.n_support, -1)      
        z_proto     = (output.unsqueeze(2)*z_support).sum(1)
        z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )
        z_proto_c = z_proto.mean(0)
        if self.training:
            dists = self.euclidean_dist_margin(z_query, z_proto, z_proto_c)
        else:
            dists = self.euclidean_dist(z_query, z_proto)
        scores = dists
        return scores, loss_u

    def set_forward_loss(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())
        y_label = np.repeat(range(self.n_way), self.n_query)
        # scores = self.set_forward(x)
        scores, loss_u = self.set_forward_adapt(x)
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_label)
        # loss = self.loss_fn(scores, y_query)
        loss = self.loss_fn(scores, y_query)+loss_u

        return float(top1_correct), len(y_label), loss, scores

    def set_forward_loss_sup(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())
        y_label = np.repeat(range(self.n_way), self.n_query)
        scores, loss_part = self.set_forward_adapt(x)
        # scores, loss_part = self.set_forward_sup(x)
        # scores = self.set_forward_adapt(x)
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_label)

        return float(top1_correct), len(y_label), self.loss_fn(scores, y_query), loss_part, scores

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

    def euclidean_dist_margin(self, x, y, c):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)
        
        # 以O为坐标系的向量
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        # 去中心化的向量
        margin_dir = x - y
        margin_norm = F.normalize(margin_dir, p = 2.0, dim = 2)
        y_c =  y - c
        # length = torch.sqrt(torch.pow(y_c, 2).sum(2))
        length = torch.sqrt(torch.pow(y, 2).sum(2))
        length = length.unsqueeze(2).expand(n, m, d)
        # v2
        T1 = np.eye(int(self.n_way))
        T2 = np.ones((self.n_query, 1))
        mask_margin = torch.FloatTensor(np.kron(T1, T2)).unsqueeze(2).expand(n, m, d).cuda()

        margin = margin_norm *(mask_margin * length)
        # margin_v1 = margin_norm * length
        x_m = x + margin*self.m
        # # v1
        # margin = margin_norm * length
        # x_m = x + margin*self.m
        # x_m = margin*self.m
        score = -(torch.pow(x_m - y, 2).sum(2))

        # score = -(torch.pow(x_m - y, 2).sum(2)+torch.pow(x_m, 2).sum(2))
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
