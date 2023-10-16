import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from .template import MetaTemplate
import ipdb

EPS=0.00001
class ProtoNet(MetaTemplate):
    def __init__(self, params, model_func, n_way, n_support):
        super(ProtoNet, self).__init__(params, model_func, n_way, n_support)
        self.loss_fn = nn.CrossEntropyLoss()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        print("pure protonet")
        # self.k = params.n_query-1
        # self.nambda = 0.2
        # print("k {:d} | nambda {:f}".format(self.k, self.nambda))
        # input_dim = 16 * self.n_way * self.n_support * self.n_way

        # out_dim = self.n_support * self.n_way
        # # self.k = 10
        # self.regularizer = nn.Sequential(
        #         nn.Linear(input_dim, input_dim),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(input_dim, out_dim))

    def feature_forward(self, x):
        out = self.avgpool(x).view(x.size(0),-1)
        return out

    def set_forward_adapt(self, x, is_feature=False):
        # print(1)
        z_support, z_query = self.parse_feature(x, is_feature)
        z_proto = torch.relu(z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1))
        z_query = torch.relu(z_query.contiguous().view(self.n_way * self.n_query, -1))


        z_support = z_support.contiguous().view(self.n_way, self.n_support, -1)
        # ipdb.set_trace()
        z_support_1 = z_support[0]
        z_support_2 = z_support[1]

        n = z_query.size(0) # 30
        m = z_support_1.size(0) # 5
        d = z_query.size(1) # 640
        assert d == z_support_1.size(1)

        z_q = z_query.unsqueeze(1).expand(n, m, d)
        z_s_1 = z_support_1.unsqueeze(0).expand(n, m, d)
        z_s_2 = z_support_2.unsqueeze(0).expand(n, m, d)

        sim_euc_1 = torch.pow(z_q - z_s_1, 2)
        sim_euc_2 = torch.pow(z_q - z_s_2, 2)
        mean_1 = torch.mean(z_s_1, dim=1)
        mean_2 = torch.mean(z_s_2, dim=1)
        var_1 = torch.var(sim_euc_1, 1)
        var_2 = torch.var(sim_euc_2, 1)

        w = torch.abs(mean_1-mean_2)/(var_1+var_2)

        n = z_query.size(0)
        m = z_proto.size(0)
        d = z_query.size(1)
        assert d == z_proto.size(1)

        x = z_query.unsqueeze(1).expand(n, m, d)
        y = z_proto.unsqueeze(0).expand(n, m, d)
        
        output_f = w.unsqueeze(1).expand(n, m, d)
        # scores = -(output_f*torch.pow(x - y, 2)).sum(2)
        scores = -(torch.pow(x - y, 2)).sum(2)

        # ipdb.set_trace()
        # mask_p = torch.where(z_proto>0, 1, 0)
        # mask_q = torch.where(z_query>0, 1, 0)
        # z_proto_st = mask_p*(1/((torch.log(1/(z_proto+1e-5)+1))**1.3)).cuda()
        # z_query_st = mask_q*(1/((torch.log(1/(z_query+1e-5)+1))**1.3)).cuda()
        # print(1)
        # scores = self.euclidean_dist(z_query_st, z_proto_st)
        # print(scores)
        # scores = self.euclidean_dist(z_query, z_proto)
        # print(1)
        # scores = self.cos_dist(z_query, z_proto)
        return scores

    def set_forward_each(self, x_s, x_q, indice, is_feature=False):
        scores=[]
        for i in range(x_q.shape[0]):
            x_support = x_s[indice[i]].contiguous().view(-1, *x_s.size()[2:])
            x = torch.cat((x_support, x_q[i].unsqueeze(0)), dim=0)
            z_support, z_query = self.parse_feature_each(x, is_feature)
            z_proto = z_support.contiguous().view(indice.shape[-1], self.n_support, -1).mean(1)
            score = self.euclidean_dist(z_query.unsqueeze(0), z_proto)
            scores.append(score)
        # print(1)
        # z_support, z_query = self.parse_feature(x, is_feature)
        # z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        # z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)
        # print(1)
        # scores = self.euclidean_dist(z_query, z_proto)
        # print(1)
        # scores = self.cos_dist(z_query, z_proto)
        scores = torch.vstack(scores)
        return scores

    def set_forward_sup(self, x, is_feature=False):
        z_support, z_query = self.parse_feature(x, is_feature)
        z_proto = z_support.contiguous().view(self.n_way, self.n_support, -1).mean(1)
        z_query = z_query.contiguous().view(self.n_way * self.n_query, -1)

        z_batch = z_query
        loss_part = self.cssf_loss_kNN(z_batch, self.n_query, self.n_way, self.k)

        # scores = self.euclidean_dist(z_query, z_proto)
        # print(1)
        scores = self.cos_dist(z_query, z_proto)
        return scores, loss_part

    # def set_forward_adapt(self,x,is_feature = False):
    #     z_support, z_query  = self.parse_feature(x,is_feature)
    #     z_s = z_support.contiguous().view(self.n_way*self.n_support, -1)
    #     z_q = z_query.contiguous().view(self.n_way*self.n_query, -1)

    #     # input = []
    #     sim_m = self.cos_dist(z_q, z_s)
    #     sim_m = torch.exp(sim_m/0.1)
    #     # for i in range(self.n_way*self.n_query):          
    #     #     for j in range(self.n_way*self.n_support):
    #     #         sim = nn.CosineSimilarity(dim=0)(z_q[i], z_s[j])
    #     #         # sim_exp = torch.exp(sim/0.1)
    #     #         input.append(sim)
    #     # input = torch.stack(input)
    #     input = sim_m.contiguous().view(self.n_query * self.n_way * self.n_support * self.n_way)
    #     # output = torch.exp(self.regularizer(input))
    #     output = self.regularizer(input)
    #     output_sum = output.contiguous().view(self.n_way, self.n_support, -1).sum(1)
    #     output_sum = output_sum.repeat(1,5).view(self.n_way*self.n_support)
    #     output = output/output_sum
    #     output = output.view(self.n_way, self.n_support)
    #         # z_support[i] = output.unsqueeze(1)*z_support[i]

    #     z_support   = z_support.contiguous().view(self.n_way, self.n_support, -1)      
    #     z_proto     = (output.unsqueeze(2)*z_support).sum(1)
    #     z_query     = z_query.contiguous().view(self.n_way* self.n_query, -1 )

    #     dists = self.euclidean_dist(z_query, z_proto)
    #     scores = dists
    #     return scores

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

    def set_forward_loss_sup(self, x):
        y_query = torch.from_numpy(np.repeat(range(self.n_way), self.n_query))
        y_query = Variable(y_query.cuda())
        y_label = np.repeat(range(self.n_way), self.n_query)
        scores, loss_part = self.set_forward_sup(x)
        # scores = self.set_forward_adapt(x)
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:, 0] == y_label)

        return float(top1_correct), len(y_label), self.loss_fn(scores, y_query), loss_part, scores

    def cssf_loss_kNN(self, z, shots_per_way, n_way, k): # shots_per_way=5 n_ul=128
    # labelled positives and all negatives
        n_pos = shots_per_way
        n_l = n_way * shots_per_way # 5 * 15
        # positive mask
        T1 = np.eye(int(n_l/n_pos)) # np.eye(5)
        T2 = np.ones((n_pos, n_pos)) # np.ones((15,15))
        mask_pos_lab = torch.FloatTensor(np.kron(T1, T2))# 矩阵克罗内克积，第一个矩阵每个元素与第二个矩阵整体相乘 25 * 25

        mask_pos = mask_pos_lab.cuda() # 148 * 148
        # negative mask
        T1 = 1-np.eye(n_way) # 5 * 5
        T2 = np.ones((shots_per_way, shots_per_way)) # 15 * 15
        mask_neg_lab = torch.FloatTensor(np.kron(T1, T2)) # 75 * 75 
        mask_neg = mask_neg_lab.cuda() # 75 * 75

        return self.kNN_contrastive_loss(z, mask_pos, mask_neg, n_way*k, k)

    def kNN_contrastive_loss(self, z, mask_pos, mask_neg, n_s, k):
        # equally weighted task and distractor negative contrastive loss
        bsz, featdim = z.size()
        z_square = z.view(bsz, 1, featdim).repeat(1, bsz, 1)
        sim = nn.CosineSimilarity(dim=2)(z_square, z_square.transpose(1, 0))
        # sim_euc = -torch.pow(z_square -  z_square.transpose(1, 0), 2).sum(2)
        Sv = torch.exp(sim/0.1)
        pos = mask_pos * Sv
        pos.fill_diagonal_(0)
        _, index = pos.topk(k, 1, True, True)#已搜寻到kNN最近邻索引
        # kNN_sim = mask_pos*torch.exp(sim_euc)
        # _, index = kNN_sim.fill_diagonal_(0).topk(3, 1, True, True)#已搜寻到kNN最近邻索引 使用欧式距离
        zeros_kNN = torch.zeros(pos.shape).cuda()
        mask_kNN = zeros_kNN.scatter_(1, index, 1)# 仅仅给最近邻置1
        # print(mask_kNN)
        neg = (Sv * mask_neg)
        neg = neg.sum(dim=1).unsqueeze(1).repeat(1, bsz)
        li = mask_kNN * torch.log(Sv / (Sv + neg) + EPS) # mask_pos去除除了正样本其他位置的矩阵
        li = li - li.diag().diag()# 去除自身与自身计算相似性
        li = (1 / (n_s - 1)) * li.sum(dim=1)
        loss = -li[mask_pos.sum(dim=1) > 0].mean()
        return loss

    def euclidean_dist(self, x, y):
        # x: N x D
        # y: M x D
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        assert d == y.size(1)

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        # weight=torch.ones(d).cuda()
        # indice=[173, 180, 135, 578, 385, 498, 406, 474,  18, 528, 332, 143, 164, 564,
        # 285,  79, 348, 497, 572, 587, 289,  50, 337, 326, 483, 276, 230, 454,
        # 219, 444, 494, 435, 432, 389,  23, 232, 519, 570, 300, 215, 425,  94,
        # 317, 506,  75,  41, 379, 261, 286, 508]
        # weight[indice]=1.1
        # weight=(weight.unsqueeze(0).unsqueeze(0)).expand(n,m,d)
        # score = -(weight*torch.pow(x - y, 2)).sum(2)

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
