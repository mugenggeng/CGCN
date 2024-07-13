# -*- coding: utf-8 -*-
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from cgcn_lib.align import Align1DLayer
import torch.nn.functional as F

class TemporalMaxer(nn.Module):
    def __init__(
            self,
            kernel_size,
            stride,
            padding,
            n_embd):
        super().__init__()
        self.ds_pooling = nn.MaxPool1d(
            kernel_size, stride=stride,padding=padding)

        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
    def forward(self, x):

        # out, out_mask = self.channel_att(x, mask)

        # if self.stride > 1:
        #     # downsample the mask using nearest neighbor
        #     out_mask = F.interpolate(
        #         mask.to(x.dtype), size=x.size(-1)//self.stride, mode='nearest')
        # else:
        #     # masking out the features
        #     out_mask = mask
        # print(self.stride,self.kernel_size,self.padding,'self.stride,self.kernel_size,self.padding')
        # print(x.shape,'x4.shape========')
        #x1 = self.ds_pooling(x)
        # print(x1.shape,'x5.shape============')
        # print(out_mask.shape,'out_mask')
        out = self.ds_pooling(x) 

        return out
    


class LayerNorm(nn.Module):
    """
    LayerNorm that supports inputs of size B, C, T
    """

    def __init__(
        self,
        num_channels,
        eps=1e-5,
        affine=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs)).cuda()
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs)).cuda()
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        # normalization along C channels
        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x**2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        # apply weight and bias
        if self.affine:
            out = out * self.weight
            out = out + self.bias

        return out
    

class TConv1D(nn.Module):
    def __init__(self, channel_in, width, channel_out, kernel_size = 5,stride=1, padding=0, dilation=1, groups=1, use_bias=True):
        super(TConv1D, self).__init__()
        # self.tconvs = MaskedConv1D(channel_in,channel_out,kernel_size=kernel_size,padding = kernel_size//2)
        self.tconvs1 = nn.Conv1d(channel_in, width, kernel_size=1,padding=0)
        self.tconvs2 = nn.Conv1d(width, channel_out, kernel_size=kernel_size, groups=groups, padding=kernel_size//2,dilation=dilation)
        # self.tconvs = nn.Sequential(
        #
        #     nn.Conv1d(channel_in, width, kernel_size=1,padding=0),nn.GELU(),
        #     nn.Conv1d(width, channel_out, kernel_size=kernel_size, groups=groups, padding=kernel_size//2,dilation=dilation),
        #     # nn.Conv1d(width, channel_out, kernel_size=1),
        # )  # semantic graph
        # Generating local adaptive weights
        # self.reset_params(init_conv_vars=init_conv_vars)
        self.gelu = nn.GELU()
        self.reset_params()

        # self.idx_list = idx
    def reset_params(self):
        torch.nn.init.kaiming_normal_(self.tconvs1.weight, a=0, mode='fan_out')

        torch.nn.init.kaiming_normal_(self.tconvs2.weight, a=0, mode='fan_out')

    def forward(self, x):

        x = self.gelu(self.tconvs1(x))
        x = self.tconvs2(x)
        return x

class DConv2D(nn.Module):
    def __init__(self, channel_in, width, channel_out,kernel_size = 5 ,stride=1, padding=0, dilation=1, groups=1, use_bias=True):
        super(DConv2D, self).__init__()
        # self.dconvs = MaskedConv2D(channel_in,channel_out,kernel_size=kernel_size,padding=kernel_size//2)

        self.dcons1 = nn.Conv2d(channel_in*2, width, kernel_size=1,padding=0)
        self.dcons2 = nn.Conv2d(width, channel_out, kernel_size=kernel_size, groups=groups, padding=kernel_size//2, dilation=dilation)
        # self.dconvs = nn.Sequential(
        #     nn.Conv2d(channel_in*2, width, kernel_size=1,padding=0), nn.GELU(),
        #     nn.Conv2d(width, channel_out, kernel_size=kernel_size, groups=groups, padding=kernel_size//2, dilation=dilation),
        #     # nn.Conv2d(width, channel_out, kernel_size=1),
        # )  # semantic graph
        # Generating local adaptive weights
        # self.ffn = FFN2D(channel_out,hidden_features=width,out_features=channel_out)

        self.gelu = nn.GELU()
        self.reset_params()
        # self.idx_list = idx
    def reset_params(self):
        torch.nn.init.kaiming_normal_(self.dcons1.weight, a=0, mode='fan_out')

        torch.nn.init.kaiming_normal_(self.dcons2.weight, a=0, mode='fan_out')
    def forward(self, x):
        x = self.gelu(self.dcons1(x))
        x = self.dcons2(x)
        return x
    

def poincare_distance(u, v):
    """
    Calculates the hyperbolic distance between two vectors in the Poincare ball model.

    Args:
    - u: A torch.Tensor representing the first vector. Shape: (batch_size, embedding_dim)
    - v: A torch.Tensor representing the second vector. Shape: (batch_size, embedding_dim)

    Returns:
    - A torch.Tensor representing the distance between the two vectors. Shape: (batch_size,)
    """
    epsilon = 1e-10
    # print(u.shape,'shape')
    # print(v.shape)
    # Euclidean norm of the input vectors
    norm_u = torch.norm(u, dim=1, keepdim=True)
    norm_v = torch.norm(v, dim=1, keepdim=True)
    # print(norm_v.shape,'norm_v.shape')
    # print(norm_u.shape,'norm_u')
    # Calculate the Poincare ball radius
    radius = 1 - epsilon * epsilon

    # Calculate the magnitude of the embedded vectors in the hyperbolic space
    magnitude_u = torch.sqrt(torch.sum(u ** 2, dim=1, keepdim=True) + epsilon ** 2)
    magnitude_v = torch.sqrt(torch.sum(v ** 2, dim=1, keepdim=True) + epsilon ** 2)
    # print(magnitude_u.shape,'magnitude_u')
    # print(magnitude_v.shape,'magnitude_v')
    # Calculate the dot product of the embedded vectors in the hyperbolic space
    dot_product = torch.sum(u * v, dim=1, keepdim=True)

    # Compute the hyperbolic distance between the vectors
    cosh_distance = 1 + 2 * (torch.norm(u - v, dim=1, keepdim=True) ** 2) / (
                (1 - magnitude_u ** 2) * (1 - magnitude_v.transpose(2, 1) ** 2))
    distance = 1 / epsilon * torch.acosh(cosh_distance)
    # print(distance.shape,'distance.shape')
    return distance



def knn(x, y=None, k=10,type='dis'):
    """
    :param x: BxCxN
    :param y: BxCxM
    :param k: scalar
    :return: BxMxk
    """
    if y is None:
        y = x

    #distant = poincare_distance(x,y)
   # print(distant)
    #distant = pairwise_distances(x,y)
    xx_p = torch.sum(x, dim=1, keepdim=True)
    yy_p = torch.sum(y, dim=1, keepdim=True)

    #print(distant.shape)
    # logging.info('Size in KNN: {} - {}'.format(x.size(), y.size()))
    inner = -2 * torch.matmul(y.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    yy = torch.sum(y ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - yy.transpose(2, 1)

    _, idx_dis = pairwise_distance.topk(k=k, dim=-1)
    # print(idx_dis,'idx_dis')
    # print(idx_dis.shape, 'idx_dis')
    # k1 = torch.Tensor(1).cuda()

    distant = poincare_distance(x, y)
    # print(torch.min(distant), 'min')
    _, idx_poi = distant.topk(k=k, dim=-1)
    # print(idx_poi, 'idx_poi')
    # print(idx_poi.shape, 'idx_poi')

    x_cos = F.normalize(x, dim=1)
    y_cos = F.normalize(y, dim=1)

    cos_distant = x_cos.transpose(2, 1) @ y_cos
    _, idx_cos = cos_distant.topk(k=k, dim=-1)
    # print(idx_cos)
    if type == 'dis':
        return idx_dis
    elif type == 'cos':
        return idx_cos
    else:
        return idx_poi


class PoincareEmbedding(torch.nn.Module):
    def __init__(self, num_features, embedding_dim):
        super(PoincareEmbedding, self).__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.embedding = torch.nn.Embedding(num_features, embedding_dim)
        self.epsilon = 1e-5

    def forward(self, x):
        # Euclidean norm of the input feature vector
        norm_x = torch.norm(x, dim=1, keepdim=True)

        # Calculate the Poincare ball radius
        radius = 1 - self.epsilon * self.epsilon

        # Calculate the magnitude of the embedded vector in the hyperbolic space
        magnitude = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + self.epsilon ** 2)

        # Project the vector into the Poincare ball
        u = x / magnitude
        u = u * torch.sqrt(1 - norm_x ** 2 / radius ** 2)
        # print(u.shape,'u')
        # Compute the hyperbolic distance between the origin and the embedded vector
        # distance = 1 / self.epsilon * torch.acosh(1 + 2 * (torch.norm(u, dim=1, keepdim=True) ** 2) / (
        #             (1 - torch.norm(u, dim=1, keepdim=True) ** 2) * (1 - radius)))
        #
        # # Apply the embedding matrix
        # out = self.embedding(torch.arange(self.num_features, device=x.device))
        # out = out.view(1, self.num_features, self.embedding_dim)
        # out = out.expand(x.shape[0], self.num_features, self.embedding_dim)
        # out = torch.sum(out * u.unsqueeze(1), dim=2)

        return u

class GCNeXt(nn.Module):
    def __init__(self,channel_in,
                 n_embd,
                 channel_out,
                 n_embd_ks=3,
                 arch = (2, 4),
                 k=[7,7,7,7,7,7],
                 norm_layer=None,
                 groups=32,
                 width_group=4,
                 with_ln = False,
                 idx=None,
                 init_conv_vars=1,
                 scale_factor=2):
        super(GCNeXt, self).__init__()
        self.k = k
        self.groups = groups
        self.channel_in = channel_in
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        # width = width_group * groups
        self.tconvs = nn.ModuleList()
        self.dconvs = nn.ModuleList()
        # self.pconvs = nn.ModuleList()
        # self.cconvs = nn.ModuleList()
        self.samping = nn.ModuleList()
        # self.ffn = nn.ModuleList()
        # self.ln_before = nn.ModuleList()
        # n_embd = channel_in #改
        self.poiembed = PoincareEmbedding(n_embd,2304)
        # self.mask_conv = nn.ModuleList()
        # self.ln_after = nn.ModuleList()
        self.poi = nn.ModuleList()
        for i in range(arch[1]):
            self.tconvs.append(TConv1D(n_embd, n_embd,channel_out,groups=groups))
            self.dconvs.append(DConv2D(n_embd, n_embd,channel_out,groups=groups))
            # self.pconvs.append(PConv2D(n_embd, n_embd, channel_out, groups=groups))
            # self.cconvs.append(CConv2D(n_embd, n_embd, channel_out, groups=groups))
            # self.ffn.append(FFN1D(channel_out,hidden_features=channel_out*2,out_features=channel_out))
            # self.ln_before.append(LayerNorm(channel_out))
            self.poi.append(PoincareEmbedding(n_embd,n_embd))
            # self.mask_conv.append(nn.Conv1d(n_embd,n_embd,kernel_size=1))
            # self.ln_after.append(LayerNorm(channel_out))

        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()

        for idx in range(arch[0]):
            if idx == 0:
                in_channels = channel_in
            else:
                in_channels = n_embd
            self.embd.append(nn.Conv1d(
                in_channels, n_embd, n_embd_ks,
                stride=1, padding=n_embd_ks // 2, bias=(not with_ln)
            )
            )

            if with_ln:
                self.embd_norm.append(
                    LayerNorm(n_embd)
                )
            else:
                self.embd_norm.append(nn.Identity())

        for i in range(arch[1]):
            self.samping.append(TemporalMaxer(kernel_size=3,
                                             stride=scale_factor,
                                             padding=1,
                                             n_embd=n_embd))
        self.relu = nn.ReLU(True)

    def forward(self, x):
        #print(x.shape,'x1')
        #x =self.tconvs(x)

        # print(x.shape)
        # print('================')
        for idx in range(len(self.embd)):
            x = self.embd[idx](x)
            x = self.relu(x)
        out_feat = (x,)
       # out_mask = (mask,)
        samp_x = x
        # masks = F.interpolate(
        #         mask.to(x.dtype), size=x.size(-1)//self.stride, mode='nearest')
        for i in range(len(self.tconvs)):

            # x = self.ln_before[i](x)
            x = self.samping[i](samp_x)
            samp_x = x
            identity = x  # residual
            tout = self.tconvs[i](x)

            x_f, idx_dis = get_distant_graph_feature(x, k=self.k[i], style=0)  # (bs,ch,100) -> (bs, 2ch, 100, k)
            dout = self.dconvs[i](x_f)
           # print(tout.shape,'tout.shape')
           # print(type(tout),'tout.type')
        #    print(x.shape)
           # print(tout.shape,'tout.shape')
           # tout = self.tcn(x)x
           #  x_a = x.unsqueeze(3)
           #  poi_in,_ = self.poiembed(tout)
           #  print(tout.shape==x.shape)
            x_p = self.poi[i](x)
            x_f, idx_poi = get_po_graph_feature(x,x_p, k=self.k[i], style=0)  # (bs,ch,100) -> (bs, 2ch, 100, k)
            pout = self.dconvs[i](x_f)  # conv on semantic graph


            # x_f, idx_dis = get_cos_graph_feature(x, k=self.k[i], style=0)  # (bs,ch,100) -> (bs, 2ch, 100, k)
            # cout = self.dconvs[i](x_f)

            #print(sout.shape)
            dout = dout.max(dim=-1, keepdim=False)[0]  # (bs, ch, 100, k) -> (bs, ch, 100)
            pout = pout.max(dim=-1, keepdim=False)[0]
            x = tout  + dout  + identity +  pout  # fusion




            out_feat = out_feat + (x,)

        return reshape_feature(out_feat)
        # return out_feat,masks.bool()
def reshape_feature(base_feature):
        out_x = None
        for i in range(len(base_feature)):
            if out_x is None:
                out_x = base_feature[i]
            else:
                # 假设 out_x 和 base_feature[i] 的第 3 个维度长度分别为 T_out 和 T_base，则
                T_out = out_x.shape[2]
                T_base = base_feature[i].shape[2]

                # 首先计算需要在左侧和右侧各填充多少列
                left_pad = (T_out - T_base) // 2
                right_pad = T_out - T_base - left_pad

                # 使用更新后的左、右填充长度进行填充操作（注意第 1 个维度已经被忽略）
                base_feature_after = F.pad(base_feature[i], [left_pad, right_pad], value=0)


                out_x = out_x + base_feature_after

        norm = LayerNorm(out_x.shape[1])
        return out_x.cuda()
def get_po_graph_feature(x,x_p,prev_x=None, k=20, idx_poi_knn=None, r=-1, style=1):
    """
    :param x:
    :param prev_x:
    :param k:
    :param idx:
    :param r: output downsampling factor (-1 for no downsampling)
    :param style: method to get graph feature
    :return:
    """
    batch_size = x.size(0)
    num_points = x.size(2)  # if prev_x is None else prev_x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx_poi_knn is None:
        idx_poi_knn = knn(x=x_p, y=prev_x, k=k,type='poi')  # (batch_size, num_points, k)
    else:
        k = idx_poi_knn.shape[-1]
    # print(idx_knn.shape)
    device = x.device  # torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx_poi_knn + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    if style == 0:  # use offset as feature
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    elif style == 1:  # use feature as feature
        feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
    else: # style == 2:
        feature = feature.permute(0,3,1,2)
    # downsample if needed
    if r != -1:
        select_idx = torch.from_numpy(np.random.choice(feature.size(2), feature.size(2) // r,
                                                       replace=False)).to(device=device)
        feature = feature[:, :, select_idx, :]
    return feature, idx_poi_knn

def get_distant_graph_feature(x, prev_x=None, k=20, idx_dis_knn=None, r=-1, style=1):
    """
    :param x:
    :param prev_x:
    :param k:
    :param idx:
    :param r: output downsampling factor (-1 for no downsampling)
    :param style: method to get graph feature
    :return:
    """
    batch_size = x.size(0)
    num_points = x.size(2)  # if prev_x is None else prev_x.size(2)
    x = x.view(batch_size, -1, num_points)

    if idx_dis_knn is None:
        idx_dis_knn = knn(x=x, y=prev_x, k=k,type='dis')  # (batch_size, num_points, k)
    else:
        k = idx_dis_knn.shape[-1]
    # print(idx_knn.shape)
    device = x.device  # torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx_dis_knn + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    if style == 0:  # use offset as feature
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    elif style == 1:  # use feature as feature
        feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
    else: # style == 2:
        feature = feature.permute(0,3,1,2)
    # downsample if needed
    if r != -1:
        select_idx = torch.from_numpy(np.random.choice(feature.size(2), feature.size(2) // r,
                                                       replace=False)).to(device=device)
        feature = feature[:, :, select_idx, :]
    return feature, idx_dis_knn


# get graph feature
def get_graph_feature(x, prev_x=None, k=20, idx_knn=None, r=-1, style=0):
    """
    :param x:
    :param prev_x:
    :param k:
    :param idx:
    :param r: output downsampling factor (-1 for no downsampling)
    :param style: method to get graph feature
    :return:
    """
    batch_size = x.size(0)
    num_points = x.size(2)  # if prev_x is None else prev_x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx_knn is None:
        idx_knn = knn(x=x, y=prev_x, k=k)  # (batch_size, num_points, k)
    else:
        k = idx_knn.shape[-1]
    # print(idx_knn.shape)
    device = x.device  # torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx_knn + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    if style == 0:  # use offset as feature
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    elif style == 1:  # use feature as feature
        feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
    else: # style == 2:
        feature = feature.permute(0,3,1,2)
    # downsample if needed
    if r != -1:
        select_idx = torch.from_numpy(np.random.choice(feature.size(2), feature.size(2) // r,
                                                       replace=False)).to(device=device)
        feature = feature[:, :, select_idx, :]
    return feature, idx_knn


# 


class GraphAlign(nn.Module):
    def __init__(self, k=3, t=100, d=100, bs=64, samp=0, style=0):
        super(GraphAlign, self).__init__()
        self.k = k
        self.t = t
        self.d = d
        self.bs = bs
        self.style = style
        self.expand_ratio = 0.5
        self.resolution = 32
        self.align_inner = Align1DLayer(self.resolution, samp)
        self.align_context = Align1DLayer(4)
        self._get_anchors()

    def forward(self, x, index):
        bs, ch, t = x.shape
        if not self.anchors.is_cuda:  # run once
            self.anchors = self.anchors.cuda()

        anchor = self.anchors[:self.anchor_num * bs, :]  # (bs*tscale*tscal, 3)
        # print('first value in anchor is', anchor[0])
        feat_inner = self.align_inner(x, anchor)  # (bs*tscale*tscal, ch, resolution)
        if self.style == 1: # use last layer neighbours
            feat, _ = get_graph_feature(x, k=self.k, style=2)  # (bs,ch,100) -> (bs, ch, 100, k)
            feat = feat.mean(dim=-1, keepdim=False)  # (bs. 2*ch, 100)
            feat_context = self.align_context(feat, anchor)  # (bs*tscale*tscal, ch, resolution//2)
            feat = torch.cat((feat_inner,feat_context), dim=2).view(bs, t, t, -1)
        elif self.style == 2: # use all layers neighbour
            feat, _ = get_graph_feature(x, k=self.k, style=2, idx_knn=index)  # (bs,ch,100) -> (bs, ch, 100, k)
            feat = feat.mean(dim=-1, keepdim=False)  # (bs. 2*ch, 100)
            feat_context = self.align_context(feat, anchor)  # (bs*tscale*tscal, ch, resolution//2)
            feat = torch.cat((feat_inner,feat_context), dim=2).view(bs, t, t, -1)
        else:
            feat = torch.cat((feat_inner,), dim=2).view(bs, t, t, -1)
        # print('shape after align is', feat_context.shape)

        return feat.permute(0, 3, 2, 1)  # (bs,2*ch*(-1),t,t)

    def _get_anchors(self):
        anchors = []
        for k in range(self.bs):
            for start_index in range(self.t):
                for duration_index in range(self.d):
                    if start_index + duration_index < self.t:
                        p_xmin = start_index
                        p_xmax = start_index + duration_index
                        center_len = float(p_xmax - p_xmin) + 1
                        sample_xmin = p_xmin - center_len * self.expand_ratio
                        sample_xmax = p_xmax + center_len * self.expand_ratio
                        anchors.append([k, sample_xmin, sample_xmax])
                    else:
                        anchors.append([k, 0, 0])
        self.anchor_num = len(anchors) // self.bs
        self.anchors = torch.tensor(np.stack(anchors)).float()  # save to cpu
        return  # anchors, anchor_num

def hook(module, input, output):
    outputs.append(output)

class CGCN(nn.Module):
    def __init__(self, opt):
        super(CGCN, self).__init__()
        self.tscale = opt["temporal_scale"]
        self.feat_dim = opt["feat_dim"]
        self.bs = opt["batch_size"]
        self.h_dim_1d = opt["h_dim_1d"]
        self.h_dim_2d = opt["h_dim_2d"]
        self.h_dim_3d = opt["h_dim_3d"]
        self.goi_style = opt['goi_style']
        self.h_dim_goi = self.h_dim_1d*(32,32+4,32+4)[opt['goi_style']]
        self.idx_list = []
        self.features_dim = (100,1)
        
        # def hook(self,module, input, output):
        #     outputs.append(output)

        self.relu = nn.ReLU(inplace=True)
        # self.loc_hook.layer4[0].conv2.register_forward_hook(hook)
        # Backbone Part 1
        self.backbone1 = nn.Sequential(
            GCNeXt(self.feat_dim, self.h_dim_1d,self.h_dim_1d, groups=32, idx=self.idx_list),
        )

        # Regularization
        self.regu_s = nn.Sequential(
            GCNeXt(self.h_dim_1d, self.h_dim_1d,self.h_dim_1d, groups=32),
            nn.Conv1d(self.h_dim_1d, 1, kernel_size=1), nn.Sigmoid()
        )
        self.regu_e = nn.Sequential(
            GCNeXt(self.h_dim_1d, self.h_dim_1d,self.h_dim_1d,groups=32),
            nn.Conv1d(self.h_dim_1d, 1, kernel_size=1), nn.Sigmoid()
        )

        # Backbone Part 2
        self.backbone2 = nn.Sequential(
            GCNeXt(self.h_dim_1d, self.h_dim_1d,self.h_dim_1d, groups=32, idx=self.idx_list),
        )#改

        # SGAlign: sub-graph of interest alignment
        self.goi_align = GraphAlign(
            t=self.tscale, d=opt['max_duration'], bs=self.bs,
            samp=opt['goi_samp'], style=opt['goi_style']  # for ablation
        )

        # Localization Module
        self.localization = nn.Sequential(
            nn.Conv2d(self.h_dim_goi, self.h_dim_3d, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(self.h_dim_3d, self.h_dim_2d, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(self.h_dim_2d, self.h_dim_2d, kernel_size=opt['kern_2d'], padding=opt['pad_2d']), nn.ReLU(inplace=True),
            nn.Conv2d(self.h_dim_2d, self.h_dim_2d, kernel_size=opt['kern_2d'], padding=opt['pad_2d']), nn.ReLU(inplace=True),
            nn.Conv2d(self.h_dim_2d, 2, kernel_size=1), nn.Sigmoid()
        )

        # self.localization_hook = self.localization[3].register_forward_hook(hook)

        # Position encoding (not useful)
        # self.pos = torch.arange(0, 1, 1.0 / self.tscale).view(1, 1, self.tscale)
        # self.pos = PositionalEncoding(self.feat_dim, dropout=0.1, max_len=self.tscale)

    def forward(self, snip_feature):
        del self.idx_list[:]  # clean the idx list
        # snip_feature = self.pos(snip_feature)
        base_feature = self.backbone1(snip_feature).contiguous()  # (bs, 2048, 256) -> (bs, 256, 256)
        gcnext_feature = self.backbone2(base_feature)  #

        regu_s = self.regu_s(base_feature).squeeze(1)  # start
        regu_e = self.regu_e(base_feature).squeeze(1)  # end

        if self.goi_style==2:
            idx_list = [idx for idx in self.idx_list if idx.device == snip_feature.device]
            idx_list = torch.cat(idx_list, dim=2)
        else:
            idx_list = None

        subgraph_map = self.goi_align(gcnext_feature, idx_list)
        # print(base_feature.shape,gcnext_feature.shape,regu_s.shape,regu_e.shape,subgraph_map.shape)

        self.bottleneck_dim = subgraph_map.size(1)
        iou_map = self.localization(subgraph_map)
        return iou_map, regu_s, regu_e
    ## added for fs setting
    def extract_features(self,snip_feature):
        base_feature = self.backbone1(snip_feature).contiguous()  # (bs, 2048, 256) -> (bs, 256, 256) # i/p : [batch, temp, feat] --> int feat [batch,cha,temp',feat']
        # print("base feat dim", base_feature.size())
        gcnext_feature = self.backbone2(base_feature)  #
        # print("gcnext feat dim", gcnext_feature.size())
        regu_s = self.regu_s(base_feature).squeeze(1)  # start
        regu_e = self.regu_e(base_feature).squeeze(1)  # end

        if self.goi_style==2:
            idx_list = [idx for idx in self.idx_list if idx.device == snip_feature.device]
            idx_list = torch.cat(idx_list, dim=2)
        else:
            idx_list = None

        subgraph_map = self.goi_align(gcnext_feature, idx_list)
        # print("subgraph feat dim", subgraph_map.size())
        subgraph_map = self.relu(subgraph_map)
        # iou_map = self.localization(subgraph_map)
        ext_feat = gcnext_feature.unsqueeze(3)#改
       # ext_feat = self.relu(gcnext_feature.unsqueeze(3))
        # print(gcnext_feature.size())
        # print(subgraph_map.size())
        return ext_feat


if __name__ == '__main__':
    from cgcn_lib import opts
    from torchsummary import summary

    parser = argparse.ArgumentParser()
    opt = opts.parse_opt(parser)
    opt = vars(opt)
    model = CGCN(opt).cuda()
    # model = torch.nn.DataParallel(model, device_ids=[0])
    # input = torch.randn(4, 400, 100).cuda()
    # a, b, c = model(input)
    # print(a.shape, b.shape, c.shape)

    summary(model, (400,100))

    '''
    Total params: 9,495,428
    Trainable params: 9,495,428
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.15
    Forward/backward pass size (MB): 1398.48
    Params size (MB): 36.22
    Estimated Total Size (MB): 1434.85
    '''
