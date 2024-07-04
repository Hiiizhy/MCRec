import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import activation_getter


class Model(nn.Module):
    def __init__(self, config, numItems):
        super(Model, self).__init__()
        self.config = config
        self.dim = self.config.dim
        self.nitems = numItems
        self.itemEmb = nn.Embedding(numItems + 1, self.dim, padding_idx=config.padIdx)
        self.gate_his = nn.Linear(numItems, 1)
        self.Lh = self.config.Lh
        self.Lv = self.config.Lv
        self.ac_conv = activation_getter[self.config.ac_conv]
        self.ac_fc = activation_getter[self.config.ac_fc]
        self.dropout = nn.Dropout(self.config.drop)

        self.gate_his = nn.Linear(self.dim, 1)
        self.gate_trans = nn.Linear(self.dim, 1)

        self.dropout = nn.Dropout(0.5)

        self.out = nn.Linear(8 * 4, self.nitems)
        self.his_embds = nn.Linear(self.nitems, self.dim)
        self.g1 = nn.Linear(self.dim, 1)
        self.g2 = nn.Linear(8 * 4, 1)
        self.g21 = nn.Linear(self.dim, 1)
        self.g22 = nn.Linear(self.dim, 1)

    def forward(self, seq, uHis, device):
        batch = seq.shape[0]
        self.max_seq = seq.shape[1]  # L
        self.max_bas = seq.shape[2]  # B
        seq_embs = self.itemEmb(seq)  # [batch, L, B, d]

        # horizontal
        lengths_h = [i + 1 for i in range(self.Lh)]
        self.conv_h = nn.ModuleList([nn.Conv3d(1, 8, (i, self.max_bas, self.dim)) for i in lengths_h]).to(device)

        out_hs = list()
        for conv in self.conv_h:
            conv_out = self.ac_conv(conv(seq_embs.unsqueeze(1))).squeeze(3).squeeze(3)
            pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            out_hs.append(pool_out.unsqueeze(2))
        out_h_cat = torch.cat(out_hs, 2)
        out_h = F.max_pool1d(out_h_cat, out_h_cat.size(2)).squeeze(2)

        # vertical
        lengths_v = [i + 1 for i in range(self.Lv)]
        self.conv_v = nn.ModuleList([nn.Conv3d(1, 8, (self.max_seq, i, self.dim)) for i in lengths_v]).to(device)

        out_vs = list()
        for convh in self.conv_v:
            convh_out = self.ac_conv(convh(seq_embs.unsqueeze(1))).squeeze(2).squeeze(3)
            pool_v_out = F.max_pool1d(convh_out, convh_out.size(2)).squeeze(2)
            out_vs.append(pool_v_out.unsqueeze(2))
        out_v_cat = torch.cat(out_vs, 2)
        out_v = F.max_pool1d(out_v_cat, out_v_cat.size(2)).squeeze(2)

        # dilation
        self.convd1 = nn.Conv3d(1, 8, (1, 3, self.dim), dilation=(1, 2 ** 0, 1)).to(device)
        out_v1 = self.dropout(self.ac_conv(self.convd1(seq_embs.unsqueeze(1))))
        self.convd2 = nn.Conv3d(8, 8, (1, 3, 1), dilation=(1, 2 ** 1, 1)).to(device)
        out_v2 = self.dropout(self.ac_conv(self.convd2(out_v1)))
        self.convd3 = nn.Conv3d(8, 8, (1, 3, 1), dilation=(1, 2 ** 2, 1)).to(device)
        out_v3 = self.dropout(self.ac_conv(self.convd3(out_v2)))
        out_d = out_v3.contiguous().view(batch, self.max_seq, -1, 8)

        # dilation
        self.convd1 = nn.Conv3d(1, 8, (3, 1, self.dim), dilation=(2 ** 0, 1, 1)).to(device)
        out_v1 = self.dropout(self.ac_conv(self.convd1(seq_embs.unsqueeze(1))))
        self.convd2 = nn.Conv3d(8, 8, (3, 1, 1), dilation=(2 ** 1, 1, 1)).to(device)
        out_v2 = self.dropout(self.ac_conv(self.convd2(out_v1)))
        self.convd3 = nn.Conv3d(8, 8, (3, 1, 1), dilation=(2 ** 2, 1, 1)).to(device)
        out_v3 = self.dropout(self.ac_conv(self.convd3(out_v2)))
        out_d2 = out_v3.contiguous().view(batch, self.max_bas, -1, 8)

        # sumpooling
        embs3d = out_d.sum(2)
        out_d = embs3d.sum(1)

        embs3d2 = out_d2.sum(2)
        embs2d2 = embs3d2.sum(1)
        out_d2 = embs2d2

        out4d = torch.tanh(torch.cat([out_h, out_v, out_d, out_d2], dim=1))

        # AFM
        scores_trans = self.out(out4d)
        scores_trans = F.softmax(scores_trans, dim=-1)
        embs_h = self.his_embds(uHis)
        gate = torch.sigmoid(self.g1(embs_h) + self.g2(out4d))

        scores = gate * scores_trans + (1 - gate) * uHis
        scores = scores / math.sqrt(self.dim)

        return scores
