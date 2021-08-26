import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import torch
import torch.nn as nn
import numpy as np

from models.models import CNN4Encoder


class FewShotModel(nn.Module):
    # this looks fine now
    def __init__(self, args):
        super().__init__()
        self.args = args
        # only CNN4 is currently supported
        self.encoder = CNN4Encoder()
        
    def split_instances(self, data):
        way = self.args.num_classes_per_set
        shot = self.args.train_num_samples_per_class
        query = self.args.num_target_samples
        return (torch.Tensor(np.arange(way*shot)).long().view(1, shot, way),
                torch.Tensor(np.arange(way*shot, way * (shot + query))).long().view(1, query, way))

    def forward(self, x, adaptation_labels, labels_aux, bit, training=True, get_feature=False):
        if get_feature:
            # get feature with the provided embeddings
            return self.encoder(x)
        else:
            # feature extraction
            x = x.squeeze(0)
            instance_embs = self.encoder(x)
            num_inst = instance_embs.shape[0]
            # split support query set for few-shot data
            support_idx, query_idx = self.split_instances(x)
            if training:
                logits, logits_reg = self._forward(
                    instance_embs, support_idx, query_idx, adaptation_labels, labels_aux, bit, training)
                return logits, logits_reg
            else:
                logits = self._forward(
                    instance_embs, support_idx, query_idx, adaptation_labels, labels_aux, bit, training)
                return logits

    def _forward(self, x, support_idx, query_idx, adaptation_labels, bit):
        raise NotImplementedError('Suppose to be implemented by subclass')


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_v, d_v)  # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(
            sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output


class FEAT(FewShotModel):
    def __init__(self, args):
        super().__init__(args)
        # only CNN4 is currently supported
        hdim = 64
        self.slf_attn = MultiHeadAttention(1, hdim, hdim, hdim, dropout=0.5)

    def _forward(self, instance_embs, support_idx, query_idx, adaptation_labels, labels_aux, bit, training):
        # extract the arguments into more readable form
        way = self.args.num_classes_per_set
        shot = self.args.train_num_samples_per_class
        query_n = self.args.num_target_samples
        use_euclidean = True  # the paper uses Euclidean distance
        temperature = 32
        temperature2 = 64
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        support = instance_embs[support_idx.contiguous().view(-1)]

        # get the prototypes
        prototype_zero = support[adaptation_labels[:, bit] < 1].mean(dim=0)
        prototype_one = support[adaptation_labels[:, bit] > 0].mean(dim=0)
        # Ntask x NK x d
        proto = torch.stack([prototype_zero, prototype_one]).unsqueeze(dim=0)

        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = np.prod(query_idx.shape[-2:])

        support = support.contiguous().view(*(support_idx.shape + (-1,)))
        query = instance_embs[query_idx.contiguous(
        ).view(-1)].contiguous().view(*(query_idx.shape + (-1,)))

        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        proto = self.slf_attn(proto, proto, proto)
        if use_euclidean:
            # (Nbatch*Nq*Nw, 1, d)
            query = query.view(-1, emb_dim).unsqueeze(1)
            proto = proto.unsqueeze(1).expand(
                num_batch, num_query, num_proto, emb_dim).contiguous()
            proto = proto.view(num_batch*num_query, num_proto,
                               emb_dim)  # (Nbatch x Nq, Nk, d)

            logits = - torch.sum((proto - query) ** 2, 2) / \
                temperature
        else:
            proto = F.normalize(proto, dim=-1)  # normalize for cosine distance
            query = query.view(num_batch, -1, emb_dim)  # (Nbatch,  Nq*Nw, d)

            logits = torch.bmm(query, proto.permute(
                [0, 2, 1])) / temperature
            logits = logits.view(-1, num_proto)

        # for regularization
        if training:
            # don't calculate the mean, just take them as they are
            aux_zero = instance_embs[labels_aux[:, bit] < 1].unsqueeze(dim=0)
            aux_zero_emb = self.slf_attn(aux_zero, aux_zero, aux_zero)
            aux_zero_center = aux_zero_emb.mean(dim=1)

            aux_one = instance_embs[labels_aux[:, bit] > 0].unsqueeze(dim=0)
            aux_one_emb = self.slf_attn(aux_one, aux_one, aux_one)
            aux_one_center = aux_one_emb.mean(dim=1)

            if use_euclidean:
                aux_zero_center = aux_zero_center.expand(instance_embs.shape)
                aux_one_center = aux_one_center.expand(instance_embs.shape)
                logits_reg_zero = - \
                    torch.sum((aux_zero_center-instance_embs)
                              ** 2, dim=1) / temperature2
                logits_reg_one = - \
                    torch.sum((aux_one_center-instance_embs)
                              ** 2, dim=1) / temperature2
                logits_reg = torch.stack([logits_reg_zero, logits_reg_one], dim=1)
            else:
                raise ValueError("Not implemented yet")

            return logits, logits_reg
        else:
            return logits
