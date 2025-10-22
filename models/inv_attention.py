import torch
import torch.nn as nn
import numpy as np
from scipy import linalg as la

class LearnablePartitioning(torch.nn.Module):
    def __init__(self, token_length):
        super().__init__()

        weight = np.random.randn(token_length, token_length)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(torch.log(torch.abs(w_s)))
        self.w_u = nn.Parameter(w_u)

    def forward(self, inputs, reverse = False):

        weight = self.calc_weight()

        if not reverse:
            out = F.conv2d(inputs, weight)
        else:
            out = self.reverse(inputs)

        logdet = height * width * torch.sum(self.w_s)

        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2)

    def reverse(self, outputs):
        weight = self.calc_weight()

        return F.conv1d(outputs, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))

class InvAttention(torch.nn.Module):
    def __init__(self, dim = 256, total_token_length = 512, split_type= "MMCA", split_option = "v1"):
        super(InvAttention, self).__init__()
        assert split_type in ["MMCA", "IMCA", "LICA"]
        self.split_type = split_type
        self.split_option = split_option
        if self.split_type == "LICA":
            self.split_layer = LearnablePartitioning(total_token_length)
        self.d_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.query  = torch.nn.Linear(dim, dim, bias=False)
        self.query_norm = torch.nn.LayerNorm(dim)
        self.key    = torch.nn.Linear(dim, dim, bias=False)
        self.key_norm = torch.nn.LayerNorm(dim)

    def split(self, inputs, split_type="IMCA"):
        N, L, D  = inputs.size()
        logdet = 0
        if split_type == "IMCA":
            inputs = inputs.reshape(N, 2, L // 2, D).permute(0, 2, 3, 1)
            xa, xb = inputs[:,:, :, 0], inputs[:, :, :, 1]
            x1a = xa[:, 0::2, :]
            x2a = xa[:, 1::2, :]
            x1b = xb[:, 0::2, :]
            x2b = xb[:, 1::2, :]

            if self.split_option == "v1":
                x1, x2 = torch.cat([x1a, x1b], dim=1), torch.cat([x2a, x2b], dim=1)
            elif self.split_option == "v2":
                x1, x2 = torch.cat([x1a, x2b], dim=1), torch.cat([x2a, x1b], dim=1)
            elif self.split_option == "v3":
                x1, x2 = torch.cat([x2a, x1b], dim=1), torch.cat([x1a, x2b], dim=1)
            else:
                x1, x2 = torch.cat([x2a, x2b], dim=1), torch.cat([x1a, x1b], dim=1)

        elif split_type == "MMCA":
            inputs = inputs.reshape(N, 2, L // 2, D).permute(0, 2, 3, 1)
            xa, xb = inputs[:, :, :, 0], inputs[:, :, :, 1]
            if self.split_option == "v1":
                x1, x2 = xa, xb
            else:
                x1, x2 = xb, xa
        elif split_type == "LICA":
            x, logdet = self.split_layer(inputs)
            x1, x2 = x[:, :, :, 0], x[:, :, :, 1]

        return x1, x2, logdet


    def merge(self, x1, x2, split_type):
        N, _, D = x1.size()
        L = x1.size(1) + x2.size(1)
        logdet = 0
        K1 = x1.size(1) // 2
        K2 = x2.size(1) // 2
        logdet = 0
        if split_type == "IMCA":
            if self.split_option == "v1":
               x1a, x1b, x2a, x2b = x1[:, :K1, :], x1[:, K1:, :], x2[:, :K2, :], x2[:, K2:, ]
            elif self.split_option == "v2":
                x1a, x2b, x2a, x1b = x1[:, :K1, :], x1[:, K1:, :], x2[:, :K2, :], x2[:, K2:, ]
            elif self.split_option == "v3":
                x2a, x1b, x1a, x2b = x1[:, :K1, :], x1[:, K1:, :], x2[:, :K2, :], x2[:, K2:, ]
            else:
                x2a, x2b, x1a, x1b  = x1[:, :K1, :], x1[:, K1:, :], x2[:, :K2, :], x2[:, K2:, ]
            xa = torch.stack([x1a, x2a], dim = 2).reshape(N, -1, D)
            xb = torch.stack([x1b, x2b], dim = 2).reshape(N, -1, D)
            outputs = torch.cat([xa, xb], dim = 1)
        elif split_type == "MMCA":
            if self.split_option == "v1":
                outputs = torch.cat([x1, x2], dim = 1)
            else:
                outputs = torch.cat([x2, x1], dim = 1)
        elif split_type == "LICA":
          outputs = torch.stack([x1, x2], axis=-1)
          outputs, logdet = self.split_layer(outputs, reverse=True)

        return outputs, logdet

    def apply_mask(self, att):
        mask = torch.ones(size = (att.size(1), att.size(2)), dtype=att.dtype, device=att.device)
        mask = torch.triu(mask)
        mask = mask.reshape(1, att.size(1), att.size(2)).repeat(att.size(0), 1, 1)
        att.masked_fill_(mask == 0, float('-inf'))
        return att

    def forward(self, inputs, reverse = False, return_logdet=True):
        N, L, D = inputs.size()
        x1, x2, logdet = self.split(inputs, split_type=self.split_type)
        v = x2
        q = self.query_norm(self.query(x1))
        k = self.key_norm(self.key(x1))
        att = torch.bmm(q, k.permute(0, 2, 1)) / self.d_scale
        att = self.apply_mask(att)
        att = torch.softmax(att, dim=-1)
        att += torch.eye(att.size(1), dtype=att.dtype).unsqueeze(0).to(att.device) # Trick to ensure the invertibility
        if reverse:
            att = torch.linalg.pinv(att)
        out = torch.bmm(att, v)
        logdet  += torch.linalg.slogdet(att)[1] * L / 2
        if return_logdet == True:
            return self.merge(x1, out, split_type = self.split_type), logdet
        else:
            return self.merge(x1, out, split_type = self.split_type)
