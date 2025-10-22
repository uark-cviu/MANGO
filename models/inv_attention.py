import torch

class InvAttention(torch.nn.Module):
    def __init__(self, dim = 256, split_type= "split_half"):
      super(InvAttention, self).__init__()
      self.split_type = split_type
      self.d_scale = torch.nn.Parameter(torch.tensor(1.0))
      self.query  = torch.nn.Linear(dim, dim, bias=False)
      self.query_norm = torch.nn.LayerNorm(dim)
      self.key    = torch.nn.Linear(dim, dim, bias=False)
      self.key_norm = torch.nn.LayerNorm(dim)

    def split(self, inputs, split_type="split_half"):
      N, L, D  = inputs.size()
      if split_type == "split_alter":
        inputs = inputs.reshape(N, L // 2, 2, D).permute(0, 1, 3, 2)
        return inputs[:, :, :, 0], inputs[:, :, :, 1]
      elif split_type == "split_half":
        inputs = inputs.reshape(N, 2, L // 2, D).permute(0, 2, 3, 1)
        return inputs[:, :, :, 0], inputs[:, :, :, 1]

    def merge(self, x1, x2, split_type):
      N, _, D = x1.size()
      L = x1.size(1) + x2.size(1)
      outputs = torch.stack([x1, x2], axis=-1)
      if split_type == "split_alter":
        outputs = outputs.permute(0, 1, 3, 2).reshape(N, L, D)
      elif split_type == "split_half":
        outputs = outputs.permute(0, 3, 1, 2).reshape(N, L, D)
      return outputs

    def apply_mask(self, att):
      mask = torch.ones(size = (att.size(1), att.size(2)), dtype=att.dtype, device=att.device)
      mask = torch.triu(mask)
      mask = mask.reshape(1, att.size(1), att.size(2)).repeat(att.size(0), 1, 1)
      att.masked_fill_(mask == 0, float('-inf'))
      return att

    def forward(self, inputs, reverse = False, return_logdet=True):
      N, L, D = inputs.size()
      x1, x2 = self.split(inputs, split_type=self.split_type)
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
      logdet  = torch.linalg.slogdet(att)[1] * L / 2
      if return_logdet == True:
        return self.merge(x1, out, split_type = self.split_type), logdet
      else:
        return self.merge(x1, out, split_type = self.split_type), 0
