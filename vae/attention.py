import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

INF = 1e+38


class SelfAttention(nn.Module):
    def __init__(self, dimensions, attention_type='general'):
        super().__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)

    def forward(self, query):
        """

        :param query: n_len, n_batch, n_d
        :return:
        """
        query = query.t().contiguous()
        n_batch, n_len, d = query.size()

        if self.attention_type == "general":
            query = query.view(n_batch * n_len, d)
            query = self.linear_in(query)
            query = query.view(n_batch, n_len, d)

        attention_scores = torch.bmm(
            query,
            query.transpose(1, 2).contiguous()
        )
        mask = torch.tensor(np.triu(np.ones((n_len, n_len))),
                            dtype=torch.uint8, device=query.device)
        attention_scores[:, mask] = -INF

        attention_scores = attention_scores.view(n_batch * n_len, n_len)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = attention_weights.view(n_batch, n_len, n_len)
        attention_weights = attention_weights.clone()
        attention_weights[:, 0, :] = 0  # first to go gains nothing

        mix = torch.bmm(attention_weights, query)
        combined = torch.cat([mix, query], dim=2)
        combined = combined.view(n_batch * n_len, 2 * d)

        output = self.linear_out(combined).view(n_batch, n_len, d)
        output = F.tanh(output)

        return output.t().contiguous(), attention_weights

    def forward_inference(self, query, context):
        n_batch, d_h = query.size()
        _, n_len, _ = context.size()

        if self.attention_type == "general":
            query = self.linear_in(query)

        attention_scores = torch.bmm(
            query.unsqueeze(1),
            context.transpose(1, 2).contiguous()
        ).squeeze(1)


        attention_weights = F.softmax(attention_scores, dim=-1)

        mix = torch.bmm(
            attention_weights.unsqueeze(1),
            context
        ).squeeze(1)

        combined = torch.cat([mix, query], dim=1)

        output = self.linear_out(combined)
        output = F.tanh(output)

        return output, attention_weights

