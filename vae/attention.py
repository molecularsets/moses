import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

INF = 1e+38


class SelfAttention(nn.Module):
    def __init__(self, dims, activation=None):
        super().__init__()

        self.linear_in = nn.Linear(dims, dims, bias=False)
        self.linear_out = nn.Linear(dims * 2, dims, bias=False)
        self.activation = activation

    def forward(self, query):
        """Calculating output with attention

        :param query: [B, N, D] of floats, rnn outputs
        :return: [B, N, D] of floats, new outputs with attention
        """

        B, N, D = query.size()

        # In
        query = self.linear_in(query)

        # Scores
        scores = torch.bmm(query, query.transpose(1, 2))
        mask = torch.tensor(
            np.triu(np.ones((N, N))),
            dtype=torch.uint8, device=query.device
        )
        scores[:, mask] = -INF

        # Weights
        weights = F.softmax(scores, dim=-1)
        weights = weights.clone()
        weights[:, 0, :] = 0  # first to go gains nothing

        # Out
        mix = torch.bmm(weights, query)
        combined = torch.cat([mix, query], dim=2)
        output = self.linear_out(combined)
        if self.activation:
            output = self.activation(output)

        return output

    def forward_inference(self, query, context):
        """Output with attention inference

        :param query: [B, D] of floats, current output
        :param context: [B, N, D] of floats, outputs before
        :return: tuple of two:
            1. [B, D] of floats, new outputs
            2. [B, N] of floats, attention weights
        """

        if context is None:
            query = self.linear_in(query)
            mix = torch.zeros_like(query)
        else:
            # In
            query = self.linear_in(query)
            context = self.linear_in(context)

            # Calc scores & weights
            scores = torch.bmm(
                query.unsqueeze(1),
                context.transpose(1, 2)
            ).squeeze(1)
            weights = F.softmax(scores, dim=-1)

            # Combining vectors
            mix = torch.bmm(
                weights.unsqueeze(1),
                context
            ).squeeze(1)

        # Out
        combined = torch.cat([mix, query], dim=1)
        output = self.linear_out(combined)
        if context is not None:
            a = torch.bmm(
                output.unsqueeze(1),
                torch.cat([context, query.unsqueeze(1)], 1).transpose(1, 2)
            ).squeeze(1)
            a = F.softmax(a, dim=-1)
        else:
            a = torch.ones(query.size(0),
                           dtype=torch.float, device=query.device)
            a = a.unsqueeze(1)
        if self.activation:
            output = self.activation(output)

        return output, a
