
import torch


def make_length_mask(lengths, maxlen=None, device='cpu'):
    """
    Compute binary length mask.

    lengths: torch.Tensor(batch, dtype=int) should be on the desired
        output device.

    Returns
    =======

    mask: torch.ByteTensor(batch x seq_len)
    """
    maxlen = maxlen or lengths.detach().max()
    batch = len(lengths)
    return torch.arange(0, maxlen, dtype=torch.int64, device=device) \
                .repeat(batch, 1) \
                .lt(lengths.to(device).unsqueeze(1))


def log_sum_exp(x):
    """
    Numerically stable log_sum_exp

    Parameters
    ==========
    x : torch.tensor

    >>> import torch
    >>> x = torch.randn(10, 5)
    """
    max_score, _ = torch.max(x, -1)
    max_score_broadcast = max_score.unsqueeze(-1).expand_as(x)
    return max_score + torch.log(torch.sum(torch.exp(x - max_score_broadcast), -1))


def prepad(x, pad=0):
    return torch.nn.functional.pad(x, (1, 0), value=pad)
