
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
