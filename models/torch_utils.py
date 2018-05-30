
import torch
from torch.nn.utils.rnn import pack_padded_sequence


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


def pack_sort(inp, lengths, batch_first=False):
    """
    Transform input into PaddedSequence sorting batch by length (as required).
    Also return an index variable that unsorts the output back to the original
    order.

    Parameters:
    -----------
    inp: torch.Tensor(seq_len x batch x dim)
    lengths: LongTensor of length ``batch``

    >>> from torch.nn.utils.rnn import pad_packed_sequence as unpack
    >>> inp = torch.tensor([[1, 3], [2, 4], [0, 5]], dtype=torch.float)
    >>> lengths = torch.tensor([2, 3]) # unsorted order
    >>> sorted_inp, unsort = pack_sort(inp, lengths)
    >>> sorted_inp, _ = unpack(sorted_inp)
    >>> sorted_inp[:, unsort].tolist()  # original order
    [[1.0, 3.0], [2.0, 4.0], [0.0, 5.0]]
    >>> sorted_inp.tolist()  # sorted by length
    [[3.0, 1.0], [4.0, 2.0], [5.0, 0.0]]
    """
    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths)  # no need to use gpu

    lengths, sort = torch.sort(lengths, descending=True)
    _, unsort = sort.sort()

    if batch_first:
        inp = pack_padded_sequence(
            inp[sort], lengths.tolist(), batch_first=batch_first)
    else:
        inp = pack_padded_sequence(
            inp[:, sort], lengths.tolist(), batch_first=batch_first)

    return inp, unsort
