
import torch


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
    from torch.nn.utils.rnn import pack_padded_sequence

    if not isinstance(lengths, torch.Tensor):
        lengths = torch.tensor(lengths)  # no need to use gpu

    lengths, sort = torch.sort(lengths, descending=True)
    _, unsort = sort.sort()

    if batch_first:
        inp = pack_padded_sequence(inp[sort], lengths.tolist())
    else:
        inp = pack_padded_sequence(inp[:, sort], lengths.tolist())

    return inp, unsort


def pad_flat_batch(emb, nwords, maxlen):
    """
    Transform a 2D flat batch (batch of words in multiple sentences) into a 3D
    padded batch where words have been allocated to their respective sentence
    according to user passed sentence lengths `nwords`

    Parameters
    ===========
    emb : torch.Tensor(total_words x emb_dim), flattened tensor of word embeddings
    nwords : torch.Tensor(batch), number of words per sentence

    Returns
    =======
    torch.Tensor(max_seq_len x batch x emb_dim) where:
        - max_seq_len = max(nwords)
        - batch = len(nwords)

    >>> emb = [[0], [1], [2], [3], [4], [5]]
    >>> nwords = [3, 1, 2]
    >>> pad_flat_batch(torch.tensor(emb), torch.tensor(nwords)).tolist()
    [[[0], [3], [4]], [[1], [0], [5]], [[2], [0], [0]]]
    """
    with torch.no_grad():
        if len(emb) != sum(nwords):
            raise ValueError("Got {} items but was asked to pad {}"
                             .format(len(emb), sum(nwords)))

        output, last = [], 0

        for sentlen in nwords:
            padding = (0, 0, 0, maxlen - sentlen)
            output.append(torch.nn.functional.pad(emb[last:last+sentlen], padding))
            last = last + sentlen

        # (seq_len x batch x emb_dim)
        output = torch.stack(output, dim=1)

    return output


def flatten_padded_batch(batch, nwords):
    """
    Inverse of pad_flat_batch

    Parameters
    ===========
    batch : tensor(seq_len, batch, encoding_size), output of the encoder
    nwords : tensor(batch), lengths of the sequence (without padding)

    Returns
    ========
    tensor(nwords, encoding_size)

    >>> batch = [[[0], [3], [4]], [[1], [0], [5]], [[2], [0], [0]]]
    >>> nwords = [3, 1, 2]
    >>> flatten_padded_batch(torch.tensor(batch), torch.tensor(nwords)).tolist()
    [[0], [1], [2], [3], [4], [5]]
    """
    output = []
    for sent, sentlen in zip(batch.t(), nwords):
        output.extend(list(sent[:sentlen].chunk(sentlen)))

    return torch.cat(output, dim=0)


def detach_hidden(hidden):
    for l, h in enumerate(hidden):
        if isinstance(h, torch.Tensor):
            hidden[l] = h.detach()
        else:
            hidden[l] = tuple(h_.detach() for h_ in h)

    return hidden


def update_hidden(old_hidden, new_hidden, mask):
    if old_hidden[0] is None:
        return new_hidden

    mask = mask.unsqueeze(0).unsqueeze(2)

    for l, (old, new) in enumerate(zip(old_hidden, new_hidden)):
        if isinstance(old, tuple):
            new_hidden[l] = (mask.float() * new[0] + (1-mask).float() * old[0],
                             mask.float() * new[1] + (1-mask).float() * old[1])
        else:
            new_hidden[l] = mask.float() * new + (1-mask).float() * old

    return new_hidden


def sequential_dropout(inp, p, training):
    if not training or not p:
        return inp

    mask = inp.new(1, inp.size(1), inp.size(2)).bernoulli_(1 - p)
    mask = mask / (1 - p)

    return inp * mask.expand_as(inp)


def embedded_dropout(embed, words, dropout=0.1, scale=None):
    if dropout:
        mask = embed.weight.new() \
                           .resize_((embed.weight.size(0), 1)) \
                           .bernoulli_(1 - dropout) \
                           .expand_as(embed.weight) \
                           .div_(1 - dropout)
        masked_embed_weight = mask * embed.weight
    else:
        masked_embed_weight = embed.weight
    if scale:
        masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

    padding_idx = embed.padding_idx
    if padding_idx is None:
        padding_idx = -1

    return torch.nn.functional.embedding(words, masked_embed_weight,
                                         padding_idx, embed.max_norm, embed.norm_type,
                                         embed.scale_grad_by_freq, embed.sparse)


def batch_index_add(t, index, src):
    """
    Add values in `src` indexed by `index`

    t: (batch x vocab)
    index: (batch x cache_size)
    src: (batch x cache_size)
    """
    batch, vocab = t.size()
    ex = torch.arange(0, batch, out=t.new()).unsqueeze(1).long() * vocab
    added = t.view(-1).index_add(0, (index + ex).view(-1), src.view(-1))
    return added.view(batch, vocab)
