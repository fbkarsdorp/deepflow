
import itertools

import tqdm
import torch
import torch.nn.functional as F

from . import torch_utils


class Cache(object):
    """
    Continuous Cache for Neural Models following: https://arxiv.org/abs/1612.04426
    Parameters:
    -----------
    dim: int, dimensionality of the keys
    size: int, size of the cache
    vocab: int, size of the output vocabulary
    """
    def __init__(self, dim, size, vocab, device='cpu'):

        self.dim = dim
        self.size = size
        self.vocab = vocab
        self.device = device

        self.stored = 0         # number of items stored in the cache
        self.current = 0        # index along size that should get written
        self.memkeys = torch.zeros(self.size, 1, self.dim).to(device)
        self.memvals = torch.zeros(self.size, 1, dtype=torch.int64).to(device)

    def reset(self):
        self.stored = 0
        self.current = 0
        self.memkeys.zero_()
        self.memvals.zero_()

    def add(self, keys, vals):
        """
        Parameters:
        -----------
        keys: torch.Tensor(n, batch, dim)
        vals: torch.LongTensor(n, batch)
        """
        if keys.size()[:-1] != vals.size():
            raise ValueError("Wrong key-val dims. Keys: {}, vals: {}".format(
                str(keys.size()), str(vals.size())))

        batch = keys.size(1)

        if self.memkeys.size(1) == 1 and batch > 1:
            # expand along batch dimension
            self.memkeys = self.memkeys.repeat(1, batch, 1)
            self.memvals = self.memvals.repeat(1, batch)

        if self.memkeys.size(1) != batch:
            raise ValueError(
                "Wrong batch dimension. Expected {} but got {} elements".format(
                    self.memkeys.size(1), batch))

        if keys.size(0) > self.size:
            keys, vals = keys[-self.size:], vals[-self.size:]

        limit = min(self.size, self.current + keys.size(0))
        index = torch.arange(self.current, limit, dtype=torch.int64, device=self.device)
        self.memkeys.index_copy_(0, index, keys[:len(index)])
        self.memvals.index_copy_(0, index, vals[:len(index)])
        self.current = limit % self.size

        if len(index) < len(keys):
            indexed = len(index)
            index = torch.arange(self.current, len(keys) - indexed,
                                 dtype=torch.int64, device=self.device)
            self.memkeys.index_copy_(0, index, keys[indexed:])
            self.memvals.index_copy_(0, index, vals[indexed:])
            self.current = len(keys) - indexed

        self.stored = min(self.size, self.stored + len(keys))

    def query(self, query):
        """
        Return scores for words in the cache given an input query.
        Parameters:
        -----------
        query: torch.Tensor(batch, hid_dim)
        Returns:
        --------
        scores: torch.Tensor(batch, size), output is just the dotproduct
            with the keys in the cache
        vals: torch.LongTensor(batch, size)
        """
        # select full entries
        memkeys, memvals = self.memkeys[:self.stored], self.memvals[:self.stored]
        # dot product => (batch x size)
        scores = (memkeys * query.unsqueeze(0)).sum(2)

        return scores.t(), memvals.t()

    def interpolate(self, query, logits, alpha, theta):
        # query
        cache_logits, vals = self.query(query)
        # interpolate
        cache_prob = alpha * F.softmax(theta * cache_logits, dim=1)
        prob = (1 - alpha) * F.softmax(logits, dim=1)
        prob = torch_utils.batch_index_add(prob, vals, cache_prob)

        return prob


def evaluate(model, encoder, dev, cache_size=200, alpha=0.1, theta=0.1, batch=1):
    """
    Evaluate model using a cache
    """
    cache = Cache(model.hidden_dim, cache_size, len(encoder.word.w2i), model.device)
    hidden = None
    tloss, tinsts = 0, 0

    with torch.no_grad():

        for stuff in tqdm.tqdm(dev.get_batches(batch, yield_stops=True)):

            if stuff is None:  # avoid cache reaching over different sentences
                cache.reset()
                continue

            sents, conds = stuff

            (words, nwords), (chars, nchars), conds = encoder.transform_batch(
                sents, conds, model.device)

            # outs: (seq x batch x hidden_dim)
            outs, hidden = model(
                words, nwords, chars, nchars, conds, hidden=hidden, project=False)

            sources, targets = outs[:-1], words[1:]

            for source, target in zip(sources, targets):
                # (batch x vocab)
                logits = model.proj(source)
                if cache.stored > 0:
                    prob = cache.interpolate(source, logits, alpha, theta)
                else:
                    prob = F.softmax(logits, dim=1)

                tloss += F.nll_loss(
                    prob.add(1e-8).log(), target,
                    weight=model.nll_weight, size_average=False
                ).item()

                cache.add(source.unsqueeze(0), target.unsqueeze(0))

            tinsts += (sum(nwords) - batch)  # remove 1 per instance

    return model.loss_formatter(tloss / tinsts)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--dev')
    parser.add_argument('--dpath', default='./data/ohhla.vocab.phon.json')
    parser.add_argument('--size', default=200, type=int)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    from generation import utils, model_loader

    def range_float(start, stop, step):
        """
        Range with floats
        """
        factor = 1 / step
        return (i / factor for i in range(int(start * factor),
                                          int(stop * factor),
                                          int(step * factor)))

    model, encoder = model_loader(args.model)
    model.to(args.device)
    model.eval()

    dev = utils.CorpusReader(args.dev, dpath=args.dpath or None)
    grid = itertools.product(range_float(0, 1, 0.1), range_float(0, 0.5, 0.05))

    with open("{}.cache.eval.csv".format(model.modelname), 'w') as f:
        f.write("alpha,theta,loss\n")
        for theta, alpha in grid:
            loss = evaluate(
                model, encoder, dev, cache_size=args.size, alpha=alpha, theta=theta)
            f.write("{},{},{}\n".format(alpha, theta, loss))
