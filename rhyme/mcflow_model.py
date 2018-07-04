
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from models.torch_utils import pack_sort


def make_3d_mask(alen, blen, alignment):
    # alignment: (batch x max_alen x max_blen)
    mask = torch.zeros_like(alignment)
    for batch, (a, b) in enumerate(zip(alen, blen)):
        mask[batch, :a, :b] = 1.

    return mask


def init_pretrained(path, encoder, weight):
    import gensim
    model = gensim.models.KeyedVectors.load(path)
    model.init_sims()           # L2 norms
    _, dim = model.vectors.shape
    unknowns = 0
    for w, idx in encoder.index.items():
        try:
            weight[idx, :].copy_(torch.tensor(model[w]))
        except KeyError:
            unknowns += 1

    print("Found {}/{} items in pretrained vocabulary".format(
        len(encoder)-unknowns, len(encoder)))


class Model(nn.Module):
    def __init__(self, encoders, emb_dim, hid_dim, layers=1, dropout=0.25,
                 pretrained_emb=None, feats=['stress', 'onset', 'nucleus', 'coda']):

        self.encoders = encoders
        self.feats = feats
        self.dropout = dropout
        super().__init__()

        # embedding layers
        embs, rnn_inp = {}, 0
        for name, encoder in encoders.items():
            if name not in self.feats:
                continue
            if name == 'syllable' and pretrained_emb is not None:
                emb = nn.Embedding(len(encoder), pretrained_emb)
                rnn_inp += pretrained_emb
            else:
                emb = nn.Embedding(len(encoder), emb_dim)
                rnn_inp += emb_dim
            embs[name] = emb
            self.add_module(name, emb)
        self.embs = embs

        # rnns
        self.rnn = nn.GRU(rnn_inp, hid_dim, num_layers=layers,
                          bidirectional=True, dropout=self.dropout)

    def forward_single(self, inp, lengths):
        # inp: (seq_len x batch)
        emb = torch.cat([self.embs[name](inp[name]) for name in sorted(self.embs)], 2)
        emb = nn.functional.dropout(emb, p=self.dropout, training=self.training)
        emb, unsort = pack_sort(emb, lengths)
        out, _ = self.rnn(emb)
        out, _ = unpack(out)
        out = out[:, unsort]

        return out

    def forward(self, a, alen, b, blen):
        a_out, b_out = self.forward_single(a, alen), self.forward_single(b, blen)
        # compute attention matrix
        att = torch.einsum('sbm,tbm->bst', [a_out, b_out])

        return att

    def nll_loss(self, att, alen, blen, alignment):
        pos = alignment * nn.functional.logsigmoid(att)
        neg = (1-alignment) * torch.log((1-nn.functional.sigmoid(att)) + 1e-9)
        loss = -(pos + neg)
        # mask loss
        loss = loss * make_3d_mask(alen, blen, alignment)
        # mean batch loss
        loss = loss.sum(dim=2).sum(dim=1).mean()

        return loss

    def train_model(self, opt, batches):
        rloss = 0
        total_loss = 0
        report = 100

        for idx, (a, alen, b, blen, alignment) in enumerate(batches):
            att = self(a, alen, b, blen)
            loss = self.nll_loss(att, alen, blen, alignment)
            opt.zero_grad()
            if self.training:
                loss.backward()
                opt.step()

            rloss += loss.item()
            total_loss += loss.item()

            if idx and idx % 100 == 0:
                print("batch {:<5}: {}".format(idx, rloss / report))
                rloss = 0

        print("train loss", total_loss / idx)

    def runeval(self, dataset):
        def accuracy(att, alignment, alen, blen):
            acc, total = 0, 0
            for att, alignment, alen, blen in zip(att, alignment, alen, blen):
                total += 1
                n = int(alignment.sum())  # assume we now `n`
                _, idxs = torch.sort(att[:alen, :blen].contiguous().view(-1))
                mask = torch.zeros_like(alignment)
                for i, j in zip((idxs / blen)[-n:], (idxs % blen)[-n:]):
                    mask[i, j] = 1
                acc += int(alignment.masked_select(mask.byte()).sum()) / n

            return acc / total  # mean accuracy

        tloss, tlen, tacc = 0, 0, 0
        with torch.no_grad():
            for a, alen, b, blen, alignment in dataset:
                att = self(a, alen, b, blen)
                tloss += self.nll_loss(att, alen, blen, alignment).item()
                assert att.size() == alignment.size()
                tacc += accuracy(att, alignment, alen, blen)
                tlen += 1

        print("dev loss", tloss / tlen)
        print("dev acc", tacc / tlen)
        # print(att[0, :alen[0], :blen[0]])
        # print(att[0, :alen[0], :blen[0]] > 0.5)
        # print(alignment[0, :alen[0], :blen[0]])


def split_train_dev(dataset, split):
    import random
    random.shuffle(dataset)
    point = int(len(dataset) * split)
    dev, train = dataset[:point], dataset[point:]
    return train, dev


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_dim', type=int, default=20)
    parser.add_argument('--hid_dim', type=int, default=50)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--pretrained')
    args = parser.parse_args()

    from rhyme.load_rhymes import get_dataset, get_iterator, fit_encoders
    print("Loading encoders")
    encoders = fit_encoders()

    feats = ['nucleus', 'syllable']
    model = Model(encoders, args.emb_dim, args.hid_dim, feats=feats,
                  dropout=args.dropout, layers=args.layers,
                  pretrained_emb=200 if args.pretrained else None)
    if args.pretrained and 'syllable' in feats:
        init_pretrained(args.pretrained,
                        encoders['syllable'],
                        model.embs['syllable'].weight.data)

    print(model)
    print(sum(p.nelement() for p in model.parameters()))
    model.to(args.device)

    print("Loading dataset")    # for mcflow in-memory is fine
    train, dev = split_train_dev(get_dataset(encoders), 0.1)

    print("Starting training")
    opt = torch.optim.Adam(list(model.parameters()), lr=0.01)
    for epoch in range(args.epochs):
        print("epoch", epoch)
        model.train_model(opt, get_iterator(train, args.batch_size, 0, args.device))
        model.runeval(get_iterator(dev, args.batch_size, 0, args.device))
        print("-- train test --")
        model.runeval(get_iterator(train[:100], args.batch_size, 0, args.device))
