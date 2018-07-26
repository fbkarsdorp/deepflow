
import random
import math
import time
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack

import pronouncing
import tqdm

import utils


class RNNLanguageModel(nn.Module):
    def __init__(self, encoder, layers, wemb_dim, cemb_dim, hidden_dim, cond_dim,
                 dropout=0.0):

        self.layers = layers
        self.wemb_dim = wemb_dim
        self.cemb_dim = cemb_dim
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.dropout = dropout
        self.modelname = self.get_modelname()
        super().__init__()

        wvocab = encoder.word.size()
        cvocab = encoder.char.size()

        nll_weight = torch.ones(wvocab)
        nll_weight[encoder.word.pad] = 0 
        self.register_buffer('nll_weight', nll_weight)
        # TODO: pick one of those depending on loss level
        # cnll_weight = torch.ones(wvocab)
        # cnll_weight[encoder.char.pad] = 0
        # self.register_buffer('cnll_weight', cnll_weight)

        # embeddings
        self.wembs = nn.Embedding(wvocab, wemb_dim, padding_idx=encoder.word.pad)
        self.cembs = nn.Embedding(cvocab, cemb_dim, padding_idx=encoder.char.pad)
        self.crnn = nn.LSTM(cemb_dim, cemb_dim//2, bidirectional=True, dropout=dropout)
        input_dim = wemb_dim + cemb_dim

        # conds
        self.conds = {}
        for cond, cenc in encoder.conds.items():
            cemb = nn.Embedding(cenc.size(), cond_dim)
            self.add_module('cond_{}'.format(cond), cemb)
            self.conds[cond] = cemb
            input_dim += cond_dim

        # rnn
        self.rnn = nn.LSTM(input_dim, hidden_dim, layers, dropout=dropout)

        # output
        self.proj = nn.Linear(hidden_dim, wvocab)

        if wemb_dim == hidden_dim:
            print("Tying embedding and projection weights")
            self.proj.weight = self.wembs.weight

        self.init()

    def init(self):
        pass

    def save(self, fpath, encoder):
        self.eval()
        old_device = self.device
        self.to('cpu')
        with open(fpath, 'w') as f:
            torch.save({'model': self, 'encoder': encoder}, f)
        self.train()
        self.to(old_device)

    def get_modelname(self):
        from datetime import datetime

        return "{}.{}".format(
            type(self).__name__,
            datetime.now().strftime("%Y-%m-%d+%H:%M:%S"))

    @property
    def device(self):
        return next(self.parameters()).device

    def embed_chars(self, char, nchars, nwords):
        cembs = self.cembs(char)
        cembs, unsort = utils.pack_sort(cembs, nchars)
        _, hidden = self.crnn(cembs)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        cembs = hidden[:, unsort, :].transpose(0, 1).contiguous().view(sum(nwords), -1)
        cembs = utils.pad_flat_batch(cembs, nwords, max(nwords))

        return cembs

    def forward(self, word, nwords, char, nchars, conds, hidden=None):
        # (seq x batch x wemb_dim)
        wembs = self.wembs(word)
        # (seq x batch x cemb_dim)
        cembs = self.embed_chars(char, nchars, nwords)

        embs = torch.cat([wembs, cembs], -1)

        if conds:
            conds = [self.conds[c](conds[c]) for c in sorted(conds)]
            # expand
            seq, batch = word.size()
            conds = [c.expand(seq, batch, -1) for c in conds]
            # concatenate
            embs = torch.cat([embs, *conds], -1)

        embs, unsort = utils.pack_sort(embs, nwords)
        outs, hidden = self.rnn(embs, hidden)
        outs, _ = unpack(outs)
        outs = outs[:, unsort]
        if isinstance(hidden, tuple):
            hidden = hidden[0][:, unsort], hidden[1][:, unsort]
        else:
            hidden = hidden[:, unsort]

        logits = self.proj(outs)

        return logits, hidden

    def loss(self, logits, words, nwords):
        seq, batch, vocab = logits.size()

        loss = F.cross_entropy(
            logits.view(batch * seq, vocab), words.view(-1),
            weight=self.nll_weight, size_average=False)

        # remove 1 per batch instance
        insts = sum(nwords) - len(nwords)

        return loss, insts

    def sample(self, encoder, nchars=100, conds=None, hidden=None):
        """
        Generate stuff
        """
        self.eval()

        # TODO: batch sampling
        batch = 1

        # sample conditions if needed
        conds = conds or {}
        for c in self.conds:
            # sample conds
            if c not in conds:
                bcond = random.choice(list(encoder.conds[c].w2i.values()))
                bcond = torch.tensor([bcond] * batch, dtype=torch.int64)
                conds[c] = bcond.to(self.device)

        word = [encoder.word.bos] * batch  # (batch)
        word = torch.tensor(word, dtype=torch.int64).to(self.device)
        char, nchars = [[encoder.char.bol] * batch], 1  # (1 x nwords)
        char = torch.tensor(char, dtype=torch.int64).to(self.device)

        output, scores = [], 0.0

        with torch.no_grad():
            while True:
                # embeddings
                wemb = self.wembs(word.unsqueeze(0))
                cemb = self.embed_chars(char, nchars, [1] * batch)
                emb = torch.cat([wemb, cemb], -1)
                if conds:
                    conds = [self.conds[c](conds[c]) for c in sorted(conds)]
                    # expand
                    conds = [c.expand(1, batch, -1) for c in conds]
                    # concatenate
                    emb = torch.cat([emb, *conds], -1)

                outs, hidden = self.rnn(embs, hidden)
                # (1 x 1 x vocab)
                logits = self.proj(outs)
                # (1 x 1 x vocab) -> (1 x vocab)
                logits = logits.squeeze(0)

                preds = F.log_softmax(logits, dim=-1)
                word = (preds / 1).exp().multinomial(1)
                score = preds.gather(1, word)
                word, score = word.squeeze(0), score.squeeze(0)

                # accumulate
                scores += score.cpu()

                # sampled word to characters

    def dev(self, corpus, encoder, best_loss, fails):
        hidden = None
        tloss = tinsts = 0

        self.eval()

        with torch.no_grad():
            for sents, conds in tqdm.tqdm(corpus.get_batches(50)):
                (words, nwords), (chars, nchars), conds = encoder.transform_batch(
                    sents, conds, self.device)
                logits, hidden = self(words, nwords, chars, nchars, conds, hidden)
                loss, insts = self.loss(logits[:-1], words[1:], nwords)
                tloss += loss.item()
                tinsts += insts

        self.train()

        tloss = math.exp(tloss / insts)
        print("Dev loss: {:g}".format(tloss))
        
        if tloss < best_loss:
            print("New best dev loss: {:g}".format(tloss))
            best_loss = tloss
            fails = 0
            self.save(self.modelname, encoder)
        else:
            fails += 1
            print("Failed {} time to improve best dev loss: {}".format(fails, best_loss))

        print()
        for _ in range(20):
            print(self.sample(encoder))
        print()

        return best_loss, fails

    def train_model(self, corpus, encoder, epochs=5, lr=0.001, clipping=5, dev=None,
                    patience=3, minibatch=15, trainer='Adam', repfreq=1000, checkfreq=0,
                    lr_weight=1, bptt=1):

        # get trainer
        if trainer.lower() == 'adam':
            trainer = torch.optim.Adam(self.parameters(), lr=lr)
        elif trainer.lower() == 'sgd':
            trainer = torch.optim.SGD(self.parameters(), lr=lr)
        else:
            raise ValueError("Unknown trainer: {}".format(trainer))

        # local variables
        hidden = None
        best_loss, fails = float('inf'), 0

        for e in range(epochs):

            tinsts = tloss = 0.0
            start = time.time()

            for idx, (sents, conds) in enumerate(corpus.get_batches(minibatch)):
                (words, nwords), (chars, nchars), conds = encoder.transform_batch(
                    sents, conds, self.device)

                # forward
                logits, hidden = self(words, nwords, chars, nchars, conds, hidden)

                # loss
                loss, insts = self.loss(logits[:-1], words[1:], nwords)
                (loss/insts).backward(retain_graph=bptt > 1)

                # bptt
                if idx % bptt == 0:
                    # step
                    nn.utils.clip_grad_norm_(self.parameters(), clipping)
                    trainer.step()
                    trainer.zero_grad()
                    # detach
                    if isinstance(hidden, torch.Tensor):
                        hidden = hidden.detach()
                    else:
                        hidden = tuple(h.detach() for h in hidden)

                tinsts, tloss = tinsts + insts, tloss + loss.item()

                if idx and idx % (repfreq//minibatch) == 0:
                    speed = int(tinsts / (time.time() - start))
                    print("Epoch {:<3} items={:<10} loss={:<10g} items/sec={}"
                          .format(e, idx, math.exp(min(tloss/tinsts, 100)), speed))
                    tinsts = tloss = 0.0
                    start = time.time()

                if dev and checkfreq and idx and idx % (checkfreq // minibatch) == 0:
                    best_loss, fails = self.dev(dev, encoder, best_loss, fails)

                    # update lr
                    if fails > 0:
                        for pgroup in trainer.param_groups:
                            pgroup['lr'] = pgroup['lr'] * lr_weight

            if dev and not checkfreq:
                best_loss, fails = self.dev(dev, encoder, best_loss, fails)
                # update lr
                if fails > 0:
                    for pgroup in trainer.param_groups:
                        pgroup['lr'] = pgroup['lr'] * lr_weight


def sample(model, encoder, nchars=100, conds=None, hidden=None):
    """
    Generate stuff
    """
    model.eval()

    # TODO: batch sampling
    batch = 1

    # sample conditions if needed
    conds = conds or {}
    for c in model.conds:
        # sample conds
        if c not in conds:
            bcond = random.choice(list(encoder.conds[c].w2i.values()))
            bcond = torch.tensor([bcond] * batch, dtype=torch.int64)
            conds[c] = bcond.to(model.device)

    word = [encoder.word.bos] * batch  # (batch)
    word = torch.tensor(word, dtype=torch.int64).to(model.device)
    char, nchars = [[encoder.char.bol] * batch], [1] * batch  # (1 x nwords)
    char = torch.tensor(char, dtype=torch.int64).to(model.device)

    output, scores = [], 0.0

    with torch.no_grad():
        while True:
            # embeddings
            wemb = model.wembs(word.unsqueeze(0))
            cemb = model.embed_chars(char, nchars, [1] * batch)
            embs = torch.cat([wemb, cemb], -1)
            if conds:
                conds = [model.conds[c](conds[c]) for c in sorted(conds)]
                # expand
                conds = [c.expand(1, batch, -1) for c in conds]
                # concatenate
                embs = torch.cat([embs, *conds], -1)

            outs, hidden = model.rnn(embs, hidden)
            # (1 x 1 x vocab)
            logits = model.proj(outs)
            # (1 x 1 x vocab) -> (1 x vocab)
            logits = logits.squeeze(0)

            preds = F.log_softmax(logits, dim=-1)
            word = (preds / 1).exp().multinomial(1)
            score = preds.gather(1, word)
            word, score = word.squeeze(0), score.squeeze(0)
            print(word.size())

            # accumulate
            scores += score.cpu()
            return


# from utils import CorpusEncoder, CorpusReader
# reader = CorpusReader('./data/ohhla-beatstress.dev.jsonl')
# encoder = CorpusEncoder.from_corpus(reader)
# model = RNNLanguageModel(encoder, 1, 100, 100, 250, 50)

sample(model, encoder)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train')
    parser.add_argument('--dev')
    parser.add_argument('--wemb_dim', type=int, default=100)
    parser.add_argument('--cemb_dim', type=int, default=100)
    parser.add_argument('--cond_dim', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=250)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--maxsize', type=int, default=10000)
    parser.add_argument('--disable_chars', action='store_true')
    # train
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_weight', type=float, default=1.0)
    parser.add_argument('--trainer', default='Adam')
    parser.add_argument('--clipping', type=float, default=5.0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--word_dropout', type=float, default=0.2)
    parser.add_argument('--minibatch', type=int, default=20)
    parser.add_argument('--repfreq', type=int, default=1000)
    parser.add_argument('--checkfreq', type=int, default=0)
    # pytorch
    parser.add_argument('--device', default='cpu')
    # extra
    parser.add_argument('--penn', action='store_true')

    args = parser.parse_args()

    from utils import CorpusEncoder, CorpusReader

    print("Encoding corpus")
    start = time.time()
    train = CorpusReader(args.train)
    dev = CorpusReader(args.dev)

    encoder = CorpusEncoder.from_corpus(train, most_common=args.maxsize)
    print("... took {} secs".format(time.time() - start))

    print("Building model")
    lm = RNNLanguageModel(encoder, args.layers, args.wemb_dim, args.cemb_dim,
                          args.hidden_dim, args.cond_dim, dropout=args.dropout)
    print(lm)
    print("Storing model to path {}".format(lm.modelname))
    lm.to(args.device)

    print("Training model")
    lm.train_model(train, encoder, epochs=args.epochs, minibatch=args.minibatch,
                   dev=dev, lr=args.lr, trainer=args.trainer, clipping=args.clipping,
                   repfreq=args.repfreq, checkfreq=args.checkfreq,
                   lr_weight=args.lr_weight)
